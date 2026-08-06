use crate::anthropic::{json_error, schema::MessagesRequest};
use crate::model_setting::ConfiguredAnthropicRoute;
use crate::openai_compat::stream::SseDecoder;
use crate::provider::{CliHandlers, Provider, RequestContext};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use axum::body::Body;
use axum::http::{StatusCode, header};
use axum::response::Response;
use futures_util::StreamExt;
use serde_json::Value;
use std::time::Duration;

pub struct ConfiguredAnthropicProvider {
    route: ConfiguredAnthropicRoute,
    client: reqwest::Client,
}

fn update_usage_from_event(
    event: &Value,
    input_tokens: &mut Option<u64>,
    output_tokens: &mut Option<u64>,
) {
    for usage in [
        event.pointer("/usage"),
        event.pointer("/delta/usage"),
        event.pointer("/message/usage"),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(tokens) = usage.get("input_tokens").and_then(Value::as_u64) {
            *input_tokens = Some(tokens);
        }
        if let Some(tokens) = usage.get("output_tokens").and_then(Value::as_u64) {
            *output_tokens = Some(tokens);
        }
    }
}

impl ConfiguredAnthropicProvider {
    pub fn new(route: ConfiguredAnthropicRoute) -> anyhow::Result<Self> {
        Self::build(route, false)
    }

    #[cfg(test)]
    fn new_for_test(route: ConfiguredAnthropicRoute) -> anyhow::Result<Self> {
        Self::build(route, true)
    }

    fn build(route: ConfiguredAnthropicRoute, disable_proxies: bool) -> anyhow::Result<Self> {
        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10));
        if disable_proxies {
            builder = builder.no_proxy();
        }
        Ok(Self {
            route,
            client: builder.build()?,
        })
    }

    async fn forward_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let body = match self.route.prepare_body(&body) {
            Ok(body) => body,
            Err(_) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "Failed to serialize configured model route request".to_string(),
                );
            }
        };

        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, self.route.model());
            if let Some(effort) = body
                .pointer("/output_config/effort")
                .and_then(Value::as_str)
            {
                monitor.effort_resolved(&ctx.req_id, effort);
            }
            monitor.upstream_started(&ctx.req_id);
        }

        let response = match self
            .client
            .post(self.route.url().clone())
            .headers(self.route.headers().clone())
            .header(header::AUTHORIZATION, self.route.authorization().clone())
            .header(header::ACCEPT, "application/json")
            .header(header::CONTENT_TYPE, "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(_) => {
                return json_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    "Configured model route request failed".to_string(),
                );
            }
        };

        let status = response.status();
        let content_type = response.headers().get(header::CONTENT_TYPE).cloned();
        let monitor = if status.is_success() {
            ctx.monitor.clone()
        } else {
            None
        };
        let req_id = ctx.req_id.clone();
        let mut decoder = Some(SseDecoder::default());
        let mut generation_started = false;
        let stream = response.bytes_stream().map(move |result| {
            if let Ok(chunk) = result.as_ref()
                && !chunk.is_empty()
            {
                let mut input_tokens = None;
                let mut output_tokens = None;
                match decoder.as_mut().map(|decoder| decoder.push(chunk)) {
                    Some(Ok(events)) => {
                        for event in events {
                            update_usage_from_event(
                                &event.data,
                                &mut input_tokens,
                                &mut output_tokens,
                            );
                        }
                    }
                    Some(Err(_)) => decoder = None,
                    None => {}
                }
                if let Some(monitor) = monitor.as_ref() {
                    if !generation_started {
                        monitor.generation_started(&req_id);
                        generation_started = true;
                    }
                    monitor.stream_progress(
                        &req_id,
                        chunk.len() as u64,
                        1,
                        input_tokens,
                        output_tokens,
                    );
                }
            }
            result
        });
        let mut downstream = Response::builder().status(status);
        if let Some(content_type) = content_type {
            downstream = downstream.header(header::CONTENT_TYPE, content_type);
        }
        downstream
            .body(Body::from_stream(stream))
            .unwrap_or_else(|_| {
                json_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    "Configured model route response could not be forwarded".to_string(),
                )
            })
    }
}

#[async_trait]
impl Provider for ConfiguredAnthropicProvider {
    fn name(&self) -> &'static str {
        "model-setting"
    }

    fn supported_models(&self) -> Vec<String> {
        Vec::new()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &CONFIGURED_ANTHROPIC_CLI
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        self.forward_messages(body, ctx).await
    }

    async fn handle_count_tokens(&self, _body: MessagesRequest, _ctx: RequestContext) -> Response {
        json_error(
            StatusCode::NOT_IMPLEMENTED,
            "unsupported_provider_error",
            "Configured model routes do not handle count_tokens".to_string(),
        )
    }
}

struct ConfiguredAnthropicCli;

impl CliHandlers for ConfiguredAnthropicCli {
    fn login(&self) -> Result<()> {
        Err(anyhow!(
            "configured model routes do not provide authentication commands"
        ))
    }

    fn device(&self) -> Result<()> {
        Err(anyhow!(
            "configured model routes do not provide authentication commands"
        ))
    }

    fn status(&self) -> Result<()> {
        Err(anyhow!(
            "configured model routes do not provide authentication commands"
        ))
    }

    fn logout(&self) -> Result<()> {
        Err(anyhow!(
            "configured model routes do not provide authentication commands"
        ))
    }
}

static CONFIGURED_ANTHROPIC_CLI: ConfiguredAnthropicCli = ConfiguredAnthropicCli;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_setting::{ClaudeAliasFamily, ModelSetting};
    use crate::monitor::{EndpointKind, MonitorHandle, Throughput};
    use axum::{
        Json, Router,
        extract::{OriginalUri, State},
        http::HeaderMap,
        response::IntoResponse,
        routing::post,
    };
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Debug)]
    struct SeenRequest {
        uri: String,
        authorization: String,
        client_type: String,
        body: Value,
    }

    type Seen = Arc<Mutex<Vec<SeenRequest>>>;

    const TEST_SSE: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12,\"output_tokens\":0}}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":7}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    async fn capture_request(
        State(seen): State<Seen>,
        OriginalUri(uri): OriginalUri,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> impl IntoResponse {
        seen.lock().unwrap().push(SeenRequest {
            uri: uri.to_string(),
            authorization: headers
                .get(header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_string(),
            client_type: headers
                .get("x-client-type")
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_string(),
            body,
        });
        ([(header::CONTENT_TYPE, "text/event-stream")], TEST_SSE)
    }

    fn request_context() -> RequestContext {
        RequestContext {
            req_id: "request-1".to_string(),
            session_id: None,
            session_seq: None,
            provider: "model-setting".to_string(),
            traffic: None,
            monitor: None,
        }
    }

    #[tokio::test]
    async fn forwards_configured_headers_model_default_effort_query_and_stream() {
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/v1/messages", post(capture_request))
            .with_state(seen.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let setting = ModelSetting::parse(
            &json!({
                "version": 1,
                "routes": {
                    "sonnet": {
                        "url": format!("http://{address}/v1/messages?beta=true"),
                        "apiKey": "test-key",
                        "headers": {"x-client-type": "configured"},
                        "model": "target-model",
                        "defaultEffort": "xhigh",
                        "effortMap": {"max": "xhigh"}
                    }
                }
            })
            .to_string(),
        )
        .unwrap();
        let route = setting
            .into_routes()
            .find_map(|(family, route)| (family == ClaudeAliasFamily::Sonnet).then_some(route))
            .unwrap();
        let provider = ConfiguredAnthropicProvider::new_for_test(route).unwrap();
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }))
        .unwrap();

        let monitor = MonitorHandle::new(8);
        monitor.request_started("request-1", None, None, EndpointKind::Messages);
        let mut context = request_context();
        context.monitor = Some(monitor.clone());

        let response = provider.handle_messages(request, context).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(bytes.as_ref(), TEST_SSE.as_bytes());
        monitor.request_completed("request-1", 200, None, None);

        let snapshot = monitor.snapshot();
        let completed = &snapshot.recent[0];
        assert_eq!(completed.input_tokens, Some(12));
        assert_eq!(completed.output_tokens, Some(7));
        assert_eq!(completed.effort.as_deref(), Some("xhigh"));
        assert_eq!(completed.streamed_bytes, TEST_SSE.len() as u64);
        assert!(completed.stream_chunks > 0);
        assert!(matches!(completed.rate(), Throughput::TokensPerSecond(_)));
        server.abort();

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].uri, "/v1/messages?beta=true");
        assert_eq!(seen[0].authorization, "Bearer test-key");
        assert_eq!(seen[0].client_type, "configured");
        assert_eq!(seen[0].body["model"], "target-model");
        assert_eq!(seen[0].body["output_config"]["effort"], "xhigh");
        assert_eq!(seen[0].body["messages"][0]["content"], "hello");
    }
}
