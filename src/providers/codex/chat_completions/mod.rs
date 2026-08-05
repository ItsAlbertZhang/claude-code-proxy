pub mod request;
pub mod response;
pub mod stream;

use std::{sync::Arc, time::Duration};

use axum::{
    Json,
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use http::StatusCode;
use serde_json::{Value, json};

use crate::openai_compat::{MAX_PROVIDER_STREAM_BYTES, MAX_SSE_EVENT_BYTES};
use crate::provider::{RequestContext, ScopedRequestContext, legacy_scope};
use crate::request_identity::{LaneDomain, RequestPurpose};

use super::client::{AuthRejectionBudget, CodexError, CodexHttpClient, InBandAuthRefreshDetector};
use super::state::{CodexBoundRoute, ProtocolLane};
use request::TranslatedRequest;

pub(super) const MAX_CHAT_UPSTREAM_BYTES: usize = MAX_PROVIDER_STREAM_BYTES;
pub(super) const MAX_CHAT_SSE_FRAME_BYTES: usize = MAX_SSE_EVENT_BYTES;
pub(super) const MAX_CHAT_OUTPUT_BYTES: usize = MAX_PROVIDER_STREAM_BYTES;

pub struct ChatCompletionsBackend {
    client: Arc<CodexHttpClient>,
}

impl Default for ChatCompletionsBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatCompletionsBackend {
    pub fn new() -> Self {
        Self {
            client: Arc::new(CodexHttpClient::new()),
        }
    }

    #[cfg(test)]
    fn with_client(client: CodexHttpClient) -> Self {
        Self {
            client: Arc::new(client),
        }
    }

    pub async fn handle(&self, request: TranslatedRequest, ctx: RequestContext) -> Response {
        let scope = legacy_scope(&ctx, RequestPurpose::Conversation);
        self.handle_scoped(request, ScopedRequestContext::new(ctx, scope))
            .await
    }

    pub(crate) async fn handle_scoped(
        &self,
        request: TranslatedRequest,
        scoped: ScopedRequestContext,
    ) -> Response {
        let (ctx, scope) = scoped.into_parts();
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &request.model);
            monitor.codex_request_lane(&ctx.req_id, request.use_responses_lite);
        }
        let lane = scope.provider_lane(LaneDomain::CodexConversation);
        let protocol = ProtocolLane::from_uses_responses_lite(request.use_responses_lite);
        let mut route = match self.client.bind_conversation_route(lane, protocol).await {
            Ok(route) => route,
            Err(error) => return codex_error_response(error),
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.upstream_started(&ctx.req_id);
        }
        let rejection_budget = Arc::new(AuthRejectionBudget::default());
        let upstream = loop {
            let bound_request = bind_chat_request_to_route(&request.upstream, &route);
            let upstream = match self
                .client
                .post_native_responses_bound(
                    &route,
                    &bound_request,
                    &ctx,
                    request.use_responses_lite,
                    true,
                )
                .await
            {
                Ok(upstream) => upstream,
                Err(error) => return codex_error_response(error),
            };
            if upstream.status() == StatusCode::UNAUTHORIZED
                && rejection_budget.try_claim()
                && let Some(next_route) = self
                    .client
                    .refresh_conversation_route_after_rejection(&route)
                    .await
                    .into_route()
            {
                route = next_route;
                continue;
            }
            break upstream;
        };

        if !upstream.status().is_success() {
            return upstream_error_response(upstream, self.client.body_idle_timeout_ms()).await;
        }
        let auth_refresh =
            InBandAuthRefreshDetector::sse(self.client.clone(), route, rejection_budget);
        if request.stream {
            return stream::streaming_response_with_auth_refresh(
                upstream,
                ctx,
                request.model,
                request.include_usage,
                self.client.body_idle_timeout_ms(),
                auth_refresh,
            );
        }

        let headers = stream::response_headers(upstream.headers());
        let completion = match aggregate_completion_body(
            upstream,
            self.client.body_idle_timeout_ms(),
            &ctx,
            auth_refresh,
            &request.model,
        )
        .await
        {
            Ok(completion) => completion,
            Err(error) => return error.response(),
        };
        if let Some(usage) = completion.get("usage")
            && let Some(monitor) = ctx.monitor.as_ref()
        {
            monitor.usage_updated(
                &ctx.req_id,
                usage.get("prompt_tokens").and_then(Value::as_u64),
                usage.get("completion_tokens").and_then(Value::as_u64),
            );
        }
        if let Some(traffic) = ctx.traffic.as_deref() {
            traffic.write_json("050-openai-chat-completion-response", &completion);
        }
        let mut downstream = Json(completion).into_response();
        *downstream.headers_mut() = headers;
        downstream.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        downstream
    }
}

fn bind_chat_request_to_route(body: &Value, route: &CodexBoundRoute) -> Value {
    let mut bound = body.clone();
    let Some(caller_key) = body.get("prompt_cache_key").and_then(Value::as_str) else {
        return bound;
    };
    let Some(namespaced) = route.namespace_prompt_cache_key(caller_key) else {
        return bound;
    };
    if let Some(object) = bound.as_object_mut() {
        object.insert("prompt_cache_key".to_string(), Value::String(namespaced));
    }
    bound
}

async fn aggregate_completion_body(
    upstream: reqwest::Response,
    idle_timeout_ms: u64,
    ctx: &RequestContext,
    mut auth_refresh: InBandAuthRefreshDetector,
    model: &str,
) -> Result<Value, ChatError> {
    let mut upstream = upstream.bytes_stream();
    let mut decoder = stream::ChatSseDecoder::default();
    let mut completion = response::CompletionState::new(model);
    let mut raw = Vec::new();
    let mut started = false;

    loop {
        match tokio::time::timeout(Duration::from_millis(idle_timeout_ms), upstream.next()).await {
            Ok(Some(Ok(chunk))) => {
                auth_refresh.observe(&chunk);
                if !started {
                    if let Some(monitor) = ctx.monitor.as_ref() {
                        monitor.generation_started(&ctx.req_id);
                    }
                    started = true;
                }
                let remaining = crate::traffic::MAX_SSE_CAPTURE_BYTES.saturating_sub(raw.len());
                raw.extend_from_slice(&chunk[..remaining.min(chunk.len())]);
                let events = decoder.observe(&chunk)?;
                if let Some(monitor) = ctx.monitor.as_ref() {
                    monitor.stream_progress(
                        &ctx.req_id,
                        chunk.len() as u64,
                        events.len() as u64,
                        Some(completion.usage.prompt_tokens),
                        Some(completion.usage.completion_tokens),
                    );
                }
                for data in events {
                    if data == "[DONE]" {
                        continue;
                    }
                    let event: Value = serde_json::from_str(&data).map_err(|_| {
                        ChatError::upstream("Codex returned malformed JSON in its event stream")
                    })?;
                    if let Some(traffic) = ctx.traffic.as_deref() {
                        traffic.write_json_event("040-upstream-event", &event);
                    }
                    completion.observe(&event)?;
                }
            }
            Ok(Some(Err(error))) => {
                return Err(ChatError::upstream(format!(
                    "Codex response body read failed: {error}"
                )));
            }
            Ok(None) => {
                auth_refresh.finish();
                decoder.finish()?;
                break;
            }
            Err(_) => {
                return Err(ChatError::timeout(format!(
                    "Timed out waiting {idle_timeout_ms}ms for the next Codex response body chunk"
                )));
            }
        }
    }

    if let Some(traffic) = ctx.traffic.as_deref()
        && !raw.is_empty()
    {
        traffic.write_bytes("032-upstream-response-body.sse", &raw);
    }
    if !completion.completed {
        return Err(ChatError::upstream(
            "Codex event stream ended before completion",
        ));
    }
    if !completion.has_output_text() {
        return Err(ChatError::upstream("Codex completed without output text"));
    }
    let value = response::completion_value(&completion);
    if serde_json::to_vec(&value).map_or(true, |body| body.len() > MAX_CHAT_OUTPUT_BYTES) {
        return Err(ChatError::upstream(
            "Codex translated output exceeded the configured limit",
        ));
    }
    Ok(value)
}

async fn collect_body(
    upstream: reqwest::Response,
    idle_timeout_ms: u64,
    ctx: Option<&RequestContext>,
    mut auth_refresh: Option<InBandAuthRefreshDetector>,
) -> Result<Vec<u8>, ChatError> {
    let mut stream = upstream.bytes_stream();
    let mut bytes = Vec::new();
    let mut started = false;
    loop {
        match tokio::time::timeout(Duration::from_millis(idle_timeout_ms), stream.next()).await {
            Ok(Some(Ok(chunk))) => {
                if let Some(auth_refresh) = auth_refresh.as_mut() {
                    auth_refresh.observe(&chunk);
                }
                if !started {
                    if let Some(ctx) = ctx
                        && let Some(monitor) = ctx.monitor.as_ref()
                    {
                        monitor.generation_started(&ctx.req_id);
                    }
                    started = true;
                }
                if bytes.len().saturating_add(chunk.len()) > MAX_CHAT_UPSTREAM_BYTES {
                    return Err(ChatError::upstream(
                        "Codex response body exceeded the configured limit",
                    ));
                }
                bytes.extend_from_slice(&chunk);
                if let Some(ctx) = ctx
                    && let Some(monitor) = ctx.monitor.as_ref()
                {
                    monitor.stream_progress(&ctx.req_id, chunk.len() as u64, 0, None, None);
                }
            }
            Ok(Some(Err(error))) => {
                return Err(ChatError::upstream(format!(
                    "Codex response body read failed: {error}"
                )));
            }
            Ok(None) => {
                if let Some(auth_refresh) = auth_refresh.as_mut() {
                    auth_refresh.finish();
                }
                return Ok(bytes);
            }
            Err(_) => {
                return Err(ChatError::timeout(format!(
                    "Timed out waiting {idle_timeout_ms}ms for the next Codex response body chunk"
                )));
            }
        }
    }
}

fn codex_error_response(error: CodexError) -> Response {
    let retry_after = error.retry_after.clone();
    let response = ChatError::from_codex(error).response();
    if let Some(retry_after) = retry_after
        && let Ok(value) = http::HeaderValue::from_str(&retry_after)
    {
        let (mut parts, body) = response.into_parts();
        parts.headers.insert(http::header::RETRY_AFTER, value);
        Response::from_parts(parts, body)
    } else {
        response
    }
}

async fn upstream_error_response(upstream: reqwest::Response, idle_timeout_ms: u64) -> Response {
    let status = upstream.status();
    let retry_after = upstream.headers().get(http::header::RETRY_AFTER).cloned();
    let bytes = match collect_body(upstream, idle_timeout_ms, None, None).await {
        Ok(bytes) => bytes,
        Err(error) => return error.response(),
    };
    let message = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .or_else(|| value.get("message"))
                .or_else(|| value.get("detail"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| format!("Codex request failed with status {}", status.as_u16()));
    let kind = match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        _ => "api_error",
    };
    let response = ChatError::new(status, kind, message, None, None).response();
    if let Some(retry_after) = retry_after {
        let (mut parts, body) = response.into_parts();
        parts.headers.insert(http::header::RETRY_AFTER, retry_after);
        Response::from_parts(parts, body)
    } else {
        response
    }
}

#[derive(Debug, Clone)]
pub struct ChatError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub message: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ChatError {
    pub fn new(
        status: StatusCode,
        kind: &'static str,
        message: impl Into<String>,
        param: Option<&str>,
        code: Option<&str>,
    ) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
            param: param.map(str::to_string),
            code: code.map(str::to_string),
        }
    }

    pub fn invalid(message: impl Into<String>, param: Option<&str>, code: Option<&str>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
            param,
            code,
        )
    }

    pub fn unsupported(param: impl Into<String>) -> Self {
        let param = param.into();
        Self::invalid(
            format!("Unsupported parameter: {param}"),
            Some(&param),
            Some("unsupported_parameter"),
        )
    }

    pub fn upstream(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, "api_error", message, None, None)
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::GATEWAY_TIMEOUT,
            "api_error",
            message,
            None,
            None,
        )
    }

    fn from_codex(error: CodexError) -> Self {
        let status = match error.status {
            401 => StatusCode::UNAUTHORIZED,
            403 => StatusCode::FORBIDDEN,
            429 => StatusCode::TOO_MANY_REQUESTS,
            _ if error.message.contains("Timed out waiting") => StatusCode::GATEWAY_TIMEOUT,
            _ => StatusCode::BAD_GATEWAY,
        };
        let kind = match status {
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::FORBIDDEN => "permission_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            _ => "api_error",
        };
        Self::new(
            status,
            kind,
            error.detail.unwrap_or(error.message),
            None,
            None,
        )
    }

    pub fn value(&self) -> Value {
        json!({"error":{"message":self.message,"type":self.kind,"param":self.param,"code":self.code}})
    }

    pub fn response(self) -> Response {
        (self.status, Json(self.value())).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        monitor::{EndpointKind, MonitorHandle},
        provider::ScopedRequestContext,
        providers::codex::auth::token_store::StoredAuth,
        request_identity::{ConversationIdentity, RequestScope},
    };
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    fn context(monitor: MonitorHandle) -> RequestContext {
        monitor.request_started(
            "chat-test",
            Some("session".into()),
            None,
            EndpointKind::ChatCompletions,
        );
        RequestContext {
            req_id: "chat-test".into(),
            session_id: Some("session".into()),
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: Some(monitor),
        }
    }

    fn plain_context(req_id: &str) -> RequestContext {
        RequestContext {
            req_id: req_id.into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        }
    }

    fn scoped_context(identity: ConversationIdentity, req_id: &str) -> ScopedRequestContext {
        ScopedRequestContext::new(
            plain_context(req_id),
            RequestScope::from_conversation_identity(Some(identity), RequestPurpose::Conversation),
        )
    }

    fn translated_request(stream: bool) -> TranslatedRequest {
        request::translate_request(json!({
            "model":"gpt-5.4",
            "messages":[{"role":"user","content":"unchanged message"}],
            "stream":stream
        }))
        .unwrap()
    }

    fn test_backend(address: std::net::SocketAddr, auth: StoredAuth) -> ChatCompletionsBackend {
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        );
        client.auth_manager().set_test_auth(auth);
        ChatCompletionsBackend::with_client(client)
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> (String, Value) {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = socket.read(&mut buffer).await.unwrap();
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")?
                        .trim()
                        .parse::<usize>()
                        .ok()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + length {
                break;
            }
        }
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap();
        (
            String::from_utf8_lossy(&request[..header_end]).to_string(),
            serde_json::from_slice(&request[header_end + 4..]).unwrap(),
        )
    }

    async fn write_sse_response(socket: &mut tokio::net::TcpStream, body: &[u8]) {
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(head.as_bytes()).await.unwrap();
        socket.write_all(body).await.unwrap();
    }

    const COMPLETED_SSE: &[u8] = b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_ok\",\"model\":\"gpt-5.4\",\"status\":\"completed\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";

    async fn mock_backend(
        sse_body: &'static [u8],
    ) -> (ChatCompletionsBackend, tokio::task::JoinHandle<Value>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = socket.read(&mut buffer).await.unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
                else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")?
                            .trim()
                            .parse::<usize>()
                            .ok()
                    })
                    .unwrap_or(0);
                if request.len() >= header_end + 4 + length {
                    break;
                }
            }
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .unwrap();
            let body: Value = serde_json::from_slice(&request[header_end + 4..]).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nx-request-id: upstream-1\r\nconnection: close\r\n\r\n",
                sse_body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.write_all(sse_body).await.unwrap();
            body
        });
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::new(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        );
        client.auth_manager().set_test_auth(StoredAuth {
            access: "test-token".into(),
            refresh: String::new(),
            account_id: Some("account".into()),
            expires: u64::MAX,
        });
        (ChatCompletionsBackend::with_client(client), server)
    }

    #[tokio::test]
    async fn buffered_request_translates_upstream_and_downstream() {
        const SSE: &[u8] = b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"{\\\"answer\\\":\\\"yes\\\"}\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_buffered\",\"model\":\"gpt-5.6-sol\",\"status\":\"completed\",\"usage\":{\"input_tokens\":8,\"output_tokens\":4}}}\n\n";
        let (backend, server) = mock_backend(SSE).await;
        let request = request::translate_request(json!({
            "model":"gpt-5.6-sol",
            "messages":[{"role":"system","content":"JSON only"},{"role":"user","content":"answer"}],
            "reasoning_effort":"low",
            "response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object"}}}
        })).unwrap();
        let monitor = MonitorHandle::new(10);
        let response = backend.handle(request, context(monitor.clone())).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-request-id"], "upstream-1");
        let value: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(value["object"], "chat.completion");
        assert_eq!(
            value["choices"][0]["message"]["content"],
            r#"{"answer":"yes"}"#
        );
        assert_eq!(value["usage"]["total_tokens"], 12);

        let upstream = server.await.unwrap();
        assert_eq!(upstream["store"], false);
        assert_eq!(upstream["stream"], true);
        assert_eq!(upstream["input"][0]["role"], "developer");
        assert_eq!(upstream["reasoning"]["effort"], "low");
        assert_eq!(upstream["reasoning"]["context"], "all_turns");
        assert_eq!(upstream["text"]["format"]["name"], "answer");
        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.active[0].input_tokens, Some(8));
        assert_eq!(snapshot.active[0].output_tokens, Some(4));
    }

    #[tokio::test]
    async fn streaming_request_emits_chat_chunks_usage_and_done() {
        const SSE: &[u8] = b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_stream\",\"model\":\"gpt-5.6-sol\"}}\n\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_stream\",\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n";
        let (backend, server) = mock_backend(SSE).await;
        let request = request::translate_request(json!({
            "model":"gpt-5.6-sol",
            "messages":[{"role":"user","content":"hello"}],
            "stream":true,
            "stream_options":{"include_usage":true}
        }))
        .unwrap();
        let response = backend
            .handle(request, context(MonitorHandle::new(10)))
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let body = String::from_utf8(
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains(r#""delta":{"role":"assistant"}"#));
        assert!(body.contains(r#""delta":{"content":"hello"}"#));
        assert!(body.contains(r#""finish_reason":"stop""#));
        assert!(body.contains(r#""prompt_tokens":3"#));
        assert!(body.ends_with("data: [DONE]\n\n"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn chat_scoped_routes_keep_sibling_agents_opaque_and_distinct() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = test_backend(
            listener.local_addr().unwrap(),
            StoredAuth {
                access: "chat-sibling-token".into(),
                refresh: String::new(),
                account_id: Some("chat-sibling-account".into()),
                expires: u64::MAX,
            },
        );
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(read_http_request(&mut socket).await);
                write_sse_response(&mut socket, COMPLETED_SSE).await;
            }
            captured
        });

        for agent in ["agent-a", "agent-b"] {
            let response = backend
                .handle_scoped(
                    translated_request(false),
                    scoped_context(
                        ConversationIdentity::Agent("shared-session".into(), agent.into()),
                        agent,
                    ),
                )
                .await;
            assert_eq!(response.status(), StatusCode::OK);
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
        }

        let captured = server.await.unwrap();
        let bound_session = |headers: &str| {
            headers
                .lines()
                .find_map(|line| line.strip_prefix("session_id: "))
                .unwrap()
                .to_string()
        };
        assert_ne!(bound_session(&captured[0].0), bound_session(&captured[1].0));
        for (headers, body) in captured {
            let wire = format!("{headers}{body}");
            for raw in ["shared-session", "agent-a", "agent-b"] {
                assert!(!wire.contains(raw));
            }
        }
    }

    #[tokio::test]
    async fn chat_header_401_rebuilds_once_from_original_messages_and_cache_key() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = test_backend(
            listener.local_addr().unwrap(),
            StoredAuth {
                access: "chat-a".into(),
                refresh: "refresh-a".into(),
                account_id: Some("chat-account".into()),
                expires: u64::MAX,
            },
        );
        let server_client = backend.client.clone();
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(read_http_request(&mut socket).await);
                if attempt == 0 {
                    server_client.auth_manager().set_test_auth(StoredAuth {
                        access: "chat-b".into(),
                        refresh: "refresh-b".into(),
                        account_id: Some("chat-account".into()),
                        expires: u64::MAX,
                    });
                    socket
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 7\r\nconnection: close\r\n\r\nroute-a",
                        )
                        .await
                        .unwrap();
                } else {
                    write_sse_response(&mut socket, COMPLETED_SSE).await;
                }
            }
            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err()
            );
            captured
        });
        let identity = ConversationIdentity::Main("chat-session".into());
        let mut request = translated_request(false);
        request.upstream["prompt_cache_key"] = json!("caller-cache");
        let response = backend
            .handle_scoped(request, scoped_context(identity.clone(), "chat-rebind"))
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        let captured = server.await.unwrap();
        assert_eq!(captured.len(), 2);
        assert!(captured[0].0.contains("Bearer chat-a"));
        assert!(captured[1].0.contains("Bearer chat-b"));
        assert_eq!(captured[0].1["input"], captured[1].1["input"]);
        assert_ne!(
            captured[0].1["prompt_cache_key"],
            captured[1].1["prompt_cache_key"]
        );
        let scope =
            RequestScope::from_conversation_identity(Some(identity), RequestPurpose::Conversation);
        let route_b = backend
            .client
            .bind_conversation_route(
                scope.provider_lane(LaneDomain::CodexConversation),
                ProtocolLane::ResponsesFull,
            )
            .await
            .unwrap();
        assert_eq!(
            captured[1].1["prompt_cache_key"],
            route_b.namespace_prompt_cache_key("caller-cache").unwrap()
        );
    }

    #[tokio::test]
    async fn chat_route_b_401_cannot_refresh_or_send_route_c() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = test_backend(
            listener.local_addr().unwrap(),
            StoredAuth {
                access: "chat-route-a".into(),
                refresh: "refresh-a".into(),
                account_id: Some("chat-route-account".into()),
                expires: u64::MAX,
            },
        );
        let server_client = backend.client.clone();
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(read_http_request(&mut socket).await);
                let next = if attempt == 0 {
                    ("chat-route-b", "refresh-b")
                } else {
                    ("chat-route-c", "refresh-c")
                };
                server_client.auth_manager().set_test_auth(StoredAuth {
                    access: next.0.into(),
                    refresh: next.1.into(),
                    account_id: Some("chat-route-account".into()),
                    expires: u64::MAX,
                });
                let body = if attempt == 0 { b"route-a" } else { b"route-b" };
                let head = format!(
                    "HTTP/1.1 401 Unauthorized\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(body).await.unwrap();
            }
            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err()
            );
            captured
        });

        let response = backend
            .handle_scoped(
                translated_request(false),
                scoped_context(
                    ConversationIdentity::Main("route-session".into()),
                    "chat-route-b",
                ),
            )
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let value: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(value["error"]["type"], "authentication_error");
        let captured = server.await.unwrap();
        assert_eq!(captured.len(), 2);
        assert!(captured[0].0.contains("Bearer chat-route-a"));
        assert!(captured[1].0.contains("Bearer chat-route-b"));
    }

    #[tokio::test]
    async fn chat_buffered_in_band_401_maps_auth_error_without_replay() {
        const AUTH_SSE: &[u8] = b"data: {\"type\":\"response.failed\",\"status_code\":401,\"response\":{\"error\":{\"status\":401,\"message\":\"expired in band\"}}}\n\n";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = test_backend(
            listener.local_addr().unwrap(),
            StoredAuth {
                access: "chat-in-band".into(),
                refresh: String::new(),
                account_id: Some("chat-in-band-account".into()),
                expires: u64::MAX,
            },
        );
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let captured = read_http_request(&mut socket).await;
            write_sse_response(&mut socket, AUTH_SSE).await;
            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err()
            );
            captured
        });

        let response = backend
            .handle(translated_request(false), plain_context("chat-in-band"))
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let value: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(value["error"]["type"], "authentication_error");
        assert_eq!(value["error"]["message"], "expired in band");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn chat_streaming_in_band_401_stays_200_and_ends_with_done() {
        const AUTH_SSE: &[u8] = b"data: {\"type\":\"response.failed\",\"status_code\":401,\"response\":{\"error\":{\"status\":401,\"message\":\"stream expired\"}}}\n\n";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = test_backend(
            listener.local_addr().unwrap(),
            StoredAuth {
                access: "chat-stream".into(),
                refresh: String::new(),
                account_id: Some("chat-stream-account".into()),
                expires: u64::MAX,
            },
        );
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let captured = read_http_request(&mut socket).await;
            write_sse_response(&mut socket, AUTH_SSE).await;
            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err()
            );
            captured
        });

        let response = backend
            .handle(translated_request(true), plain_context("chat-stream"))
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains(r#""type":"authentication_error""#));
        assert!(body.contains("stream expired"));
        assert!(body.ends_with("data: [DONE]\n\n"));
        server.await.unwrap();
    }

    #[test]
    fn codex_errors_map_status_and_preserve_retry_metadata() {
        let response = codex_error_response(CodexError {
            status: 429,
            message: "Rate limited".into(),
            detail: Some("Try later".into()),
            retry_after: Some("7".into()),
            origin: super::super::client::CodexErrorOrigin::Http,
        });
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[http::header::RETRY_AFTER], "7");

        let auth = ChatError::from_codex(CodexError {
            status: 401,
            message: "Auth error".into(),
            detail: None,
            retry_after: None,
            origin: super::super::client::CodexErrorOrigin::Auth,
        });
        assert_eq!(auth.status, StatusCode::UNAUTHORIZED);
        assert_eq!(auth.kind, "authentication_error");
    }
}
