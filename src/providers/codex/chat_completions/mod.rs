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

use crate::provider::RequestContext;

use super::client::{AuthRejectionBudget, CodexError, CodexHttpClient};
use request::TranslatedRequest;

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
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &request.model);
        }
        let lane_token = ctx.session_id.clone();
        let mut route = match self
            .client
            .conversation_route(request.use_responses_lite)
            .await
        {
            Ok(route) => route,
            Err(error) => return codex_error_response(error),
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.upstream_started(&ctx.req_id);
        }
        let rejection_budget = Arc::new(AuthRejectionBudget::default());
        let (upstream, ctx) = loop {
            let mut route_ctx = ctx.clone();
            if let Some(lane_token) = lane_token.as_deref() {
                route_ctx.session_id = Some(route.bind_lane(lane_token));
            }
            let upstream = match self
                .client
                .post_native_responses_bound_with_rejection_budget(
                    &route,
                    &request.upstream,
                    &route_ctx,
                    true,
                    rejection_budget.clone(),
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
            {
                route = next_route;
                continue;
            }
            break (upstream, route_ctx);
        };

        if !upstream.status().is_success() {
            return upstream_error_response(upstream, self.client.body_idle_timeout_ms()).await;
        }
        if request.stream {
            return stream::streaming_response(
                upstream,
                ctx,
                request.model,
                request.include_usage,
                self.client.body_idle_timeout_ms(),
            );
        }

        let headers = stream::response_headers(upstream.headers());
        let bytes =
            match collect_body(upstream, self.client.body_idle_timeout_ms(), Some(&ctx)).await {
                Ok(bytes) => bytes,
                Err(error) => return error.response(),
            };
        if let Some(traffic) = ctx.traffic.as_deref() {
            traffic.write_bytes("032-upstream-response-body.sse", &bytes);
        }
        let completion = match response::aggregate_sse(&bytes, &request.model) {
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

async fn collect_body(
    upstream: reqwest::Response,
    idle_timeout_ms: u64,
    ctx: Option<&RequestContext>,
) -> Result<Vec<u8>, ChatError> {
    let mut stream = upstream.bytes_stream();
    let mut bytes = Vec::new();
    let mut started = false;
    loop {
        match tokio::time::timeout(Duration::from_millis(idle_timeout_ms), stream.next()).await {
            Ok(Some(Ok(chunk))) => {
                if !started {
                    if let Some(ctx) = ctx
                        && let Some(monitor) = ctx.monitor.as_ref()
                    {
                        monitor.generation_started(&ctx.req_id);
                    }
                    started = true;
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
            Ok(None) => return Ok(bytes),
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
    let bytes = match collect_body(upstream, idle_timeout_ms, None).await {
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
        providers::codex::auth::token_store::StoredAuth,
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

    async fn read_http_body(socket: &mut tokio::net::TcpStream) -> (Vec<u8>, String) {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = socket.read(&mut buffer).await.unwrap();
            assert!(read > 0);
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + length {
                return (
                    request[header_end + 4..header_end + 4 + length].to_vec(),
                    headers.into_owned(),
                );
            }
        }
    }

    async fn read_http_json(socket: &mut tokio::net::TcpStream) -> (Value, String) {
        let (body, headers) = read_http_body(socket).await;
        (serde_json::from_slice(&body).unwrap(), headers)
    }

    async fn mock_backend(
        sse_body: &'static [u8],
    ) -> (
        ChatCompletionsBackend,
        tokio::task::JoinHandle<(Value, String)>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let (body, headers) = read_http_json(&mut socket).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nx-request-id: upstream-1\r\nconnection: close\r\n\r\n",
                sse_body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.write_all(sse_body).await.unwrap();
            (body, headers)
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

        let (upstream, upstream_headers) = server.await.unwrap();
        assert_eq!(upstream["store"], false);
        assert_eq!(upstream["stream"], true);
        assert_eq!(upstream["input"][0]["role"], "developer");
        assert_eq!(upstream["reasoning"]["effort"], "low");
        assert_eq!(upstream["reasoning"]["context"], "all_turns");
        assert_eq!(upstream["text"]["format"]["name"], "answer");
        let bound_session = upstream_headers
            .lines()
            .find_map(|line| line.strip_prefix("session_id: "))
            .unwrap();
        assert_ne!(bound_session, "session");
        assert!(upstream_headers.contains(&format!("x-codex-window-id: {bound_session}:0")));
        assert!(upstream_headers.contains("x-client-request-id: chat-test"));
        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.active[0].input_tokens, Some(8));
        assert_eq!(snapshot.active[0].output_tokens, Some(4));
    }

    #[tokio::test]
    async fn chat_header_401_rebuilds_same_account_route_once() {
        const SSE: &[u8] = b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_chat_b\",\"model\":\"gpt-5.4\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}}\n\n";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        );
        client.auth_manager().set_test_auth(StoredAuth {
            access: "chat-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("chat-account".into()),
            expires: u64::MAX,
        });
        let backend = ChatCompletionsBackend::with_client(client);
        let server_client = backend.client.clone();
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(read_http_json(&mut socket).await);
                if attempt == 0 {
                    server_client.auth_manager().set_test_auth(StoredAuth {
                        access: "chat-b".into(),
                        refresh: "refresh-b".into(),
                        account_id: Some("chat-account".into()),
                        expires: u64::MAX,
                    });
                    socket
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 5\r\nconnection: close\r\n\r\nstale",
                        )
                        .await
                        .unwrap();
                } else {
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        SSE.len()
                    );
                    socket.write_all(head.as_bytes()).await.unwrap();
                    socket.write_all(SSE).await.unwrap();
                }
            }
            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err()
            );
            captured
        });
        let translated = request::translate_request(json!({
            "model":"gpt-5.4",
            "messages":[{"role":"user","content":"unchanged"}],
            "stream":false
        }))
        .unwrap();
        let response = backend
            .handle(translated, context(MonitorHandle::new(10)))
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        let captured = server.await.unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].0, captured[1].0);
        assert!(captured[0].1.contains("authorization: Bearer chat-a"));
        assert!(captured[1].1.contains("authorization: Bearer chat-b"));
        let bound_session = |headers: &str| {
            headers
                .lines()
                .find_map(|line| line.strip_prefix("session_id: "))
                .unwrap()
                .to_string()
        };
        let session_a = bound_session(&captured[0].1);
        let session_b = bound_session(&captured[1].1);
        assert_ne!(session_a, session_b);
        assert!(
            captured[0]
                .1
                .contains(&format!("x-codex-window-id: {session_a}:0"))
        );
        assert!(
            captured[1]
                .1
                .contains(&format!("x-codex-window-id: {session_b}:0"))
        );
    }

    #[tokio::test]
    async fn chat_second_401_returns_route_b_without_third_send() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        );
        client.auth_manager().set_test_auth(StoredAuth {
            access: "chat-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("chat-account".into()),
            expires: u64::MAX,
        });
        let backend = ChatCompletionsBackend::with_client(client);
        let server_client = backend.client.clone();
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(read_http_json(&mut socket).await);
                server_client.auth_manager().set_test_auth(StoredAuth {
                    access: if attempt == 0 { "chat-b" } else { "chat-c" }.into(),
                    refresh: format!("refresh-{}", attempt + 2),
                    account_id: Some("chat-account".into()),
                    expires: u64::MAX,
                });
                let body = if attempt == 0 {
                    br#"{"error":{"message":"chat route a"}}"#.as_slice()
                } else {
                    br#"{"error":{"message":"chat route b"}}"#.as_slice()
                };
                let head = format!(
                    "HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(body).await.unwrap();
            }
            (
                tokio::time::timeout(Duration::from_millis(100), listener.accept()).await,
                captured,
            )
        });
        let translated = request::translate_request(json!({
            "model":"gpt-5.4",
            "messages":[{"role":"user","content":"unchanged"}],
            "stream":false
        }))
        .unwrap();
        let response = backend
            .handle(translated, context(MonitorHandle::new(10)))
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("chat route b"));
        assert_eq!(backend.client.route_rejection_refresh_count(), 1);

        let (third, captured) = server.await.unwrap();
        assert!(third.is_err(), "chat sent route C");
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].0, captured[1].0);
        assert!(captured[0].1.contains("authorization: Bearer chat-a"));
        assert!(captured[1].1.contains("authorization: Bearer chat-b"));
    }

    #[tokio::test]
    async fn chat_in_band_401_maps_buffered_http_status_but_streams_in_band() {
        const SSE: &[u8] = b"data: {\"type\":\"response.failed\",\"status_code\":401,\"response\":{\"error\":{\"status\":401,\"message\":\"expired in band\"}}}\n\n";

        let (buffered_backend, buffered_server) = mock_backend(SSE).await;
        let buffered = request::translate_request(json!({
            "model":"gpt-5.4",
            "messages":[{"role":"user","content":"buffered auth"}],
            "stream":false
        }))
        .unwrap();
        let response = buffered_backend
            .handle(buffered, context(MonitorHandle::new(10)))
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
        buffered_server.await.unwrap();

        let (streaming_backend, streaming_server) = mock_backend(SSE).await;
        let streaming = request::translate_request(json!({
            "model":"gpt-5.4",
            "messages":[{"role":"user","content":"streaming auth"}],
            "stream":true
        }))
        .unwrap();
        let response = streaming_backend
            .handle(streaming, context(MonitorHandle::new(10)))
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8_lossy(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .into_owned();
        assert!(body.contains("expired in band"));
        assert!(body.ends_with("data: [DONE]\n\n"));
        streaming_server.await.unwrap();
    }

    #[tokio::test]
    async fn chat_header_then_in_band_401_uses_one_total_refresh_budget() {
        const ROUTE_B: &[u8] = b"data: {\"type\":\"response.failed\",\"status_code\":401,\"response\":{\"error\":{\"status\":401,\"message\":\"chat route b in-band\"}}}\n\n";
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_address = upstream_listener.local_addr().unwrap();
        let token_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let token_address = token_listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{upstream_address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_auth_token_endpoint(format!("http://{token_address}/oauth/token"));
        client.auth_manager().set_test_auth(StoredAuth {
            access: "chat-mixed-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("chat-mixed-account".into()),
            expires: u64::MAX,
        });
        let backend = ChatCompletionsBackend::with_client(client);

        let token_server = tokio::spawn(async move {
            let (mut socket, _) = token_listener.accept().await.unwrap();
            let (request_body, request_headers) = read_http_body(&mut socket).await;
            assert!(request_headers.starts_with("POST "));
            assert!(String::from_utf8_lossy(&request_body).contains("refresh_token=refresh-a"));
            let body =
                br#"{"access_token":"chat-mixed-b","refresh_token":"refresh-b","expires_in":3600}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(body).await.unwrap();
            tokio::time::timeout(Duration::from_millis(150), token_listener.accept()).await
        });
        let upstream_server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..2 {
                let (mut socket, _) = upstream_listener.accept().await.unwrap();
                captured.push(read_http_json(&mut socket).await);
                if attempt == 0 {
                    socket
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 7\r\nconnection: close\r\n\r\nroute-a",
                        )
                        .await
                        .unwrap();
                } else {
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        ROUTE_B.len()
                    );
                    socket.write_all(head.as_bytes()).await.unwrap();
                    socket.write_all(ROUTE_B).await.unwrap();
                }
            }
            captured
        });

        let translated = request::translate_request(json!({
            "model":"gpt-5.4",
            "messages":[{"role":"user","content":"mixed form"}],
            "stream":false
        }))
        .unwrap();
        let response = backend
            .handle(translated, context(MonitorHandle::new(10)))
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let value: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(value["error"]["type"], "authentication_error");
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("route b")
        );
        assert_eq!(backend.client.route_rejection_refresh_count(), 1);
        assert_eq!(
            backend
                .client
                .auth_manager()
                .get_auth()
                .await
                .unwrap()
                .access,
            "chat-mixed-b"
        );
        assert!(
            token_server.await.unwrap().is_err(),
            "route B in-band 401 attempted a second refresh"
        );
        let captured = upstream_server.await.unwrap();
        assert_eq!(captured.len(), 2);
        assert!(captured[0].1.contains("authorization: Bearer chat-mixed-a"));
        assert!(captured[1].1.contains("authorization: Bearer chat-mixed-b"));
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
