use std::io;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use http::{HeaderMap, HeaderName, StatusCode};
use serde_json::{Map, Value, json};

use crate::anthropic::sse::parse_sse_events;
use crate::provider::{RequestContext, ScopedRequestContext, legacy_scope};
use crate::request_identity::{LaneDomain, RequestPurpose};
use crate::traffic::{
    MAX_SSE_CAPTURE_BYTES, MAX_STREAM_CAPTURE_EVENT_BYTES, MAX_STREAM_CAPTURE_EVENTS,
    MAX_STREAM_CAPTURE_FRAME_BYTES,
};

use super::client::{AuthRejectionBudget, CodexError, CodexHttpClient, InBandAuthRefreshDetector};
use super::state::{CodexBoundRoute, ProtocolLane};
use super::translate::model_allowlist::{
    ALLOWED_MODELS, MODEL_ALIASES, assert_allowed_model, full_lane_web_search_model,
    uses_responses_lite,
};

pub struct CodexNativeBackend {
    client: Arc<CodexHttpClient>,
}

impl Default for CodexNativeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexNativeBackend {
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

    pub async fn handle(&self, body: Value, ctx: RequestContext) -> Response {
        let scope = legacy_scope(&ctx, RequestPurpose::Conversation);
        self.handle_scoped(body, ScopedRequestContext::new(ctx, scope))
            .await
    }

    pub(crate) async fn handle_scoped(
        &self,
        mut body: Value,
        scoped: ScopedRequestContext,
    ) -> Response {
        let (ctx, scope) = scoped.into_parts();
        let resolved = match shape_native_request(&mut body) {
            Ok(resolved) => resolved,
            Err(response) => return response,
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &resolved.model);
            monitor.codex_request_lane(&ctx.req_id, resolved.use_responses_lite);
        }
        let lane = scope.provider_lane(LaneDomain::CodexConversation);
        let protocol = ProtocolLane::from_uses_responses_lite(resolved.use_responses_lite);
        let mut route = match self.client.bind_conversation_route(lane, protocol).await {
            Ok(route) => route,
            Err(error) => return local_codex_error(error),
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.upstream_started(&ctx.req_id);
        }
        let rejection_budget = Arc::new(AuthRejectionBudget::default());
        let can_replay_after_rebind = body.get("previous_response_id").is_none_or(Value::is_null);

        loop {
            let bound_body = bind_native_request_to_route(&body, &route);
            let upstream = match self
                .client
                .post_native_responses_bound(
                    &route,
                    &bound_body,
                    &ctx,
                    resolved.use_responses_lite,
                    resolved.stream,
                )
                .await
            {
                Ok(response) => response,
                Err(error) => return local_codex_error(error),
            };

            if upstream.status() == StatusCode::UNAUTHORIZED && rejection_budget.try_claim() {
                let next_route = self
                    .client
                    .refresh_conversation_route_after_rejection(&route)
                    .await
                    .into_route();
                if can_replay_after_rebind && let Some(next_route) = next_route {
                    route = next_route;
                    continue;
                }
            }

            return passthrough_response(
                upstream,
                ctx,
                self.client.body_idle_timeout_ms(),
                self.client.clone(),
                route,
                rejection_budget,
            );
        }
    }
}

struct NativeResolved {
    model: String,
    use_responses_lite: bool,
    stream: bool,
}

#[allow(clippy::result_large_err)]
pub fn validate_native_request_model(body: &Value) -> Result<String, Response> {
    let object = body.as_object().ok_or_else(|| {
        openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Request body must be a JSON object",
            None,
            None,
        )
    })?;
    let requested = object
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            openai_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Missing or invalid 'model' in request body",
                Some("model"),
                None,
            )
        })?;
    let (resolved, _) = resolve_native_model(&requested);
    if let Err(error) = assert_allowed_model(&resolved) {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "Model '{requested}' resolves to unsupported model '{}'. Supported: {}",
                error.model,
                ALLOWED_MODELS.join(", ")
            ),
            Some("model"),
            Some("model_not_supported"),
        ));
    }
    Ok(requested)
}

#[allow(clippy::result_large_err)]
fn shape_native_request(body: &mut Value) -> Result<NativeResolved, Response> {
    let requested = validate_native_request_model(body)?;
    let object = body
        .as_object_mut()
        .expect("validated native Responses body must be an object");

    let (mut model, priority) = resolve_native_model(&requested);

    let hosted_web_search = has_native_hosted_web_search(object);
    if hosted_web_search {
        model = full_lane_web_search_model(&model).to_string();
    }
    let use_responses_lite = uses_responses_lite(&model) && !hosted_web_search;
    object.insert("model".to_string(), Value::String(model.clone()));
    if use_responses_lite {
        object.insert("client_metadata".to_string(), json!({"lite":"true"}));
    } else {
        object.remove("client_metadata");
    }
    if priority && !object.contains_key("service_tier") {
        object.insert("service_tier".to_string(), json!("priority"));
    }

    Ok(NativeResolved {
        use_responses_lite,
        model,
        stream: object
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn bind_native_request_to_route(body: &Value, route: &CodexBoundRoute) -> Value {
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

fn resolve_native_model(requested: &str) -> (String, bool) {
    let (requested, priority) = match requested.strip_suffix("-fast") {
        Some(base) if ALLOWED_MODELS.contains(&base) => (base, true),
        _ => (requested, false),
    };
    let model = MODEL_ALIASES
        .iter()
        .find(|(alias, _)| *alias == requested)
        .map(|(_, target)| *target)
        .unwrap_or(requested);
    (model.to_string(), priority)
}

fn has_native_hosted_web_search(object: &Map<String, Value>) -> bool {
    object
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| {
            tools.iter().any(|tool| {
                matches!(
                    tool.get("type").and_then(Value::as_str),
                    Some("web_search" | "web_search_preview")
                )
            })
        })
}

fn local_codex_error(error: CodexError) -> Response {
    let status = match error.status {
        401 => StatusCode::UNAUTHORIZED,
        403 => StatusCode::FORBIDDEN,
        429 => StatusCode::TOO_MANY_REQUESTS,
        400..=599 => StatusCode::from_u16(error.status).unwrap_or(StatusCode::BAD_GATEWAY),
        _ => StatusCode::BAD_GATEWAY,
    };
    let kind = match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        _ => "api_error",
    };
    let message = error.detail.as_deref().unwrap_or(&error.message);
    let response = openai_error(status, kind, message, None, None);
    if let Some(retry_after) = error.retry_after
        && let Ok(value) = http::HeaderValue::from_str(&retry_after)
    {
        let (mut parts, body) = response.into_parts();
        parts.headers.insert(http::header::RETRY_AFTER, value);
        return Response::from_parts(parts, body);
    }
    response
}

pub fn openai_error(
    status: StatusCode,
    kind: &str,
    message: impl Into<String>,
    param: Option<&str>,
    code: Option<&str>,
) -> Response {
    (
        status,
        axum::Json(json!({
            "error": {
                "message": message.into(),
                "type": kind,
                "param": param,
                "code": code,
            }
        })),
    )
        .into_response()
}

#[derive(Clone, Default)]
pub struct NativeResponseOutcome {
    failure: Arc<Mutex<Option<String>>>,
}

impl NativeResponseOutcome {
    pub fn failure(&self) -> Option<String> {
        self.failure.lock().ok().and_then(|failure| failure.clone())
    }

    pub(crate) fn fail(&self, message: String) {
        if let Ok(mut failure) = self.failure.lock()
            && failure.is_none()
        {
            *failure = Some(message);
        }
    }
}

fn passthrough_response(
    upstream: reqwest::Response,
    ctx: RequestContext,
    body_idle_timeout_ms: u64,
    client: Arc<CodexHttpClient>,
    route: CodexBoundRoute,
    rejection_budget: Arc<AuthRejectionBudget>,
) -> Response {
    let status = upstream.status();
    let headers = passthrough_headers(upstream.headers());
    let is_sse = upstream
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    let outcome = NativeResponseOutcome::default();
    let auth_refresh = if is_sse {
        InBandAuthRefreshDetector::sse(client, route, rejection_budget)
    } else {
        InBandAuthRefreshDetector::json(client, route, rejection_budget)
    };
    let observer = NativeResponseObserver::new(ctx, is_sse, outcome.clone(), Some(auth_refresh));
    let state = Some(NativeBodyState {
        stream: Box::pin(upstream.bytes_stream()),
        observer,
        body_idle_timeout_ms,
    });
    let stream = futures_util::stream::unfold(state, |state| async move {
        let mut state = state?;
        match tokio::time::timeout(
            Duration::from_millis(state.body_idle_timeout_ms),
            state.stream.next(),
        )
        .await
        {
            Ok(Some(Ok(chunk))) => {
                state.observer.observe(&chunk);
                Some((Ok::<Bytes, io::Error>(chunk), Some(state)))
            }
            Ok(Some(Err(error))) => {
                let message = format!("Native Responses body read failed: {error}");
                state.observer.finish("read_error");
                Some((Err(io::Error::other(message)), None))
            }
            Ok(None) => {
                state.observer.finish("complete");
                None
            }
            Err(_) => {
                let message = format!(
                    "Timed out waiting {}ms for the next Codex response body chunk",
                    state.body_idle_timeout_ms
                );
                state.observer.finish("idle_timeout");
                Some((Err(io::Error::new(io::ErrorKind::TimedOut, message)), None))
            }
        }
    });

    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response.extensions_mut().insert(outcome);
    response
}

fn passthrough_headers(upstream: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (name, value) in upstream {
        if native_response_header_allowed(name) {
            headers.append(name.clone(), value.clone());
        }
    }
    headers
}

fn native_response_header_allowed(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "content-type"
            | "cache-control"
            | "retry-after"
            | "x-request-id"
            | "openai-processing-ms"
            | "openai-version"
    ) || name.as_str().starts_with("x-ratelimit-")
}

type UpstreamByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static>>;

struct NativeBodyState {
    stream: UpstreamByteStream,
    observer: NativeResponseObserver,
    body_idle_timeout_ms: u64,
}

struct NativeResponseObserver {
    ctx: RequestContext,
    is_sse: bool,
    generation_started: bool,
    raw: Vec<u8>,
    raw_truncated: u64,
    pending: Vec<u8>,
    pending_scan: usize,
    discarding_oversized_frame: bool,
    pending_truncated: bool,
    captured_events: Vec<Value>,
    captured_event_bytes: usize,
    captured_events_truncated: u64,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    outcome: NativeResponseOutcome,
    auth_refresh: Option<InBandAuthRefreshDetector>,
    finished: bool,
}

impl NativeResponseObserver {
    fn new(
        ctx: RequestContext,
        is_sse: bool,
        outcome: NativeResponseOutcome,
        auth_refresh: Option<InBandAuthRefreshDetector>,
    ) -> Self {
        Self {
            ctx,
            is_sse,
            generation_started: false,
            raw: Vec::with_capacity(64 * 1024),
            raw_truncated: 0,
            pending: Vec::new(),
            pending_scan: 0,
            discarding_oversized_frame: false,
            pending_truncated: false,
            captured_events: Vec::new(),
            captured_event_bytes: 0,
            captured_events_truncated: 0,
            input_tokens: None,
            output_tokens: None,
            outcome,
            auth_refresh,
            finished: false,
        }
    }

    fn observe(&mut self, chunk: &[u8]) {
        if let Some(auth_refresh) = self.auth_refresh.as_mut() {
            auth_refresh.observe(chunk);
        }
        if !chunk.is_empty() && !self.generation_started {
            if let Some(monitor) = self.ctx.monitor.as_ref() {
                monitor.generation_started(&self.ctx.req_id);
            }
            self.generation_started = true;
        }
        self.capture_raw(chunk);

        let events = if self.is_sse {
            self.pending.extend_from_slice(chunk);
            self.drain_sse_events()
        } else {
            0
        };
        if let Some(monitor) = self.ctx.monitor.as_ref() {
            monitor.stream_progress(
                &self.ctx.req_id,
                chunk.len() as u64,
                events,
                self.input_tokens,
                self.output_tokens,
            );
        }
    }

    fn capture_raw(&mut self, chunk: &[u8]) {
        let remaining = MAX_SSE_CAPTURE_BYTES.saturating_sub(self.raw.len());
        let captured = remaining.min(chunk.len());
        self.raw.extend_from_slice(&chunk[..captured]);
        if captured < chunk.len() {
            self.raw_truncated = self
                .raw_truncated
                .saturating_add((chunk.len() - captured) as u64);
        }
    }

    fn drain_sse_events(&mut self) -> u64 {
        if self.discarding_oversized_frame {
            let Some((end, separator_len)) = find_sse_boundary(&self.pending) else {
                retain_boundary_prefix(&mut self.pending);
                return 0;
            };
            self.pending.drain(..end + separator_len);
            self.discarding_oversized_frame = false;
            self.pending_scan = 0;
        }

        let mut consumed = 0;
        let mut parsed = Vec::new();
        while let Some((relative_end, separator_len)) =
            find_sse_boundary_from(&self.pending, self.pending_scan)
        {
            let end = relative_end + separator_len;
            parsed.extend(parse_sse_events(&self.pending[consumed..end]));
            consumed = end;
            self.pending_scan = consumed;
        }
        if consumed > 0 {
            self.pending.drain(..consumed);
            self.pending_scan = 0;
        } else {
            self.pending_scan = self.pending.len().saturating_sub(3);
        }

        let mut count = 0_u64;
        for event in parsed {
            count += 1;
            if event.data == "[DONE]" {
                continue;
            }
            match serde_json::from_str::<Value>(&event.data) {
                Ok(value) => self.record_event(event.event.as_deref(), value),
                Err(_) => self.capture_event(json!({
                    "event": event.event,
                    "unparseable": true,
                    "bytes": event.data.len(),
                })),
            }
        }

        if self.pending.len() > MAX_STREAM_CAPTURE_FRAME_BYTES {
            self.pending_truncated = true;
            self.discarding_oversized_frame = true;
            retain_boundary_prefix(&mut self.pending);
            self.pending_scan = 0;
        }
        count
    }

    fn record_event(&mut self, event: Option<&str>, value: Value) {
        self.update_usage(&value);
        self.update_outcome(&value);
        let mut captured = value;
        if let Some(event) = event
            && let Some(object) = captured.as_object_mut()
        {
            object
                .entry("_sse_event")
                .or_insert_with(|| Value::String(event.to_string()));
        }
        self.capture_event(captured);
    }

    fn capture_event(&mut self, value: Value) {
        if self.ctx.traffic.is_none() {
            return;
        }
        let bytes = serde_json::to_vec(&value).map_or(0, |value| value.len());
        if self.captured_events.len() < MAX_STREAM_CAPTURE_EVENTS
            && self.captured_event_bytes.saturating_add(bytes) <= MAX_STREAM_CAPTURE_EVENT_BYTES
        {
            self.captured_event_bytes += bytes;
            self.captured_events.push(value);
        } else {
            self.captured_events_truncated = self.captured_events_truncated.saturating_add(1);
        }
    }

    fn update_outcome(&self, value: &Value) {
        let event_type = value.get("type").and_then(Value::as_str);
        let has_error = value.get("error").is_some_and(|error| !error.is_null());
        let failed_status = value.get("status").and_then(Value::as_str) == Some("failed");
        if matches!(
            event_type,
            Some("response.failed" | "response.error" | "error")
        ) || has_error
            || failed_status
        {
            let message = value
                .pointer("/response/error/message")
                .or_else(|| value.pointer("/error/message"))
                .or_else(|| value.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Native Responses stream failed");
            self.outcome.fail(message.to_string());
        }
    }

    fn update_usage(&mut self, value: &Value) {
        let usage = value
            .pointer("/response/usage")
            .or_else(|| value.get("usage"));
        if let Some(usage) = usage {
            self.input_tokens = usage
                .get("input_tokens")
                .and_then(Value::as_u64)
                .or(self.input_tokens);
            self.output_tokens = usage
                .get("output_tokens")
                .and_then(Value::as_u64)
                .or(self.output_tokens);
        }
    }

    fn finish(&mut self, outcome: &str) {
        if self.finished {
            return;
        }
        self.finished = true;
        if let Some(auth_refresh) = self.auth_refresh.as_mut() {
            auth_refresh.finish();
        }
        if !self.is_sse
            && let Ok(value) = serde_json::from_slice::<Value>(&self.raw)
        {
            self.update_usage(&value);
            self.update_outcome(&value);
            self.capture_event(value);
            if let Some(monitor) = self.ctx.monitor.as_ref() {
                monitor.usage_updated(&self.ctx.req_id, self.input_tokens, self.output_tokens);
            }
        }
        self.write_capture(outcome);
    }

    fn write_capture(&self, outcome: &str) {
        let Some(traffic) = self.ctx.traffic.as_deref() else {
            return;
        };
        if !self.raw.is_empty() {
            traffic.write_bytes(
                if self.is_sse {
                    "032-upstream-response-body.sse"
                } else {
                    "032-upstream-response-body.json"
                },
                &self.raw,
            );
        }
        for event in &self.captured_events {
            traffic.write_json_event("040-upstream-event", event);
        }
        traffic.write_json(
            "033-native-response-capture",
            &json!({
                "outcome": outcome,
                "capturedBytes": self.raw.len(),
                "truncatedBytes": self.raw_truncated,
                "pendingFrameTruncated": self.pending_truncated,
                "capturedEvents": self.captured_events.len(),
                "capturedEventBytes": self.captured_event_bytes,
                "truncatedEvents": self.captured_events_truncated,
                "inputTokens": self.input_tokens,
                "outputTokens": self.output_tokens,
            }),
        );
    }
}

impl Drop for NativeResponseObserver {
    fn drop(&mut self) {
        if !self.finished {
            self.finish("downstream_cancelled");
        }
    }
}

fn find_sse_boundary(bytes: &[u8]) -> Option<(usize, usize)> {
    find_sse_boundary_from(bytes, 0)
}

fn find_sse_boundary_from(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    for index in start.min(bytes.len())..bytes.len() {
        if bytes[index..].starts_with(b"\r\n\r\n") {
            return Some((index, 4));
        }
        if bytes[index..].starts_with(b"\n\n") || bytes[index..].starts_with(b"\r\r") {
            return Some((index, 2));
        }
    }
    None
}

fn retain_boundary_prefix(bytes: &mut Vec<u8>) {
    let keep = bytes.len().min(3);
    if bytes.len() > keep {
        bytes.drain(..bytes.len() - keep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::codex::auth::{
        manager::CodexAuthManager,
        token_store::{StoredAuth, file_store},
    };
    use crate::request_identity::{ConversationIdentity, RequestScope};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    fn request(body: Value) -> Value {
        body
    }

    fn observer_context() -> RequestContext {
        RequestContext {
            req_id: "native-test".into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        }
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> (String, Value) {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = socket.read(&mut buffer).await.unwrap();
            assert!(read > 0);
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                return (
                    headers.into_owned(),
                    serde_json::from_slice(
                        &request[header_end + 4..header_end + 4 + content_length],
                    )
                    .unwrap(),
                );
            }
        }
    }

    fn test_backend(address: std::net::SocketAddr, auth: StoredAuth) -> CodexNativeBackend {
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        );
        client.auth_manager().set_test_auth(auth);
        CodexNativeBackend::with_client(client)
    }

    fn scoped_context(identity: ConversationIdentity, req_id: &str) -> ScopedRequestContext {
        let mut ctx = observer_context();
        ctx.req_id = req_id.to_string();
        ctx.session_id = Some(identity.session_component().to_string());
        let scope =
            RequestScope::from_conversation_identity(Some(identity), RequestPurpose::Conversation);
        ScopedRequestContext::new(ctx, scope)
    }

    #[tokio::test]
    async fn native_scoped_routes_keep_sibling_agents_opaque_and_distinct() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(read_http_request(&mut socket).await);
                let body = br#"{"id":"resp_native","object":"response","status":"completed"}"#;
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(body).await.unwrap();
            }
            captured
        });
        let backend = test_backend(
            address,
            StoredAuth {
                access: "native-token".into(),
                refresh: String::new(),
                account_id: Some("native-account".into()),
                expires: u64::MAX,
            },
        );

        for agent in ["agent-a", "agent-b"] {
            let response = backend
                .handle_scoped(
                    json!({"model":"gpt-5.4","input":"hello","stream":false}),
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
        let session_a = bound_session(&captured[0].0);
        let session_b = bound_session(&captured[1].0);
        assert_ne!(session_a, session_b);
        for (headers, _) in captured {
            for raw in ["shared-session", "agent-a", "agent-b"] {
                assert!(!headers.contains(raw));
            }
        }
    }

    #[tokio::test]
    async fn native_header_401_rebuilds_once_and_namespaces_original_cache_key() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let backend = test_backend(
            address,
            StoredAuth {
                access: "native-a".into(),
                refresh: "refresh-a".into(),
                account_id: Some("native-account".into()),
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
                        access: "native-b".into(),
                        refresh: "refresh-b".into(),
                        account_id: Some("native-account".into()),
                        expires: u64::MAX,
                    });
                    socket
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 7\r\nconnection: close\r\n\r\nroute-a",
                        )
                        .await
                        .unwrap();
                } else {
                    let body = br#"{"id":"resp_b","object":"response","status":"completed"}"#;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        body.len()
                    );
                    socket.write_all(head.as_bytes()).await.unwrap();
                    socket.write_all(body).await.unwrap();
                }
            }
            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err()
            );
            captured
        });
        let identity = ConversationIdentity::Main("native-session".into());
        let response = backend
            .handle_scoped(
                json!({
                    "model":"gpt-5.4",
                    "input":"unchanged",
                    "stream":false,
                    "prompt_cache_key":"caller-cache"
                }),
                scoped_context(identity.clone(), "native-rebind"),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        let captured = server.await.unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].1["input"], captured[1].1["input"]);
        assert_ne!(
            captured[0].1["prompt_cache_key"],
            captured[1].1["prompt_cache_key"]
        );
        assert_ne!(captured[1].1["prompt_cache_key"], "caller-cache");
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
    async fn native_continuation_header_401_refreshes_without_replay() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let backend = test_backend(
            address,
            StoredAuth {
                access: "continuation-a".into(),
                refresh: "refresh-a".into(),
                account_id: Some("continuation-account".into()),
                expires: u64::MAX,
            },
        );
        let server_client = backend.client.clone();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let captured = read_http_request(&mut socket).await;
            server_client.auth_manager().set_test_auth(StoredAuth {
                access: "continuation-b".into(),
                refresh: "refresh-b".into(),
                account_id: Some("continuation-account".into()),
                expires: u64::MAX,
            });
            socket
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 14\r\nconnection: close\r\n\r\nstale response",
                )
                .await
                .unwrap();
            (
                captured,
                tokio::time::timeout(Duration::from_millis(100), listener.accept()).await,
            )
        });

        let response = backend
            .handle(
                json!({
                    "model":"gpt-5.4",
                    "previous_response_id":"resp_route_a",
                    "input":"delta only",
                    "stream":false
                }),
                observer_context(),
            )
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let response_body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(response_body.as_ref(), b"stale response");
        let ((_, captured), second) = server.await.unwrap();
        assert!(second.is_err());
        assert_eq!(captured["previous_response_id"], "resp_route_a");
        assert_eq!(
            backend
                .client
                .auth_manager()
                .get_auth()
                .await
                .unwrap()
                .access,
            "continuation-b"
        );
    }

    #[tokio::test]
    async fn native_route_b_401_cannot_refresh_or_send_route_c() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let backend = test_backend(
            address,
            StoredAuth {
                access: "native-a".into(),
                refresh: "refresh-a".into(),
                account_id: Some("native-account".into()),
                expires: u64::MAX,
            },
        );
        let server_client = backend.client.clone();
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(read_http_request(&mut socket).await);
                server_client.auth_manager().set_test_auth(StoredAuth {
                    access: if attempt == 0 { "native-b" } else { "native-c" }.into(),
                    refresh: format!("refresh-{}", attempt + 2),
                    account_id: Some("native-account".into()),
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
            (
                tokio::time::timeout(Duration::from_millis(100), listener.accept()).await,
                captured,
            )
        });

        let response = backend
            .handle(
                json!({"model":"gpt-5.4","input":"hello","stream":false}),
                observer_context(),
            )
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), b"route-b");
        let (third, captured) = server.await.unwrap();
        assert!(third.is_err());
        assert_eq!(captured.len(), 2);
        assert!(captured[0].0.contains("authorization: Bearer native-a"));
        assert!(captured[1].0.contains("authorization: Bearer native-b"));
    }

    #[tokio::test]
    async fn native_split_json_in_band_401_preserves_response_and_refreshes_once_later() {
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_address = upstream_listener.local_addr().unwrap();
        let oauth_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let oauth_address = oauth_listener.local_addr().unwrap();
        let auth_manager = CodexAuthManager::new_for_test(
            file_store(),
            format!("http://{oauth_address}/oauth/token"),
        );
        auth_manager.set_test_auth(StoredAuth {
            access: "in-band-a".into(),
            refresh: "in-band-refresh".into(),
            account_id: Some("in-band-account".into()),
            expires: u64::MAX,
        });
        let client = CodexHttpClient::new_for_test_with_auth_manager(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            auth_manager,
            format!("http://{upstream_address}/v1/responses"),
            1_000,
            1_000,
            0,
        );
        let backend = CodexNativeBackend::with_client(client);
        let observable = br#"{"type":"response.failed","status_code":401,"response":{"error":{"status":401,"message":"expired in band"}}}"#;
        let upstream = tokio::spawn(async move {
            let (mut socket, _) = upstream_listener.accept().await.unwrap();
            let _ = read_http_request(&mut socket).await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            for chunk in observable.chunks(17) {
                socket
                    .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                    .await
                    .unwrap();
                socket.write_all(chunk).await.unwrap();
                socket.write_all(b"\r\n").await.unwrap();
            }
            socket.write_all(b"0\r\n\r\n").await.unwrap();
        });
        let oauth = tokio::spawn(async move {
            let (mut socket, _) = oauth_listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let read = socket.read(&mut request).await.unwrap();
            assert!(String::from_utf8_lossy(&request[..read]).contains("in-band-refresh"));
            let body = br#"{"access_token":"in-band-b","refresh_token":"in-band-refresh-b","expires_in":3600}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(body).await.unwrap();
            tokio::time::timeout(Duration::from_millis(100), oauth_listener.accept()).await
        });

        let response = backend
            .handle(
                json!({"model":"gpt-5.4","input":"hello","stream":false}),
                observer_context(),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), observable);
        upstream.await.unwrap();
        assert!(oauth.await.unwrap().is_err());
        assert_eq!(
            backend
                .client
                .auth_manager()
                .get_auth()
                .await
                .unwrap()
                .access,
            "in-band-b"
        );
    }

    #[tokio::test]
    async fn native_sse_in_band_401_preserves_status_body_and_framing_without_replay() {
        const AUTH_SSE: &[u8] = b"event: response.failed\r\ndata: {\"type\":\"response.failed\",\"status_code\":401,\"response\":{\"error\":{\"status\":401,\"message\":\"stream expired\"}}}\r\n\r\ndata: [DONE]\r\n\r\n";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = test_backend(
            listener.local_addr().unwrap(),
            StoredAuth {
                access: "native-sse".into(),
                refresh: String::new(),
                account_id: Some("native-sse-account".into()),
                expires: u64::MAX,
            },
        );
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut socket).await;
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                AUTH_SSE.len()
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            for chunk in AUTH_SSE.chunks(19) {
                socket.write_all(chunk).await.unwrap();
            }
            tokio::time::timeout(Duration::from_millis(75), listener.accept()).await
        });

        let response = backend
            .handle(
                json!({"model":"gpt-5.4","input":"hello","stream":true}),
                observer_context(),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[http::header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), AUTH_SSE);
        assert!(server.await.unwrap().is_err());
    }

    #[test]
    fn failed_sse_event_records_native_outcome() {
        let outcome = NativeResponseOutcome::default();
        let mut observer =
            NativeResponseObserver::new(observer_context(), true, outcome.clone(), None);
        observer.observe(
            b"event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"generation failed\"}}}\n\n",
        );

        assert_eq!(outcome.failure().as_deref(), Some("generation failed"));
    }

    #[test]
    fn completed_json_with_null_error_stays_successful() {
        let outcome = NativeResponseOutcome::default();
        let mut observer =
            NativeResponseObserver::new(observer_context(), false, outcome.clone(), None);
        observer
            .observe(br#"{"id":"resp_ok","object":"response","status":"completed","error":null}"#);
        observer.finish("complete");

        assert_eq!(outcome.failure(), None);
    }

    #[test]
    fn failed_json_records_error_message() {
        let outcome = NativeResponseOutcome::default();
        let mut observer =
            NativeResponseObserver::new(observer_context(), false, outcome.clone(), None);
        observer.observe(
            br#"{"id":"resp_failed","object":"response","status":"failed","error":{"message":"request failed"}}"#,
        );
        observer.finish("complete");

        assert_eq!(outcome.failure().as_deref(), Some("request failed"));
    }

    #[test]
    fn response_error_event_records_failure() {
        let outcome = NativeResponseOutcome::default();
        let mut observer =
            NativeResponseObserver::new(observer_context(), true, outcome.clone(), None);
        observer.observe(
            b"event: response.error\ndata: {\"type\":\"response.error\",\"response\":{\"error\":{\"message\":\"stream error\"}}}\n\n",
        );

        assert_eq!(outcome.failure().as_deref(), Some("stream error"));
    }

    #[test]
    fn event_capture_obeys_count_limit() {
        let temp = tempfile::TempDir::new().unwrap();
        let outcome = NativeResponseOutcome::default();
        let mut context = observer_context();
        context.traffic = Some(Arc::new(crate::traffic::test_capture(
            temp.path().to_path_buf(),
        )));
        let mut observer = NativeResponseObserver::new(context, true, outcome, None);
        for index in 0..MAX_STREAM_CAPTURE_EVENTS + 10 {
            observer.capture_event(json!({"index": index}));
        }

        assert_eq!(observer.captured_events.len(), MAX_STREAM_CAPTURE_EVENTS);
        assert_eq!(observer.captured_events_truncated, 10);
        observer.finished = true;
    }

    #[test]
    fn native_request_requires_object_and_model() {
        assert!(shape_native_request(&mut json!([])).is_err());
        assert!(shape_native_request(&mut json!({})).is_err());
        assert!(shape_native_request(&mut json!({"model": 7})).is_err());
    }

    #[test]
    fn native_request_resolves_alias_and_fast_tier() {
        let mut body = request(json!({"model":"claude-opus-5","input":[]}));
        let resolved = shape_native_request(&mut body).unwrap();
        assert_eq!(resolved.model, "gpt-5.6-sol");
        assert_eq!(body["model"], "gpt-5.6-sol");
        assert!(resolved.use_responses_lite);
        assert_eq!(body["client_metadata"]["lite"], "true");

        let mut fast = request(json!({
            "model":"gpt-5.4-fast",
            "input":[],
            "client_metadata":{"lite":"true"}
        }));
        let resolved = shape_native_request(&mut fast).unwrap();
        assert_eq!(resolved.model, "gpt-5.4");
        assert_eq!(fast["service_tier"], "priority");
        assert!(fast.get("client_metadata").is_none());
    }

    #[test]
    fn native_request_preserves_parallel_tool_calls() {
        for parallel in [false, true] {
            let mut body = request(json!({
                "model":"gpt-5.4",
                "input":[],
                "parallel_tool_calls":parallel
            }));
            shape_native_request(&mut body).unwrap();
            assert_eq!(body["parallel_tool_calls"], parallel);
        }
    }

    #[test]
    fn explicit_service_tier_is_preserved() {
        let mut body = request(json!({
            "model":"gpt-5.4-fast",
            "service_tier":"flex",
            "input":[]
        }));
        shape_native_request(&mut body).unwrap();
        assert_eq!(body["service_tier"], "flex");
    }

    #[test]
    fn hosted_search_uses_full_lane_and_upgrades_luna() {
        for tool_type in ["web_search", "web_search_preview"] {
            let mut body = request(json!({
                "model":"gpt-5.6-luna",
                "tools":[{"type":tool_type}],
                "input":[],
                "client_metadata":{"lite":"true"}
            }));
            let resolved = shape_native_request(&mut body).unwrap();
            assert_eq!(resolved.model, "gpt-5.6-sol");
            assert!(!resolved.use_responses_lite);
            assert!(body.get("client_metadata").is_none());
        }
    }

    #[tokio::test]
    async fn openai_error_has_native_envelope() {
        let response = openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "bad model",
            Some("model"),
            Some("invalid"),
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(value.get("type").is_none());
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["param"], "model");
        assert_eq!(value["error"]["code"], "invalid");
    }

    #[test]
    fn response_headers_use_allowlist() {
        let mut upstream = HeaderMap::new();
        upstream.insert(
            http::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );
        upstream.insert(http::header::SET_COOKIE, "secret=1".parse().unwrap());
        upstream.insert(http::header::CONTENT_LENGTH, "12".parse().unwrap());
        upstream.insert("x-request-id", "req_1".parse().unwrap());
        upstream.insert("x-ratelimit-remaining-requests", "2".parse().unwrap());

        let headers = passthrough_headers(&upstream);
        assert_eq!(
            headers.get(http::header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        assert_eq!(headers.get("x-request-id").unwrap(), "req_1");
        assert_eq!(headers.get("x-ratelimit-remaining-requests").unwrap(), "2");
        assert!(headers.get(http::header::SET_COOKIE).is_none());
        assert!(headers.get(http::header::CONTENT_LENGTH).is_none());
    }

    #[test]
    fn sse_boundary_handles_lf_and_crlf() {
        assert_eq!(find_sse_boundary(b"data: {}\n\nnext"), Some((8, 2)));
        assert_eq!(find_sse_boundary(b"data: {}\r\n\r\nnext"), Some((8, 4)));
        assert_eq!(find_sse_boundary(b"data: {}"), None);
    }
}
