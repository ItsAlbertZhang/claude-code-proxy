use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::anthropic::sse::parse_sse_events;
use crate::config;
use crate::logging::create_logger;
use crate::provider::RequestContext;
use crate::retry::{compute_backoff_delay, should_retry_status, sleep};
use crate::traffic::TrafficCapture;

use super::auth::constants::{CODEX_API_ENDPOINT, ORIGINATOR, RESPONSES_LITE_ORIGINATOR};
use super::auth::manager::CodexAuthManager;
use super::auth::token_store::{DefaultCodexAuthStore, StoredAuth, file_store};
use super::search::{SearchRequest, SearchResponse};
use super::state::{ConversationBinding, ProtocolLane, SocketPoolKey};
use super::translate::request::ResponsesRequest;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CodexError {
    pub status: u16,
    pub message: String,
    pub detail: Option<String>,
    pub retry_after: Option<String>,
    pub origin: CodexErrorOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexErrorOrigin {
    Http,
    WebSocket,
    WebSocketHandshake,
    Auth,
    BufferedHttp,
    BufferedWebSocket,
}

impl CodexError {
    pub fn new(status: u16, message: String) -> Self {
        Self {
            status,
            message,
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        }
    }
}

impl std::fmt::Display for CodexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Codex error {}: {}", self.status, self.message)
    }
}

#[derive(Debug)]
pub struct CodexHeaderTimeoutError {
    pub timeout_ms: u64,
}

impl std::fmt::Display for CodexHeaderTimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Timed out waiting {}ms for Codex response headers",
            self.timeout_ms
        )
    }
}

#[derive(Debug)]
pub struct CodexTransportError {
    pub message: String,
}

impl std::fmt::Display for CodexTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Codex transport error: {}", self.message)
    }
}

// ---------------------------------------------------------------------------
// Header builder
// ---------------------------------------------------------------------------

fn default_user_agent(use_responses_lite: bool) -> String {
    if use_responses_lite {
        RESPONSES_LITE_ORIGINATOR.to_string()
    } else {
        format!("claude-code-proxy/{}", env!("CARGO_PKG_VERSION"))
    }
}

pub fn build_codex_headers(
    auth: &StoredAuth,
    ctx: &RequestContext,
    use_responses_lite: bool,
) -> Result<http::HeaderMap, CodexError> {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        header_value("content-type", "application/json")?,
    );
    headers.insert(
        http::header::ACCEPT,
        header_value("accept", "text/event-stream")?,
    );
    let bearer = format!("Bearer {}", auth.access);
    headers.insert(
        http::header::AUTHORIZATION,
        header_value("authorization", &bearer)?,
    );
    let originator = if use_responses_lite {
        RESPONSES_LITE_ORIGINATOR.to_string()
    } else {
        config::codex_originator(ORIGINATOR)
    };
    headers.insert("originator", header_value("originator", &originator)?);
    headers.insert(
        "openai-beta",
        header_value("openai-beta", "responses=experimental")?,
    );
    headers.insert(
        "x-codex-beta-features",
        header_value("x-codex-beta-features", "remote_compaction_v2")?,
    );
    if use_responses_lite {
        headers.insert(
            "x-openai-internal-codex-responses-lite",
            header_value("x-openai-internal-codex-responses-lite", "true")?,
        );
    }
    if let Some(ref account_id) = auth.account_id {
        headers.insert(
            "ChatGPT-Account-Id",
            header_value("ChatGPT-Account-Id", account_id)?,
        );
    }
    if let Some(ref session_id) = ctx.session_id {
        headers.insert("session_id", header_value("session_id", session_id)?);
        headers.insert(
            "x-client-request-id",
            header_value("x-client-request-id", &ctx.req_id)?,
        );
        let window_id = format!("{session_id}:0");
        headers.insert(
            "x-codex-window-id",
            header_value("x-codex-window-id", &window_id)?,
        );
    }
    let user_agent = config::codex_user_agent(&default_user_agent(use_responses_lite));
    if !user_agent.is_empty() {
        headers.insert(
            http::header::USER_AGENT,
            header_value("user-agent", &user_agent)?,
        );
    }
    Ok(headers)
}

pub fn build_native_codex_headers(
    auth: &StoredAuth,
    ctx: &RequestContext,
    use_responses_lite: bool,
    stream: bool,
) -> Result<http::HeaderMap, CodexError> {
    let mut headers = build_codex_headers(auth, ctx, use_responses_lite)?;
    headers.insert(
        http::header::ACCEPT,
        header_value(
            "accept",
            if stream {
                "text/event-stream"
            } else {
                "application/json"
            },
        )?,
    );
    Ok(headers)
}

pub fn build_codex_search_headers(
    auth: &StoredAuth,
    ctx: &RequestContext,
) -> Result<http::HeaderMap, CodexError> {
    let mut headers = build_codex_headers(auth, ctx, false)?;
    headers.insert(
        http::header::ACCEPT,
        header_value("accept", "application/json")?,
    );
    let originator = config::codex_originator(RESPONSES_LITE_ORIGINATOR);
    headers.insert("originator", header_value("originator", &originator)?);
    let user_agent = config::codex_user_agent(RESPONSES_LITE_ORIGINATOR);
    if !user_agent.is_empty() {
        headers.insert(
            http::header::USER_AGENT,
            header_value("user-agent", &user_agent)?,
        );
    }
    Ok(headers)
}

pub fn build_codex_image_headers(
    auth: &StoredAuth,
    ctx: &RequestContext,
) -> Result<http::HeaderMap, CodexError> {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        header_value("content-type", "application/json")?,
    );
    headers.insert(
        http::header::ACCEPT,
        header_value("accept", "application/json")?,
    );
    headers.insert(
        http::header::AUTHORIZATION,
        header_value("authorization", &format!("Bearer {}", auth.access))?,
    );
    headers.insert(
        "originator",
        header_value("originator", &config::codex_originator(ORIGINATOR))?,
    );
    if let Some(account_id) = auth.account_id.as_deref() {
        headers.insert(
            "ChatGPT-Account-Id",
            header_value("ChatGPT-Account-Id", account_id)?,
        );
    }
    if ctx.session_id.is_some() {
        headers.insert(
            "x-client-request-id",
            header_value("x-client-request-id", &ctx.req_id)?,
        );
    }
    let user_agent = config::codex_user_agent(&default_user_agent(false));
    if !user_agent.is_empty() {
        headers.insert(
            http::header::USER_AGENT,
            header_value("user-agent", &user_agent)?,
        );
    }
    Ok(headers)
}

pub fn build_codex_transcription_headers(
    auth: &StoredAuth,
    ctx: &RequestContext,
) -> Result<http::HeaderMap, CodexError> {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::ACCEPT,
        header_value("accept", "application/json")?,
    );
    headers.insert(
        http::header::AUTHORIZATION,
        header_value("authorization", &format!("Bearer {}", auth.access))?,
    );
    headers.insert("originator", header_value("originator", "Codex Desktop")?);
    if let Some(account_id) = auth.account_id.as_deref() {
        headers.insert(
            "ChatGPT-Account-Id",
            header_value("ChatGPT-Account-Id", account_id)?,
        );
    }
    if ctx.session_id.is_some() {
        headers.insert(
            "x-client-request-id",
            header_value("x-client-request-id", &ctx.req_id)?,
        );
    }
    let user_agent = config::codex_user_agent(&default_user_agent(false));
    if !user_agent.is_empty() {
        headers.insert(
            http::header::USER_AGENT,
            header_value("user-agent", &user_agent)?,
        );
    }
    Ok(headers)
}

fn header_value(name: &str, value: &str) -> Result<http::HeaderValue, CodexError> {
    http::HeaderValue::from_str(value).map_err(|e| CodexError {
        status: 500,
        message: format!("Failed to parse {name} header"),
        detail: Some(e.to_string()),
        retry_after: None,
        origin: CodexErrorOrigin::Http,
    })
}

fn search_endpoint(base_url: &str) -> String {
    let base_url = base_url.trim_end_matches('/');
    match base_url.strip_suffix("/responses") {
        Some(api_root) => format!("{api_root}/alpha/search"),
        None => format!("{base_url}/alpha/search"),
    }
}

// ---------------------------------------------------------------------------
// WebSocket request shaping
// ---------------------------------------------------------------------------

pub fn build_websocket_request(
    body: &ResponsesRequest,
    continuation: Option<&super::continuation::ContinuationCandidate>,
) -> serde_json::Value {
    let mut payload = serde_json::to_value(body).unwrap_or_default();
    let obj = payload.as_object_mut().expect("request must be an object");

    // Omit the stream field for WebSocket transport
    obj.remove("stream");
    obj.insert("type".to_string(), serde_json::json!("response.create"));

    // Apply continuation if available
    if let Some(candidate) = continuation {
        if let Some(ref prev_id) = candidate.previous_response_id {
            obj.insert(
                "previous_response_id".to_string(),
                serde_json::json!(prev_id),
            );
        }
        if let Some(ref delta) = candidate.input_delta {
            obj.insert(
                "input".to_string(),
                serde_json::to_value(delta).unwrap_or_default(),
            );
        }
    }

    payload
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActualTransport {
    Http,
    WebSocket,
}

pub struct CodexResponse {
    pub body: Vec<u8>,
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub transport: ActualTransport,
    pub socket_id: Option<u64>,
}

#[derive(Clone)]
pub struct CodexConversationRoute {
    auth: StoredAuth,
    binding: ConversationBinding,
    protocol_lane: ProtocolLane,
}

impl CodexConversationRoute {
    pub fn bind_lane(&self, lane_token: &str) -> String {
        self.binding.bind_lane(lane_token)
    }

    fn matches_request(&self, body: &ResponsesRequest) -> bool {
        self.matches_protocol(body.client_metadata.is_some())
    }

    fn matches_protocol(&self, use_responses_lite: bool) -> bool {
        self.protocol_lane == ProtocolLane::from_responses_lite(use_responses_lite)
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

const MAX_BUFFERED_TRANSPORT_RETRIES: u32 = 3;
const MAX_BUFFERED_TRANSPORT_ATTEMPTS: u32 = MAX_BUFFERED_TRANSPORT_RETRIES + 1;
const MAX_NATIVE_FAILURE_EVENT_BYTES: usize = 64 * 1024;
const HTTP_RESPONSE_BODY_IDLE_TIMEOUT_MS: u64 = 300_000;
const IMAGE_HEADER_TIMEOUT_MS: u64 = 300_000;

struct NativeFailureDetector {
    kind: NativeResponseBodyKind,
    buffer: Vec<u8>,
    delimiter_tail: Vec<u8>,
    oversized: bool,
    finished: bool,
}

#[derive(Clone, Copy)]
enum NativeResponseBodyKind {
    Sse,
    Json,
}

impl NativeFailureDetector {
    fn new(stream: bool) -> Self {
        Self {
            kind: if stream {
                NativeResponseBodyKind::Sse
            } else {
                NativeResponseBodyKind::Json
            },
            buffer: Vec::new(),
            delimiter_tail: Vec::with_capacity(4),
            oversized: false,
            finished: false,
        }
    }

    fn observe(&mut self, chunk: &[u8]) -> bool {
        if self.finished {
            return false;
        }
        match self.kind {
            NativeResponseBodyKind::Sse => self.observe_sse(chunk),
            NativeResponseBodyKind::Json => self.observe_json(chunk),
        }
    }

    fn finish(&mut self) -> bool {
        if self.finished {
            return false;
        }
        self.finished = true;
        if self.oversized {
            return false;
        }
        match self.kind {
            NativeResponseBodyKind::Sse => contains_in_band_unauthorized(&self.buffer),
            NativeResponseBodyKind::Json => {
                serde_json::from_slice::<serde_json::Value>(&self.buffer)
                    .ok()
                    .as_ref()
                    .is_some_and(is_in_band_unauthorized)
            }
        }
    }

    fn abort(&mut self) {
        self.finished = true;
        self.buffer.clear();
        self.delimiter_tail.clear();
    }

    fn observe_sse(&mut self, chunk: &[u8]) -> bool {
        for &byte in chunk {
            if !self.oversized {
                if self.buffer.len() < MAX_NATIVE_FAILURE_EVENT_BYTES {
                    self.buffer.push(byte);
                } else {
                    self.buffer.clear();
                    self.oversized = true;
                }
            }

            if self.delimiter_tail.len() == 4 {
                self.delimiter_tail.remove(0);
            }
            self.delimiter_tail.push(byte);
            if !sse_event_is_complete(&self.delimiter_tail) {
                continue;
            }

            let unauthorized = !self.oversized && contains_in_band_unauthorized(&self.buffer);
            self.buffer.clear();
            self.delimiter_tail.clear();
            self.oversized = false;
            if unauthorized {
                self.finished = true;
                return true;
            }
        }
        false
    }

    fn observe_json(&mut self, chunk: &[u8]) -> bool {
        let Some(next_len) = self.buffer.len().checked_add(chunk.len()) else {
            self.finished = true;
            self.oversized = true;
            self.buffer.clear();
            return false;
        };
        if next_len > MAX_NATIVE_FAILURE_EVENT_BYTES {
            self.finished = true;
            self.oversized = true;
            self.buffer.clear();
            return false;
        }
        self.buffer.extend_from_slice(chunk);
        false
    }
}

fn sse_event_is_complete(tail: &[u8]) -> bool {
    tail.ends_with(b"\n\n")
        || tail.ends_with(b"\n\r")
        || tail.ends_with(b"\r\r")
        || tail.ends_with(b"\r\n\n")
        || tail.ends_with(b"\r\n\r")
}

fn contains_in_band_unauthorized(body: &[u8]) -> bool {
    parse_sse_events(body).into_iter().any(|event| {
        serde_json::from_str::<serde_json::Value>(&event.data)
            .ok()
            .as_ref()
            .is_some_and(is_in_band_unauthorized)
    })
}

fn is_in_band_unauthorized(payload: &serde_json::Value) -> bool {
    super::events::classify_event_failure(payload).is_some_and(|failure| failure.status == 401)
}

#[derive(Clone)]
struct ProxyEnvironment {
    http_proxy: Option<String>,
    https_proxy: Option<String>,
    all_proxy: Option<String>,
    no_proxy: Option<reqwest::NoProxy>,
    no_proxy_value: Option<String>,
}

impl ProxyEnvironment {
    fn from_env() -> Self {
        if std::env::var_os("REQUEST_METHOD").is_some() {
            return Self {
                http_proxy: None,
                https_proxy: None,
                all_proxy: None,
                no_proxy: None,
                no_proxy_value: None,
            };
        }

        let no_proxy_value = std::env::var("NO_PROXY")
            .or_else(|_| std::env::var("no_proxy"))
            .ok();
        Self {
            http_proxy: proxy_env_value("HTTP_PROXY", "http_proxy")
                .unwrap_or_else(|name| panic!("invalid {name} proxy URL")),
            https_proxy: proxy_env_value("HTTPS_PROXY", "https_proxy")
                .unwrap_or_else(|name| panic!("invalid {name} proxy URL")),
            all_proxy: proxy_env_value("ALL_PROXY", "all_proxy")
                .unwrap_or_else(|name| panic!("invalid {name} proxy URL")),
            no_proxy: no_proxy_value
                .as_deref()
                .and_then(reqwest::NoProxy::from_string),
            no_proxy_value,
        }
    }

    fn websocket_proxy_config(&self) -> super::websocket::WebSocketProxyConfig {
        super::websocket::WebSocketProxyConfig::new(
            self.http_proxy.as_deref(),
            self.https_proxy.as_deref(),
            self.all_proxy.as_deref(),
            self.no_proxy_value.as_deref(),
        )
    }

    fn apply(&self, mut builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
        builder = builder.no_proxy();
        if let Some(proxy) = self.http_proxy.as_deref() {
            builder = builder.proxy(
                reqwest::Proxy::http(proxy)
                    .expect("validated HTTP_PROXY URL")
                    .no_proxy(self.no_proxy.clone()),
            );
        }
        if let Some(proxy) = self.https_proxy.as_deref() {
            builder = builder.proxy(
                reqwest::Proxy::https(proxy)
                    .expect("validated HTTPS_PROXY URL")
                    .no_proxy(self.no_proxy.clone()),
            );
        }
        if let Some(proxy) = self.all_proxy.as_deref() {
            builder = builder.proxy(
                reqwest::Proxy::all(proxy)
                    .expect("validated ALL_PROXY URL")
                    .no_proxy(self.no_proxy.clone()),
            );
        }
        builder
    }
}

fn native_http_client(proxy_environment: &ProxyEnvironment) -> reqwest::Client {
    proxy_environment
        .apply(
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .redirect(reqwest::redirect::Policy::none()),
        )
        .build()
        .expect("failed to create native Responses HTTP client")
}

fn proxy_env_value(
    uppercase: &'static str,
    lowercase: &'static str,
) -> Result<Option<String>, &'static str> {
    let Some(raw) = std::env::var_os(uppercase).or_else(|| std::env::var_os(lowercase)) else {
        return Ok(None);
    };
    let raw = raw.into_string().map_err(|_| uppercase)?;
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    normalize_proxy_url(raw).map(Some).ok_or(uppercase)
}

fn normalize_proxy_url(raw: &str) -> Option<String> {
    let candidate = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    let parsed = url::Url::parse(&candidate).ok()?;
    if !matches!(
        parsed.scheme(),
        "http" | "https" | "socks4" | "socks4a" | "socks5" | "socks5h"
    ) || parsed.host_str().is_none()
        || matches!(parsed.scheme(), "socks4" | "socks4a")
            && (!parsed.username().is_empty() || parsed.password().is_some())
    {
        return None;
    }
    Some(parsed.to_string())
}

fn websocket_http_client(proxy_environment: &ProxyEnvironment) -> reqwest::Client {
    let tls_config = super::websocket::websocket_tls_config();
    proxy_environment
        .apply(
            reqwest::Client::builder()
                .http1_only()
                .redirect(reqwest::redirect::Policy::none())
                .use_preconfigured_tls((*tls_config).clone()),
        )
        .build()
        .expect("failed to create Codex WebSocket HTTP client")
}

fn custom_client_auto_http_fallback_enabled(
    base_url: &str,
    proxy_config: &super::websocket::WebSocketProxyConfig,
) -> bool {
    let Ok(websocket_url) = super::websocket::to_websocket_url(base_url) else {
        return false;
    };
    !proxy_config.uses_proxy_for(&websocket_url)
}

#[cfg(test)]
fn test_native_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .expect("failed to create test native Responses HTTP client")
}

#[cfg(test)]
fn test_websocket_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .http1_only()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .expect("failed to create test WebSocket HTTP client")
}

pub struct CodexHttpClient {
    client: reqwest::Client,
    native_client: reqwest::Client,
    websocket_client: reqwest::Client,
    websocket_proxy_config: super::websocket::WebSocketProxyConfig,
    auto_http_fallback_enabled: bool,
    auth_manager: Arc<CodexAuthManager<DefaultCodexAuthStore>>,
    #[cfg(test)]
    native_rejection_refreshes: Option<Arc<std::sync::atomic::AtomicUsize>>,
    base_url: String,
    header_timeout_ms: u64,
    body_idle_timeout_ms: u64,
    #[allow(dead_code)]
    header_timeout_retries: u32,
}

impl Default for CodexHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexHttpClient {
    pub fn new() -> Self {
        let timeout_ms = 60_000;
        let proxy_environment = ProxyEnvironment::from_env();
        Self {
            client: proxy_environment
                .apply(reqwest::Client::builder().connect_timeout(Duration::from_secs(15)))
                .build()
                .expect("failed to create HTTP client"),
            native_client: native_http_client(&proxy_environment),
            websocket_client: websocket_http_client(&proxy_environment),
            websocket_proxy_config: proxy_environment.websocket_proxy_config(),
            auto_http_fallback_enabled: true,
            auth_manager: Arc::new(CodexAuthManager::new(file_store())),
            #[cfg(test)]
            native_rejection_refreshes: None,
            base_url: config::codex_base_url(CODEX_API_ENDPOINT),
            header_timeout_ms: timeout_ms,
            body_idle_timeout_ms: HTTP_RESPONSE_BODY_IDLE_TIMEOUT_MS,
            header_timeout_retries: 1,
        }
    }

    pub fn new_with_client(
        client: reqwest::Client,
        auth_manager: CodexAuthManager<DefaultCodexAuthStore>,
        base_url: String,
    ) -> Self {
        let proxy_environment = ProxyEnvironment::from_env();
        let websocket_proxy_config = proxy_environment.websocket_proxy_config();
        let auto_http_fallback_enabled =
            custom_client_auto_http_fallback_enabled(&base_url, &websocket_proxy_config);
        Self {
            native_client: native_http_client(&proxy_environment),
            websocket_client: websocket_http_client(&proxy_environment),
            websocket_proxy_config,
            auto_http_fallback_enabled,
            client,
            auth_manager: Arc::new(auth_manager),
            #[cfg(test)]
            native_rejection_refreshes: None,
            base_url,
            header_timeout_ms: 60_000,
            body_idle_timeout_ms: HTTP_RESPONSE_BODY_IDLE_TIMEOUT_MS,
            header_timeout_retries: 1,
        }
    }

    #[cfg(test)]
    pub fn new_for_test(
        client: reqwest::Client,
        base_url: String,
        header_timeout_ms: u64,
        body_idle_timeout_ms: u64,
        header_timeout_retries: u32,
    ) -> Self {
        Self {
            native_client: test_native_http_client(),
            websocket_client: test_websocket_http_client(),
            websocket_proxy_config: super::websocket::WebSocketProxyConfig::direct(),
            auto_http_fallback_enabled: true,
            client,
            auth_manager: Arc::new(CodexAuthManager::new(file_store())),
            native_rejection_refreshes: None,
            base_url,
            header_timeout_ms,
            body_idle_timeout_ms,
            header_timeout_retries,
        }
    }

    pub fn auth_manager(&self) -> &CodexAuthManager<DefaultCodexAuthStore> {
        &self.auth_manager
    }

    pub fn body_idle_timeout_ms(&self) -> u64 {
        self.body_idle_timeout_ms
    }

    pub(crate) async fn post_transcription(
        &self,
        base_url: &str,
        input: &super::transcription::PreparedTranscription,
        ctx: &RequestContext,
    ) -> Result<reqwest::Response, CodexError> {
        let url = format!("{}/transcribe", base_url.trim_end_matches('/'));
        let mut auth = self
            .auth_manager
            .get_auth()
            .await
            .map_err(|error| CodexError {
                status: 401,
                message: "Auth error".to_string(),
                detail: Some(error.to_string()),
                retry_after: None,
                origin: CodexErrorOrigin::Auth,
            })?;
        let mut refresh_attempted = false;

        loop {
            let headers = build_codex_transcription_headers(&auth, ctx)?;
            let response = self.attempt_transcription(&url, &headers, input).await?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED && !refresh_attempted {
                refresh_attempted = true;
                drop(response);
                auth = self
                    .auth_manager
                    .force_refresh(&auth.access)
                    .await
                    .map_err(auth_refresh_error)?;
                continue;
            }
            return Ok(response);
        }
    }

    async fn attempt_transcription(
        &self,
        url: &str,
        headers: &http::HeaderMap,
        input: &super::transcription::PreparedTranscription,
    ) -> Result<reqwest::Response, CodexError> {
        let part = reqwest::multipart::Part::bytes(input.audio.to_vec())
            .file_name(input.filename.clone())
            .mime_str(&input.content_type)
            .map_err(|error| CodexError {
                status: 400,
                message: "Invalid audio content type".to_string(),
                detail: Some(error.to_string()),
                retry_after: None,
                origin: CodexErrorOrigin::Http,
            })?;
        let mut form = reqwest::multipart::Form::new().part("file", part);
        if let Some(language) = input.language.as_deref() {
            form = form.text("language", language.to_string());
        }
        let mut request = self.native_client.post(url).multipart(form);
        for (key, value) in headers {
            request = request.header(key.as_str(), value.as_bytes());
        }
        tokio::time::timeout(
            Duration::from_millis(IMAGE_HEADER_TIMEOUT_MS),
            request.send(),
        )
        .await
        .map_err(|_| CodexError {
            status: 0,
            message: format!(
                "Timed out waiting {}ms for Codex transcription response headers",
                IMAGE_HEADER_TIMEOUT_MS
            ),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })?
        .map_err(|error| CodexError {
            status: 0,
            message: format!("Codex transcription transport error: {error}"),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })
    }

    pub async fn conversation_route(
        &self,
        use_responses_lite: bool,
    ) -> Result<CodexConversationRoute, CodexError> {
        let auth = self
            .auth_manager
            .get_auth()
            .await
            .map_err(|error| CodexError {
                status: 401,
                message: "Auth error".to_string(),
                detail: Some(error.to_string()),
                retry_after: None,
                origin: CodexErrorOrigin::Auth,
            })?;
        let protocol_lane = ProtocolLane::from_responses_lite(use_responses_lite);
        let binding = ConversationBinding::for_request(&self.base_url, &auth, protocol_lane);
        Ok(CodexConversationRoute {
            auth,
            binding,
            protocol_lane,
        })
    }

    pub async fn conversation_binding(
        &self,
        use_responses_lite: bool,
    ) -> Result<ConversationBinding, CodexError> {
        Ok(self.conversation_route(use_responses_lite).await?.binding)
    }

    pub(crate) async fn post_image_json(
        &self,
        base_url: &str,
        operation: super::images::ImageOperation,
        body: &serde_json::Value,
        ctx: &RequestContext,
    ) -> Result<reqwest::Response, CodexError> {
        let body_json = serde_json::to_vec(body).map_err(|error| CodexError {
            status: 500,
            message: "Failed to serialize image request".to_string(),
            detail: Some(error.to_string()),
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })?;
        let url = format!(
            "{}/{}",
            base_url.trim_end_matches('/'),
            operation.upstream_path()
        );
        let mut auth = self
            .auth_manager
            .get_auth()
            .await
            .map_err(|error| CodexError {
                status: 401,
                message: "Auth error".to_string(),
                detail: Some(error.to_string()),
                retry_after: None,
                origin: CodexErrorOrigin::Auth,
            })?;
        let mut refresh_attempted = false;

        loop {
            let headers = build_codex_image_headers(&auth, ctx)?;
            let response = self
                .attempt_image_json(&url, &headers, body_json.clone())
                .await?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED && !refresh_attempted {
                refresh_attempted = true;
                drop(response);
                auth = self
                    .auth_manager
                    .force_refresh(&auth.access)
                    .await
                    .map_err(auth_refresh_error)?;
                continue;
            }
            return Ok(response);
        }
    }

    async fn attempt_image_json(
        &self,
        url: &str,
        headers: &http::HeaderMap,
        body_json: Vec<u8>,
    ) -> Result<reqwest::Response, CodexError> {
        let mut request = self.native_client.post(url);
        for (key, value) in headers {
            request = request.header(key.as_str(), value.as_bytes());
        }
        tokio::time::timeout(
            Duration::from_millis(IMAGE_HEADER_TIMEOUT_MS),
            request.body(body_json).send(),
        )
        .await
        .map_err(|_| CodexError {
            status: 0,
            message: format!(
                "Timed out waiting {}ms for Codex image response headers",
                IMAGE_HEADER_TIMEOUT_MS
            ),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })?
        .map_err(|error| CodexError {
            status: 0,
            message: format!("Codex image transport error: {error}"),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })
    }

    pub async fn post_native_responses(
        &self,
        body: &serde_json::Value,
        ctx: &RequestContext,
        use_responses_lite: bool,
        stream: bool,
    ) -> Result<reqwest::Response, CodexError> {
        let auth = self
            .auth_manager
            .get_auth()
            .await
            .map_err(|err| CodexError {
                status: 401,
                message: "Auth error".to_string(),
                detail: Some(err.to_string()),
                retry_after: None,
                origin: CodexErrorOrigin::Auth,
            })?;
        self.post_native_responses_with_auth(body, ctx, use_responses_lite, stream, auth, true)
            .await
    }

    pub async fn post_native_responses_bound(
        &self,
        route: &CodexConversationRoute,
        body: &serde_json::Value,
        ctx: &RequestContext,
        stream: bool,
    ) -> Result<reqwest::Response, CodexError> {
        let use_responses_lite = body
            .get("client_metadata")
            .is_some_and(|metadata| !metadata.is_null());
        if !route.matches_protocol(use_responses_lite) {
            return Err(CodexError {
                status: 500,
                message: "Codex route protocol mismatch".to_string(),
                detail: None,
                retry_after: None,
                origin: CodexErrorOrigin::Auth,
            });
        }
        self.post_native_responses_with_auth(
            body,
            ctx,
            use_responses_lite,
            stream,
            route.auth.clone(),
            false,
        )
        .await
    }

    async fn post_native_responses_with_auth(
        &self,
        body: &serde_json::Value,
        ctx: &RequestContext,
        use_responses_lite: bool,
        stream: bool,
        mut auth: StoredAuth,
        allow_auth_refresh: bool,
    ) -> Result<reqwest::Response, CodexError> {
        let body_json = serde_json::to_string(body).map_err(|err| CodexError {
            status: 500,
            message: "Failed to serialize native Responses request".to_string(),
            detail: Some(err.to_string()),
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })?;
        let mut refresh_attempted = false;

        loop {
            let started_at = Instant::now();
            let headers = build_native_codex_headers(&auth, ctx, use_responses_lite, stream)?;
            if let Some(traffic) = ctx.traffic.as_deref() {
                write_codex_http_request_capture(traffic, &self.base_url, &headers, &body_json);
            }

            let response = self
                .attempt_native_responses(&headers, body_json.clone())
                .await?;
            let status = response.status().as_u16();
            if status == 401 && !refresh_attempted {
                refresh_attempted = true;
                if allow_auth_refresh {
                    drop(response);
                    auth = self
                        .auth_manager
                        .force_refresh(&auth.access)
                        .await
                        .map_err(auth_refresh_error)?;
                    continue;
                }
                // Preserve this request's immutable route. Refresh or observed
                // rotation is available only when the next route binds.
                let _ = self.auth_manager.refresh_after_rejection(&auth).await;
            }

            if let Some(traffic) = ctx.traffic.as_deref() {
                write_live_upstream_response_headers(traffic, &response, started_at.elapsed());
            }
            return Ok(self.wrap_native_response_body(response, auth, stream));
        }
    }

    fn wrap_native_response_body(
        &self,
        response: reqwest::Response,
        auth: StoredAuth,
        stream: bool,
    ) -> reqwest::Response {
        if response.status() != reqwest::StatusCode::OK {
            return response;
        }

        use reqwest::ResponseBuilderExt;

        let url = response.url().clone();
        let response: http::Response<reqwest::Body> = response.into();
        let (mut parts, body) = response.into_parts();
        let url_response = http::Response::builder()
            .url(url)
            .body(())
            .expect("response URL extension must be valid");
        let (url_parts, ()) = url_response.into_parts();
        parts.extensions.extend(url_parts.extensions);

        let source = http_body_util::BodyDataStream::new(body);
        let detector = NativeFailureDetector::new(stream);
        let auth_manager = self.auth_manager.clone();
        #[cfg(test)]
        let refreshes = self.native_rejection_refreshes.clone();
        let body_stream = futures_util::stream::unfold(
            (source, detector, auth_manager, auth),
            move |(mut source, mut detector, auth_manager, auth)| {
                #[cfg(test)]
                let refreshes = refreshes.clone();
                async move {
                    match futures_util::StreamExt::next(&mut source).await {
                        Some(item) => {
                            let unauthorized = match &item {
                                Ok(chunk) => detector.observe(chunk),
                                Err(_) => {
                                    detector.abort();
                                    false
                                }
                            };
                            if unauthorized {
                                let _ = auth_manager.refresh_after_rejection(&auth).await;
                                #[cfg(test)]
                                if let Some(refreshes) = refreshes.as_ref() {
                                    refreshes.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                }
                            }
                            Some((item, (source, detector, auth_manager, auth)))
                        }
                        None => {
                            if detector.finish() {
                                let _ = auth_manager.refresh_after_rejection(&auth).await;
                                #[cfg(test)]
                                if let Some(refreshes) = refreshes.as_ref() {
                                    refreshes.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                }
                            }
                            None
                        }
                    }
                }
            },
        );
        reqwest::Response::from(http::Response::from_parts(
            parts,
            reqwest::Body::wrap_stream(body_stream),
        ))
    }

    async fn attempt_native_responses(
        &self,
        headers: &http::HeaderMap,
        body_json: String,
    ) -> Result<reqwest::Response, CodexError> {
        let mut request = self.native_client.post(&self.base_url);
        for (key, value) in headers {
            request = request.header(key.as_str(), value.as_bytes());
        }

        tokio::time::timeout(
            Duration::from_millis(self.header_timeout_ms),
            request.body(body_json).send(),
        )
        .await
        .map_err(|_| CodexError {
            status: 0,
            message: format!(
                "Timed out waiting {}ms for Codex response headers",
                self.header_timeout_ms
            ),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })?
        .map_err(|err| CodexError {
            status: 0,
            message: format!("Native Responses transport error: {err}"),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })
    }

    pub async fn post_codex(
        &self,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
    ) -> Result<CodexResponse, CodexError> {
        self.post_codex_with_transport(body, ctx, continuation, crate::config::codex_transport())
            .await
    }

    pub async fn post_codex_bound(
        &self,
        route: &CodexConversationRoute,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
    ) -> Result<CodexResponse, CodexError> {
        self.post_codex_bound_with_transport(
            route,
            body,
            ctx,
            continuation,
            crate::config::codex_transport(),
        )
        .await
    }

    async fn post_codex_bound_with_transport(
        &self,
        route: &CodexConversationRoute,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
        transport: crate::config::CodexTransport,
    ) -> Result<CodexResponse, CodexError> {
        if !route.matches_request(body) {
            return Err(CodexError {
                status: 500,
                message: "Codex route protocol mismatch".to_string(),
                detail: None,
                retry_after: None,
                origin: CodexErrorOrigin::Auth,
            });
        }
        self.post_codex_with_transport_auth(
            body,
            ctx,
            continuation,
            transport,
            route.auth.clone(),
            false,
        )
        .await
    }

    pub async fn post_search(
        &self,
        body: &SearchRequest,
        ctx: &RequestContext,
    ) -> Result<SearchResponse, CodexError> {
        let auth = self.auth_manager.get_auth().await.map_err(|e| CodexError {
            status: 401,
            message: "Auth error".to_string(),
            detail: Some(e.to_string()),
            retry_after: None,
            origin: CodexErrorOrigin::Auth,
        })?;
        self.post_search_with_auth(body, ctx, auth, true).await
    }

    pub async fn post_search_bound(
        &self,
        route: &CodexConversationRoute,
        body: &SearchRequest,
        ctx: &RequestContext,
    ) -> Result<SearchResponse, CodexError> {
        self.post_search_with_auth(body, ctx, route.auth.clone(), false)
            .await
    }

    async fn post_search_with_auth(
        &self,
        body: &SearchRequest,
        ctx: &RequestContext,
        mut auth: StoredAuth,
        allow_auth_refresh: bool,
    ) -> Result<SearchResponse, CodexError> {
        let body_json = serde_json::to_string(body).map_err(|e| CodexError {
            status: 500,
            message: "Failed to serialize search request".to_string(),
            detail: Some(e.to_string()),
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })?;
        let mut auth_refresh_attempted = false;
        let mut retries = 0_u32;

        loop {
            let response = self.attempt_post_search(&auth, &body_json, ctx).await?;
            if response.status == 401 && !auth_refresh_attempted {
                auth_refresh_attempted = true;
                match self.auth_manager.refresh_after_rejection(&auth).await {
                    Ok(new_auth) if allow_auth_refresh => {
                        auth = new_auth;
                        continue;
                    }
                    Ok(_) | Err(_) if !allow_auth_refresh => {
                        // Keep this search on its immutable route. Any refresh or
                        // observed rotation is available when the next route binds.
                    }
                    Err(error) => return Err(auth_refresh_error(error)),
                    Ok(_) => unreachable!("refresh-enabled branch handled above"),
                }
            }
            if response.status == 401 {
                return Err(codex_status_error(response));
            }
            if should_retry_codex_status(response.status)
                && retries < MAX_BUFFERED_TRANSPORT_RETRIES
            {
                let retry_after = response
                    .headers
                    .iter()
                    .find(|(key, _)| key.eq_ignore_ascii_case("retry-after"))
                    .map(|(_, value)| value.as_str());
                let delay = compute_backoff_delay(retries, retry_after);
                if delay.exceeds_budget {
                    return Err(codex_status_error(response));
                }
                retries += 1;
                sleep(delay.wait_ms).await;
                continue;
            }
            if !(200..300).contains(&response.status) {
                return Err(codex_status_error(response));
            }
            return serde_json::from_slice(&response.body).map_err(|e| CodexError {
                status: 502,
                message: "Failed to decode Codex search response".to_string(),
                detail: Some(e.to_string()),
                retry_after: None,
                origin: CodexErrorOrigin::Http,
            });
        }
    }

    async fn post_codex_with_transport(
        &self,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
        transport: crate::config::CodexTransport,
    ) -> Result<CodexResponse, CodexError> {
        let auth = self.auth_manager.get_auth().await.map_err(|e| CodexError {
            status: 401,
            message: "Auth error".to_string(),
            detail: Some(e.to_string()),
            retry_after: None,
            origin: CodexErrorOrigin::Auth,
        })?;
        self.post_codex_with_transport_auth(body, ctx, continuation, transport, auth, true)
            .await
    }

    async fn post_codex_with_transport_auth(
        &self,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
        transport: crate::config::CodexTransport,
        mut auth: StoredAuth,
        allow_auth_refresh: bool,
    ) -> Result<CodexResponse, CodexError> {
        use crate::config::CodexTransport;

        let use_responses_lite = body.client_metadata.is_some();
        let initial_pool_key =
            websocket_pool_key(ctx, continuation, &auth, &self.base_url, use_responses_lite);
        if should_reset_websocket_pool(continuation)
            && let Some(key) = initial_pool_key.as_ref()
        {
            super::websocket::invalidate_codex_websocket_pool_turn(
                key.as_str(),
                ctx.session_id.as_deref(),
                continuation.and_then(|candidate| candidate.turn_id),
            );
        }

        let turn_id = continuation.and_then(|candidate| candidate.turn_id);
        let mut active_continuation = continuation.cloned();
        let mut auth_refresh_attempted = false;
        let mut transport_failures = 0u32;
        loop {
            let pool_key = websocket_pool_key(
                ctx,
                active_continuation.as_ref(),
                &auth,
                &self.base_url,
                use_responses_lite,
            );
            let pool_key = pool_key.as_ref().map(SocketPoolKey::as_str);
            let result = match transport {
                CodexTransport::Http => {
                    let body_json = serde_json::to_string(body).map_err(|e| CodexError {
                        status: 500,
                        message: "Failed to serialize request".to_string(),
                        detail: Some(e.to_string()),
                        retry_after: None,
                        origin: CodexErrorOrigin::Http,
                    })?;
                    self.attempt_post_http(&auth, &body_json, ctx, body.client_metadata.is_some())
                        .await
                }
                CodexTransport::WebSocket => {
                    let ws_headers =
                        build_codex_headers(&auth, ctx, body.client_metadata.is_some())?;
                    let ws_headers = super::websocket::codex_websocket_headers(&ws_headers);
                    let ws_body = build_websocket_request(body, active_continuation.as_ref());

                    super::websocket::codex_websocket_request(
                        &self.websocket_client,
                        &self.websocket_proxy_config,
                        &self.base_url,
                        &ws_headers,
                        &ws_body,
                        ctx,
                        ctx.traffic.as_deref(),
                        pool_key,
                        super::websocket::WEBSOCKET_CONNECT_TIMEOUT_MS,
                        super::websocket::WEBSOCKET_IDLE_TIMEOUT_MS,
                        active_continuation.as_ref(),
                    )
                    .await
                }
                CodexTransport::Auto => {
                    let ws_headers =
                        build_codex_headers(&auth, ctx, body.client_metadata.is_some())?;
                    let ws_headers = super::websocket::codex_websocket_headers(&ws_headers);
                    let ws_body = build_websocket_request(body, active_continuation.as_ref());

                    // Try WebSocket first
                    let ws_result = super::websocket::codex_websocket_request(
                        &self.websocket_client,
                        &self.websocket_proxy_config,
                        &self.base_url,
                        &ws_headers,
                        &ws_body,
                        ctx,
                        ctx.traffic.as_deref(),
                        pool_key,
                        super::websocket::WEBSOCKET_CONNECT_TIMEOUT_MS,
                        super::websocket::WEBSOCKET_IDLE_TIMEOUT_MS,
                        active_continuation.as_ref(),
                    )
                    .await;

                    match ws_result {
                        Ok(response) => Ok(response),
                        Err(err)
                            if self.auto_http_fallback_enabled
                                && should_fallback_to_http(&err)
                                && !should_retry_without_continuation(
                                    &err,
                                    active_continuation.as_ref(),
                                ) =>
                        {
                            // Fall back to HTTP only if WebSocket failed before sending
                            let body_json =
                                serde_json::to_string(body).map_err(|e| CodexError {
                                    status: 500,
                                    message: "Failed to serialize request".to_string(),
                                    detail: Some(e.to_string()),
                                    retry_after: None,
                                    origin: CodexErrorOrigin::Http,
                                })?;
                            self.attempt_post_http(
                                &auth,
                                &body_json,
                                ctx,
                                body.client_metadata.is_some(),
                            )
                            .await
                        }
                        Err(err) => Err(err),
                    }
                }
            };

            let buffered_unauthorized = result.as_ref().ok().and_then(|response| {
                (200..300)
                    .contains(&response.status)
                    .then(|| super::events::first_failure_with_status(&response.body, 401))
                    .flatten()
                    .map(|failure| (failure, response.transport))
            });
            if (buffered_unauthorized.is_some()
                || should_refresh_after_unauthorized(&result, auth_refresh_attempted, transport))
                && !auth_refresh_attempted
            {
                auth_refresh_attempted = true;
                if let Some(key) = pool_key {
                    super::websocket::invalidate_codex_websocket_pool_turn(
                        key,
                        ctx.session_id.as_deref(),
                        turn_id,
                    );
                }
                match self.auth_manager.refresh_after_rejection(&auth).await {
                    Ok(new_auth) if allow_auth_refresh => {
                        auth = new_auth;
                        active_continuation =
                            full_context_continuation(active_continuation.as_ref());
                        continue;
                    }
                    Ok(_) | Err(_) if !allow_auth_refresh => {
                        // Keep this request on its immutable route. The refresh or
                        // observed rotation is available when the next route binds.
                    }
                    Err(e) => {
                        return Err(CodexError {
                            status: 401,
                            message: "Unauthorized".to_string(),
                            detail: Some(e.to_string()),
                            retry_after: None,
                            origin: CodexErrorOrigin::Http,
                        });
                    }
                    Ok(_) => unreachable!("refresh-enabled branch handled above"),
                }
            }

            if let Some((failure, actual_transport)) = buffered_unauthorized {
                return Err(CodexError {
                    status: 401,
                    message: failure.message.clone(),
                    detail: Some(failure.message),
                    retry_after: failure.retry_after,
                    origin: match actual_transport {
                        ActualTransport::Http => CodexErrorOrigin::BufferedHttp,
                        ActualTransport::WebSocket => CodexErrorOrigin::BufferedWebSocket,
                    },
                });
            }

            if let Ok(response) = &result
                && (200..300).contains(&response.status)
                && let Some(failure) = super::events::first_retryable_failure(&response.body)
            {
                if transport_failures < MAX_BUFFERED_TRANSPORT_RETRIES {
                    let delay =
                        compute_backoff_delay(transport_failures, failure.retry_after.as_deref());
                    if delay.exceeds_budget {
                        return Err(CodexError {
                            status: failure.status,
                            message: failure.message.clone(),
                            detail: Some(failure.message),
                            retry_after: failure.retry_after,
                            origin: match response.transport {
                                ActualTransport::Http => CodexErrorOrigin::BufferedHttp,
                                ActualTransport::WebSocket => CodexErrorOrigin::BufferedWebSocket,
                            },
                        });
                    }
                    log_buffered_retry(
                        ctx,
                        transport,
                        transport_failures + 1,
                        delay.wait_ms,
                        failure.status,
                        "upstream_event",
                        &failure.message,
                    );
                    transport_failures += 1;
                    active_continuation = full_context_continuation(active_continuation.as_ref());
                    sleep(delay.wait_ms).await;
                    continue;
                }

                log_buffered_retry_exhausted(
                    ctx,
                    transport,
                    failure.status,
                    "upstream_event",
                    &failure.message,
                );
                return Err(CodexError {
                    status: failure.status,
                    message: failure.message.clone(),
                    detail: Some(failure.message),
                    retry_after: failure.retry_after,
                    origin: CodexErrorOrigin::Http,
                });
            }

            match result {
                Ok(response) if response.status == 401 => {
                    let detail = String::from_utf8_lossy(&response.body).to_string();
                    return Err(CodexError {
                        status: 401,
                        message: "Unauthorized".to_string(),
                        detail: Some(detail),
                        retry_after: None,
                        origin: CodexErrorOrigin::Http,
                    });
                }
                Ok(response) if response.status == 403 => {
                    let detail = String::from_utf8_lossy(&response.body).to_string();
                    return Err(CodexError {
                        status: 403,
                        message: "Forbidden".to_string(),
                        detail: Some(detail),
                        retry_after: None,
                        origin: CodexErrorOrigin::Http,
                    });
                }
                Ok(response) if response.status == 429 => {
                    let retry_after = response
                        .headers
                        .iter()
                        .find(|(k, _)| k.to_lowercase() == "retry-after")
                        .map(|(_, v)| v.clone());
                    if transport_failures < MAX_BUFFERED_TRANSPORT_RETRIES {
                        let delay =
                            compute_backoff_delay(transport_failures, retry_after.as_deref());
                        if delay.exceeds_budget {
                            let detail = String::from_utf8_lossy(&response.body).to_string();
                            return Err(CodexError {
                                status: 429,
                                message: "Rate limited".to_string(),
                                detail: Some(detail),
                                retry_after,
                                origin: CodexErrorOrigin::Http,
                            });
                        }
                        log_buffered_retry(
                            ctx,
                            transport,
                            transport_failures + 1,
                            delay.wait_ms,
                            response.status,
                            "upstream",
                            "rate limited",
                        );
                        transport_failures += 1;
                        sleep(delay.wait_ms).await;
                        continue;
                    }
                    let detail = String::from_utf8_lossy(&response.body).to_string();
                    log_buffered_retry_exhausted(
                        ctx,
                        transport,
                        response.status,
                        "upstream",
                        "rate limited",
                    );
                    return Err(CodexError {
                        status: 429,
                        message: "Rate limited".to_string(),
                        detail: Some(detail),
                        retry_after,
                        origin: CodexErrorOrigin::Http,
                    });
                }
                Ok(response) if should_retry_codex_status(response.status) => {
                    if transport_failures < MAX_BUFFERED_TRANSPORT_RETRIES {
                        let retry_after = response
                            .headers
                            .iter()
                            .find(|(key, _)| key.eq_ignore_ascii_case("retry-after"))
                            .map(|(_, value)| value.as_str());
                        let delay = compute_backoff_delay(transport_failures, retry_after);
                        if delay.exceeds_budget {
                            return Err(codex_status_error(response));
                        }
                        log_buffered_retry(
                            ctx,
                            transport,
                            transport_failures + 1,
                            delay.wait_ms,
                            response.status,
                            "upstream",
                            "retryable upstream status",
                        );
                        transport_failures += 1;
                        sleep(delay.wait_ms).await;
                        continue;
                    }
                    log_buffered_retry_exhausted(
                        ctx,
                        transport,
                        response.status,
                        "upstream",
                        "retryable upstream status",
                    );
                    return Err(codex_status_error(response));
                }
                Ok(response) if !(200..300).contains(&response.status) => {
                    return Err(codex_status_error(response));
                }
                Ok(response) => return Ok(response),
                Err(err)
                    if should_retry_without_continuation(&err, active_continuation.as_ref()) =>
                {
                    if let Some(key) = pool_key {
                        super::websocket::invalidate_codex_websocket_pool_turn(
                            key,
                            ctx.session_id.as_deref(),
                            turn_id,
                        );
                    }
                    active_continuation = full_context_continuation(active_continuation.as_ref());
                    continue;
                }
                Err(err) => {
                    // Determine if retryable
                    let retryable = is_retryable_transport_error(&err);
                    if retryable && transport_failures < MAX_BUFFERED_TRANSPORT_RETRIES {
                        let delay =
                            compute_backoff_delay(transport_failures, err.retry_after.as_deref());
                        if delay.exceeds_budget {
                            return Err(err);
                        }
                        log_buffered_retry(
                            ctx,
                            transport,
                            transport_failures + 1,
                            delay.wait_ms,
                            err.status,
                            codex_error_origin_name(err.origin),
                            &err.message,
                        );
                        transport_failures += 1;
                        sleep(delay.wait_ms).await;
                        continue;
                    }
                    if retryable {
                        log_buffered_retry_exhausted(
                            ctx,
                            transport,
                            err.status,
                            codex_error_origin_name(err.origin),
                            &err.message,
                        );
                    }
                    return Err(err);
                }
            }
        }
    }

    pub async fn stream_codex_websocket_events(
        self: &Arc<Self>,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
    ) -> Result<super::websocket::CodexWebSocketEventReceiver, CodexError> {
        let auth = self.auth_manager.get_auth().await.map_err(|e| CodexError {
            status: 401,
            message: "Auth error".to_string(),
            detail: Some(e.to_string()),
            retry_after: None,
            origin: CodexErrorOrigin::Auth,
        })?;
        self.stream_codex_websocket_events_with_auth(body, ctx, continuation, auth, true)
            .await
    }

    pub async fn stream_codex_websocket_events_bound(
        self: &Arc<Self>,
        route: &CodexConversationRoute,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
    ) -> Result<super::websocket::CodexWebSocketEventReceiver, CodexError> {
        if !route.matches_request(body) {
            return Err(CodexError {
                status: 500,
                message: "Codex route protocol mismatch".to_string(),
                detail: None,
                retry_after: None,
                origin: CodexErrorOrigin::Auth,
            });
        }
        self.stream_codex_websocket_events_with_auth(
            body,
            ctx,
            continuation,
            route.auth.clone(),
            false,
        )
        .await
    }

    async fn stream_codex_websocket_events_with_auth(
        self: &Arc<Self>,
        body: &ResponsesRequest,
        ctx: &RequestContext,
        continuation: Option<&super::continuation::ContinuationCandidate>,
        auth: StoredAuth,
        allow_auth_refresh: bool,
    ) -> Result<super::websocket::CodexWebSocketEventReceiver, CodexError> {
        let turn_id = continuation.and_then(|candidate| candidate.turn_id);
        let pool_key = websocket_pool_key(
            ctx,
            continuation,
            &auth,
            &self.base_url,
            body.client_metadata.is_some(),
        )
        .map(|key| key.as_str().to_string());
        if should_reset_websocket_pool(continuation)
            && let Some(key) = pool_key.as_deref()
        {
            super::websocket::invalidate_codex_websocket_pool_turn(
                key,
                ctx.session_id.as_deref(),
                turn_id,
            );
        }

        let client = self.clone();
        let body = body.clone();
        let ctx = ctx.clone();
        let continuation = continuation.cloned();
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let (rx, socket_id_publisher) = super::websocket::CodexWebSocketEventReceiver::pending(rx);
        tokio::spawn(async move {
            client
                .coordinate_live_websocket_events(
                    body,
                    ctx,
                    continuation,
                    auth,
                    pool_key,
                    tx,
                    socket_id_publisher,
                    allow_auth_refresh,
                )
                .await;
        });

        Ok(rx)
    }

    #[allow(clippy::too_many_arguments)]
    async fn coordinate_live_websocket_events(
        &self,
        body: ResponsesRequest,
        ctx: RequestContext,
        mut continuation: Option<super::continuation::ContinuationCandidate>,
        mut auth: StoredAuth,
        initial_pool_key: Option<String>,
        tx: tokio::sync::mpsc::Sender<Result<serde_json::Value, CodexError>>,
        socket_id_publisher: super::websocket::CodexWebSocketSocketIdPublisher,
        allow_auth_refresh: bool,
    ) {
        let mut pool_key = initial_pool_key;
        let turn_id = continuation
            .as_ref()
            .and_then(|candidate| candidate.turn_id);
        let mut auth_refresh_attempted = false;
        let mut continuation_retry_available = continuation
            .as_ref()
            .and_then(|candidate| candidate.previous_response_id.as_deref())
            .is_some();
        let mut forwarded_any = false;

        'attempt: loop {
            socket_id_publisher.publish(None);
            let ws_headers = match build_codex_headers(&auth, &ctx, body.client_metadata.is_some())
            {
                Ok(headers) => super::websocket::codex_websocket_headers(&headers),
                Err(err) => {
                    let _ = tx.send(Err(err)).await;
                    return;
                }
            };
            let ws_body = build_websocket_request(&body, continuation.as_ref());
            let start = super::websocket::codex_websocket_event_stream(
                &self.websocket_client,
                &self.websocket_proxy_config,
                &self.base_url,
                &ws_headers,
                &ws_body,
                &ctx,
                ctx.traffic.clone(),
                pool_key.as_deref(),
                super::websocket::WEBSOCKET_CONNECT_TIMEOUT_MS,
                super::websocket::WEBSOCKET_IDLE_TIMEOUT_MS,
                continuation.as_ref(),
            );
            let mut stream = tokio::select! {
                _ = tx.closed() => {
                    super::continuation::abort_continuation(ctx.session_id.as_deref(), turn_id);
                    if let Some(key) = pool_key.as_deref() {
                        super::websocket::invalidate_codex_websocket_pool_turn(
                                key,
                                ctx.session_id.as_deref(),
                                turn_id,
                            );
                    }
                    return;
                }
                result = start => match result {
                    Ok(stream) => stream,
                    Err(err) if err.status == 401 && !auth_refresh_attempted && !forwarded_any => {
                        auth_refresh_attempted = true;
                        if let Some(key) = pool_key.as_deref() {
                            super::websocket::invalidate_codex_websocket_pool_turn(
                                key,
                                ctx.session_id.as_deref(),
                                turn_id,
                            );
                        }
                        match self.auth_manager.refresh_after_rejection(&auth).await {
                            Ok(new_auth) if allow_auth_refresh => {
                                if tx.is_closed() {
                                    return;
                                }
                                auth = new_auth;
                                pool_key = websocket_pool_key(
                                    &ctx,
                                    continuation.as_ref(),
                                    &auth,
                                    &self.base_url,
                                    body.client_metadata.is_some(),
                                )
                                .map(|key| key.as_str().to_string());
                                continuation = full_context_continuation(continuation.as_ref());
                                continue 'attempt;
                            }
                            Ok(_) | Err(_) if !allow_auth_refresh => {
                                let _ = tx.send(Err(err)).await;
                                return;
                            }
                            Err(refresh_err) => {
                                let _ = tx.send(Err(auth_refresh_error(refresh_err))).await;
                                return;
                            }
                            Ok(_) => unreachable!("refresh-enabled branch handled above"),
                        }
                    }
                    Err(err) if continuation_retry_available && is_continuation_retry_error(&err) => {
                        continuation_retry_available = false;
                        if let Some(key) = pool_key.as_deref() {
                            super::websocket::invalidate_codex_websocket_pool_turn(
                                key,
                                ctx.session_id.as_deref(),
                                turn_id,
                            );
                        }
                        continuation = full_context_continuation(continuation.as_ref());
                        continue 'attempt;
                    }
                    Err(err) => {
                        let _ = tx.send(Err(err)).await;
                        return;
                    }
                }
            };
            socket_id_publisher.publish(stream.socket_id());

            loop {
                let item = tokio::select! {
                    _ = tx.closed() => {
                        if let Some(key) = pool_key.as_deref() {
                            super::websocket::invalidate_codex_websocket_pool_turn(
                                key,
                                ctx.session_id.as_deref(),
                                turn_id,
                            );
                        }
                        return;
                    }
                    item = stream.recv() => item,
                };
                let Some(item) = item else {
                    return;
                };

                let unauthorized = match &item {
                    Err(err) => err.status == 401,
                    Ok(payload) => super::websocket::event_error_status(payload) == Some(401),
                };
                if unauthorized && !auth_refresh_attempted && !forwarded_any {
                    auth_refresh_attempted = true;
                    if let Some(key) = pool_key.as_deref() {
                        super::websocket::invalidate_codex_websocket_pool_turn(
                            key,
                            ctx.session_id.as_deref(),
                            turn_id,
                        );
                    }
                    match self.auth_manager.refresh_after_rejection(&auth).await {
                        Ok(new_auth) if allow_auth_refresh => {
                            if tx.is_closed() {
                                return;
                            }
                            auth = new_auth;
                            pool_key = websocket_pool_key(
                                &ctx,
                                continuation.as_ref(),
                                &auth,
                                &self.base_url,
                                body.client_metadata.is_some(),
                            )
                            .map(|key| key.as_str().to_string());
                            continuation = full_context_continuation(continuation.as_ref());
                            continuation_retry_available = false;
                            continue 'attempt;
                        }
                        Ok(_) | Err(_) if !allow_auth_refresh => {
                            let error = match item {
                                Err(err) => err,
                                Ok(payload) => CodexError {
                                    status: 401,
                                    message: "Unauthorized".to_string(),
                                    detail: Some(payload.to_string()),
                                    retry_after: None,
                                    origin: CodexErrorOrigin::WebSocket,
                                },
                            };
                            let _ = tx.send(Err(error)).await;
                            return;
                        }
                        Err(refresh_err) => {
                            let _ = tx.send(Err(auth_refresh_error(refresh_err))).await;
                            return;
                        }
                        Ok(_) => unreachable!("refresh-enabled branch handled above"),
                    }
                }

                if let Err(err) = &item
                    && continuation_retry_available
                    && is_continuation_retry_error(err)
                    && !forwarded_any
                {
                    continuation_retry_available = false;
                    if let Some(key) = pool_key.as_deref() {
                        super::websocket::invalidate_codex_websocket_pool_turn(
                            key,
                            ctx.session_id.as_deref(),
                            turn_id,
                        );
                    }
                    continuation = full_context_continuation(continuation.as_ref());
                    continue 'attempt;
                }

                if item.as_ref().is_ok_and(event_closes_live_retry_window) {
                    forwarded_any = true;
                }
                if tx.send(item).await.is_err() {
                    super::continuation::abort_continuation(ctx.session_id.as_deref(), turn_id);
                    if let Some(key) = pool_key.as_deref() {
                        super::websocket::invalidate_codex_websocket_pool_turn(
                            key,
                            ctx.session_id.as_deref(),
                            turn_id,
                        );
                    }
                    return;
                }
            }
        }
    }

    async fn attempt_post_http(
        &self,
        auth: &StoredAuth,
        body_json: &str,
        ctx: &RequestContext,
        use_responses_lite: bool,
    ) -> Result<CodexResponse, CodexError> {
        let url = &self.base_url;
        let headers = build_codex_headers(auth, ctx, use_responses_lite)?;

        if let Some(traffic) = ctx.traffic.as_deref() {
            write_codex_http_request_capture(traffic, url, &headers, body_json);
        }

        // Build headers
        let mut req_builder = self.client.post(url);
        for (key, value) in headers.iter() {
            req_builder = req_builder.header(key.as_str(), value.as_bytes());
        }

        // Apply header timeout
        let started_at = Instant::now();
        let send_fut = req_builder.body(body_json.to_string()).send();
        let header_timeout_dur = Duration::from_millis(self.header_timeout_ms);

        let mut resp = tokio::time::timeout(header_timeout_dur, send_fut)
            .await
            .map_err(|_| CodexError {
                status: 0,
                message: format!(
                    "Timed out waiting {}ms for Codex response headers",
                    self.header_timeout_ms
                ),
                detail: None,
                retry_after: None,
                origin: CodexErrorOrigin::Http,
            })?
            .map_err(|e| {
                if is_retryable_reqwest_error(&e) {
                    CodexError {
                        status: 0,
                        message: format!("Transport error: {e}"),
                        detail: None,
                        retry_after: None,
                        origin: CodexErrorOrigin::Http,
                    }
                } else {
                    CodexError {
                        status: 0,
                        message: format!("Network error: {e}"),
                        detail: None,
                        retry_after: None,
                        origin: CodexErrorOrigin::Http,
                    }
                }
            })?;

        let status = resp.status().as_u16();
        let headers: Vec<(String, String)> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        let mut body_bytes = Vec::new();
        let mut response_started = false;
        loop {
            let chunk = tokio::time::timeout(
                Duration::from_millis(self.body_idle_timeout_ms),
                resp.chunk(),
            )
            .await
            .map_err(|_| CodexError {
                status: 0,
                message: format!(
                    "Timed out waiting {}ms for the next Codex response body chunk",
                    self.body_idle_timeout_ms
                ),
                detail: Some("http_response_body".to_string()),
                retry_after: None,
                origin: CodexErrorOrigin::Http,
            })?
            .map_err(|e| CodexError {
                status: 0,
                message: format!("Transport error reading Codex response body: {e}"),
                detail: Some("http_response_body".to_string()),
                retry_after: None,
                origin: CodexErrorOrigin::Http,
            })?;

            let Some(chunk) = chunk else {
                break;
            };
            if !response_started {
                if let Some(monitor) = ctx.monitor.as_ref() {
                    monitor.generation_started(&ctx.req_id);
                }
                response_started = true;
            }
            body_bytes.extend_from_slice(&chunk);
        }

        if let Some(traffic) = ctx.traffic.as_deref() {
            write_upstream_response_capture(
                traffic,
                status,
                started_at.elapsed(),
                &headers,
                &body_bytes,
            );
        }

        Ok(CodexResponse {
            body: body_bytes,
            status,
            headers,
            transport: ActualTransport::Http,
            socket_id: None,
        })
    }

    async fn attempt_post_search(
        &self,
        auth: &StoredAuth,
        body_json: &str,
        ctx: &RequestContext,
    ) -> Result<CodexResponse, CodexError> {
        let url = search_endpoint(&self.base_url);
        let headers = build_codex_search_headers(auth, ctx)?;

        if let Some(traffic) = ctx.traffic.as_deref() {
            write_codex_http_request_capture(traffic, &url, &headers, body_json);
        }

        let mut request = self.client.post(&url);
        for (key, value) in headers.iter() {
            request = request.header(key.as_str(), value.as_bytes());
        }
        let started_at = Instant::now();
        let mut response = tokio::time::timeout(
            Duration::from_millis(self.header_timeout_ms),
            request.body(body_json.to_string()).send(),
        )
        .await
        .map_err(|_| CodexError {
            status: 0,
            message: format!(
                "Timed out waiting {}ms for Codex search response headers",
                self.header_timeout_ms
            ),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })?
        .map_err(|e| CodexError {
            status: 0,
            message: format!("Codex search network error: {e}"),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::Http,
        })?;

        let status = response.status().as_u16();
        let headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .map(|(key, value)| {
                (
                    key.to_string(),
                    value.to_str().unwrap_or_default().to_string(),
                )
            })
            .collect();
        let mut body = Vec::new();
        let mut response_started = false;
        loop {
            let chunk = tokio::time::timeout(
                Duration::from_millis(self.body_idle_timeout_ms),
                response.chunk(),
            )
            .await
            .map_err(|_| CodexError {
                status: 0,
                message: format!(
                    "Timed out waiting {}ms for the next Codex search response body chunk",
                    self.body_idle_timeout_ms
                ),
                detail: Some("http_response_body".to_string()),
                retry_after: None,
                origin: CodexErrorOrigin::Http,
            })?
            .map_err(|e| CodexError {
                status: 0,
                message: format!("Transport error reading Codex search response body: {e}"),
                detail: Some("http_response_body".to_string()),
                retry_after: None,
                origin: CodexErrorOrigin::Http,
            })?;
            let Some(chunk) = chunk else {
                break;
            };
            if !response_started {
                if let Some(monitor) = ctx.monitor.as_ref() {
                    monitor.generation_started(&ctx.req_id);
                }
                response_started = true;
            }
            body.extend_from_slice(&chunk);
        }

        if let Some(traffic) = ctx.traffic.as_deref() {
            write_upstream_response_capture(traffic, status, started_at.elapsed(), &headers, &body);
        }

        Ok(CodexResponse {
            body,
            status,
            headers,
            transport: ActualTransport::Http,
            socket_id: None,
        })
    }
}

fn write_codex_http_request_capture(
    traffic: &TrafficCapture,
    url: &str,
    headers: &http::HeaderMap,
    body_json: &str,
) {
    let body = serde_json::from_str(body_json).unwrap_or_else(|_| {
        serde_json::json!({
            "unparseable": true,
            "bytes": body_json.len(),
        })
    });
    traffic.write_json("020-upstream-request", &body);
    traffic.write_json(
        "021-upstream-request-metadata",
        &serde_json::json!({
            "provider": "codex",
            "transport": "http",
            "url": url,
            "method": "POST",
            "headers": headers_to_json(headers),
            "size": summarize_json_request_size(&body, body_json),
        }),
    );
}

fn write_live_upstream_response_headers(
    traffic: &TrafficCapture,
    response: &reqwest::Response,
    elapsed: Duration,
) {
    traffic.write_json(
        "030-upstream-response-headers",
        &serde_json::json!({
            "status": response.status().as_u16(),
            "elapsedMs": elapsed.as_millis(),
            "headers": headers_to_json(response.headers()),
        }),
    );
}

fn write_upstream_response_capture(
    traffic: &TrafficCapture,
    status: u16,
    elapsed: Duration,
    headers: &[(String, String)],
    body: &[u8],
) {
    traffic.write_json(
        "030-upstream-response-headers",
        &serde_json::json!({
            "status": status,
            "elapsedMs": elapsed.as_millis(),
            "headers": headers_to_json_from_pairs(headers),
        }),
    );
    if status >= 400 {
        traffic.write_text("031-upstream-error-body", &String::from_utf8_lossy(body));
    } else {
        traffic.write_bytes("032-upstream-response-body.sse", body);
        write_codex_sse_event_capture(traffic, body);
    }
}

fn write_codex_sse_event_capture(traffic: &TrafficCapture, body: &[u8]) {
    for event in parse_sse_events(body) {
        if event.data == "[DONE]" {
            traffic.write_json_event(
                "040-upstream-event",
                &serde_json::json!({
                    "event": event.event,
                    "data": "[DONE]",
                }),
            );
            continue;
        }

        match serde_json::from_str::<serde_json::Value>(&event.data) {
            Ok(mut value) => {
                if let Some(name) = event.event
                    && let Some(obj) = value.as_object_mut()
                {
                    obj.entry("_sse_event").or_insert(serde_json::json!(name));
                }
                traffic.write_json_event("040-upstream-event", &value);
            }
            Err(_) => {
                traffic.write_json_event(
                    "040-upstream-event",
                    &serde_json::json!({
                        "event": event.event,
                        "unparseable": true,
                        "data": event.data,
                    }),
                );
            }
        }
    }
}

fn headers_to_json(headers: &http::HeaderMap) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (key, value) in headers.iter() {
        out.insert(
            key.to_string(),
            serde_json::Value::String(value.to_str().unwrap_or("").to_string()),
        );
    }
    serde_json::Value::Object(out)
}

fn headers_to_json_from_pairs(headers: &[(String, String)]) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (key, value) in headers {
        out.insert(key.clone(), serde_json::Value::String(value.clone()));
    }
    serde_json::Value::Object(out)
}

fn summarize_json_request_size(body: &serde_json::Value, body_json: &str) -> serde_json::Value {
    serde_json::json!({
        "bytes": body_json.len(),
        "inputCount": body
            .get("input")
            .and_then(|v| v.as_array())
            .map(|items| items.len()),
        "toolCount": body
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|items| items.len()),
    })
}

fn auth_refresh_error(err: anyhow::Error) -> CodexError {
    CodexError {
        status: 401,
        message: "Unauthorized".to_string(),
        detail: Some(err.to_string()),
        retry_after: None,
        origin: CodexErrorOrigin::Auth,
    }
}

fn codex_status_error(response: CodexResponse) -> CodexError {
    let retry_after = response
        .headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("retry-after"))
        .map(|(_, value)| value.clone());
    let message = codex_status_error_message(&response.body).unwrap_or_else(|| {
        format!(
            "Upstream Codex request failed with status {}",
            response.status
        )
    });
    CodexError {
        status: response.status,
        message: message.clone(),
        detail: Some(message),
        retry_after,
        origin: match response.transport {
            ActualTransport::Http => CodexErrorOrigin::BufferedHttp,
            ActualTransport::WebSocket => CodexErrorOrigin::BufferedWebSocket,
        },
    }
}

fn codex_status_error_message(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .or_else(|| value.get("message"))
                .or_else(|| value.get("detail"))
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
        .or_else(|| {
            parse_sse_events(body).into_iter().find_map(|event| {
                let payload = serde_json::from_str::<serde_json::Value>(&event.data).ok()?;
                super::events::classify_event_failure(&payload).map(|failure| failure.message)
            })
        })
}

fn should_retry_codex_status(status: u16) -> bool {
    should_retry_status(status) || status == 529
}

fn codex_error_origin_name(origin: CodexErrorOrigin) -> &'static str {
    match origin {
        CodexErrorOrigin::Http => "http",
        CodexErrorOrigin::WebSocket => "websocket",
        CodexErrorOrigin::WebSocketHandshake => "websocket_handshake",
        CodexErrorOrigin::Auth => "auth",
        CodexErrorOrigin::BufferedHttp => "buffered_http",
        CodexErrorOrigin::BufferedWebSocket => "buffered_websocket",
    }
}

fn log_buffered_retry(
    ctx: &RequestContext,
    transport: crate::config::CodexTransport,
    failed_attempt: u32,
    delay_ms: u64,
    status: u16,
    origin: &str,
    reason: &str,
) {
    let mut fields = serde_json::Map::new();
    fields.insert("reqId".into(), serde_json::json!(ctx.req_id));
    fields.insert("transport".into(), serde_json::json!(transport.as_str()));
    fields.insert("failedAttempt".into(), serde_json::json!(failed_attempt));
    fields.insert("nextAttempt".into(), serde_json::json!(failed_attempt + 1));
    fields.insert(
        "maxAttempts".into(),
        serde_json::json!(MAX_BUFFERED_TRANSPORT_ATTEMPTS),
    );
    fields.insert("delayMs".into(), serde_json::json!(delay_ms));
    fields.insert("status".into(), serde_json::json!(status));
    fields.insert("origin".into(), serde_json::json!(origin));
    fields.insert("reason".into(), serde_json::json!(reason));
    create_logger("codex").warn("buffered_transport_retry", Some(fields));
}

fn log_buffered_retry_exhausted(
    ctx: &RequestContext,
    transport: crate::config::CodexTransport,
    status: u16,
    origin: &str,
    reason: &str,
) {
    let mut fields = serde_json::Map::new();
    fields.insert("reqId".into(), serde_json::json!(ctx.req_id));
    fields.insert("transport".into(), serde_json::json!(transport.as_str()));
    fields.insert(
        "attempts".into(),
        serde_json::json!(MAX_BUFFERED_TRANSPORT_ATTEMPTS),
    );
    fields.insert("status".into(), serde_json::json!(status));
    fields.insert("origin".into(), serde_json::json!(origin));
    fields.insert("reason".into(), serde_json::json!(reason));
    create_logger("codex").warn("buffered_transport_retry_exhausted", Some(fields));
}

fn is_retryable_transport_error(err: &CodexError) -> bool {
    if err.origin == CodexErrorOrigin::WebSocketHandshake {
        if err.detail.as_deref() == Some(super::websocket::WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL) {
            return false;
        }
        return err.status == 0 || should_retry_codex_status(err.status);
    }
    if err.detail.as_deref() == Some("websocket_pre_request") {
        return err.status == 0 || should_retry_codex_status(err.status);
    }
    if err.status != 0 {
        return false;
    }

    let message = err.message.to_ascii_lowercase();
    message.contains("timed out waiting")
        || message.contains("transport error")
        || message.contains("connection reset")
        || message.contains("connection closed")
        || message.contains("timed out")
        || message.contains("econnreset")
        || message.contains("etimedout")
}

fn is_retryable_reqwest_error(err: &reqwest::Error) -> bool {
    if err.is_timeout() || err.is_connect() {
        return true;
    }
    let msg = err.to_string().to_lowercase();
    msg.contains("connection reset")
        || msg.contains("connection closed")
        || msg.contains("econnreset")
        || msg.contains("etimedout")
        || msg.contains("epipe")
}

fn should_refresh_after_unauthorized(
    result: &Result<CodexResponse, CodexError>,
    auth_refresh_attempted: bool,
    transport: crate::config::CodexTransport,
) -> bool {
    if auth_refresh_attempted {
        return false;
    }
    match result {
        Ok(response) => response.status == 401,
        Err(err) => {
            err.status == 401
                && (err.origin != CodexErrorOrigin::WebSocketHandshake
                    || transport == crate::config::CodexTransport::WebSocket)
        }
    }
}

fn should_fallback_to_http(err: &CodexError) -> bool {
    err.origin == CodexErrorOrigin::WebSocketHandshake
        && err.status != http::StatusCode::PROXY_AUTHENTICATION_REQUIRED.as_u16()
        && err.detail.as_deref() != Some(super::websocket::WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL)
}

fn should_retry_without_continuation(
    err: &CodexError,
    continuation: Option<&super::continuation::ContinuationCandidate>,
) -> bool {
    if continuation
        .and_then(|c| c.previous_response_id.as_deref())
        .is_none()
    {
        return false;
    }

    is_continuation_retry_error(err)
}

fn full_context_continuation(
    continuation: Option<&super::continuation::ContinuationCandidate>,
) -> Option<super::continuation::ContinuationCandidate> {
    continuation.map(|candidate| super::continuation::ContinuationCandidate {
        turn_id: candidate.turn_id,
        previous_response_id: None,
        socket_id: None,
        input_delta: None,
        input_delta_count: candidate.input_delta_count,
        disabled_reason: Some("full_context_retry".to_string()),
    })
}

fn event_closes_live_retry_window(payload: &serde_json::Value) -> bool {
    !matches!(
        payload.get("type").and_then(|value| value.as_str()),
        Some("codex.rate_limits" | "keepalive")
    )
}

fn is_continuation_retry_error(err: &CodexError) -> bool {
    matches!(
        err.detail.as_deref(),
        Some("previous_response_not_found")
            | Some(super::websocket::WEBSOCKET_CONTINUATION_SOCKET_MISSING_DETAIL)
            | Some(super::websocket::WEBSOCKET_RESPONSE_START_TIMEOUT_DETAIL)
            | Some(super::websocket::WEBSOCKET_MISSING_TERMINAL_DETAIL)
    )
}

fn websocket_pool_key(
    ctx: &RequestContext,
    continuation: Option<&super::continuation::ContinuationCandidate>,
    auth: &StoredAuth,
    base_url: &str,
    use_responses_lite: bool,
) -> Option<SocketPoolKey> {
    let lane_token = ctx.session_id.as_deref()?;
    let continuation = continuation?;
    if continuation.disabled_reason.as_deref() == Some("disabled") {
        return None;
    }
    Some(SocketPoolKey::for_request(
        lane_token,
        base_url,
        auth,
        ProtocolLane::from_responses_lite(use_responses_lite),
    ))
}

fn should_reset_websocket_pool(
    continuation: Option<&super::continuation::ContinuationCandidate>,
) -> bool {
    let Some(reason) = continuation.and_then(|c| c.disabled_reason.as_deref()) else {
        return false;
    };
    reason != "disabled"
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::Message;

    #[test]
    fn normalizes_supported_proxy_urls() {
        assert_eq!(
            normalize_proxy_url("127.0.0.1:8080").as_deref(),
            Some("http://127.0.0.1:8080/")
        );
        assert_eq!(
            normalize_proxy_url("https://user:pass@proxy.example:8443").as_deref(),
            Some("https://user:pass@proxy.example:8443/")
        );
        for scheme in ["socks4", "socks4a"] {
            let proxy = format!("{scheme}://proxy.example:1080");
            assert_eq!(normalize_proxy_url(&proxy), Some(proxy));
        }
        for scheme in ["socks5", "socks5h"] {
            let proxy = format!("{scheme}://user:pass@proxy.example:1080");
            assert_eq!(normalize_proxy_url(&proxy), Some(proxy));
        }
    }

    #[test]
    fn rejects_malformed_or_unsupported_proxy_urls() {
        assert!(normalize_proxy_url("http://").is_none());
        assert!(normalize_proxy_url("ftp://proxy.example:21").is_none());
        assert!(normalize_proxy_url("socks4://user@proxy.example:1080").is_none());
        assert!(normalize_proxy_url("socks4a://user:pass@proxy.example:1080").is_none());
    }

    #[test]
    fn custom_client_auto_fallback_tracks_effective_proxy_route() {
        let proxy = "http://proxy.example:8080";
        let proxied =
            super::super::websocket::WebSocketProxyConfig::new(None, Some(proxy), None, None);
        assert!(!custom_client_auto_http_fallback_enabled(
            "https://codex.invalid/responses",
            &proxied
        ));

        let bypassed = super::super::websocket::WebSocketProxyConfig::new(
            None,
            Some(proxy),
            None,
            Some("codex.invalid"),
        );
        assert!(custom_client_auto_http_fallback_enabled(
            "https://codex.invalid/responses",
            &bypassed
        ));
    }

    fn http_test_auth() -> StoredAuth {
        StoredAuth {
            access: "test".into(),
            refresh: String::new(),
            account_id: Some("acct".into()),
            expires: u64::MAX,
        }
    }

    fn http_test_context() -> RequestContext {
        RequestContext {
            req_id: "http-body-test".into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        }
    }

    fn http_test_client(base_url: String, body_idle_timeout_ms: u64) -> CodexHttpClient {
        CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            base_url,
            100,
            body_idle_timeout_ms,
            0,
        )
    }

    fn buffered_test_request() -> ResponsesRequest {
        ResponsesRequest {
            model: "gpt-5.6-sol".into(),
            instructions: None,
            input: vec![],
            tools: None,
            tool_choice: None,
            store: false,
            stream: true,
            parallel_tool_calls: true,
            include: None,
            client_metadata: None,
            service_tier: None,
            prompt_cache_key: None,
            text: super::super::translate::request::ResponsesText {
                verbosity: None,
                format: None,
            },
            reasoning: None,
        }
    }

    fn authenticated_http_test_client(base_url: String) -> CodexHttpClient {
        let client = http_test_client(base_url, 100);
        client.auth_manager().set_test_auth(http_test_auth());
        client
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut chunk = [0_u8; 4096];
        loop {
            let read = stream.read(&mut chunk).await.unwrap();
            assert!(read > 0, "request ended before its body was complete");
            request.extend_from_slice(&chunk[..read]);
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
                return request;
            }
        }
    }

    async fn write_http_chunk(stream: &mut tokio::net::TcpStream, chunk: &[u8]) {
        stream
            .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
            .await
            .unwrap();
        stream.write_all(chunk).await.unwrap();
        stream.write_all(b"\r\n").await.unwrap();
        stream.flush().await.unwrap();
    }

    #[tokio::test]
    async fn bound_route_uses_one_auth_snapshot_and_refreshes_only_the_next_route() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));
        client.auth_manager().set_test_auth(StoredAuth {
            access: "token-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("acct-a".into()),
            expires: u64::MAX,
        });
        let route_a = client.conversation_route(false).await.unwrap();
        let bound_a = route_a.bind_lane("lane");
        client.auth_manager().set_test_auth(StoredAuth {
            access: "token-b".into(),
            refresh: "refresh-b".into(),
            account_id: Some("acct-b".into()),
            expires: u64::MAX,
        });

        let (no_retry_tx, no_retry_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut first).await;
            let request = String::from_utf8_lossy(&request);
            assert!(request.contains("authorization: Bearer token-a"));
            assert!(request.contains("chatgpt-account-id: acct-a"));
            assert!(request.contains(&format!("session_id: {bound_a}")));
            assert!(request.contains("x-client-request-id: route-a"));
            first
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();

            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err(),
                "the A-bound request must not retry with rotated auth"
            );
            no_retry_tx.send(()).unwrap();

            let (mut second, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut second).await;
            let request = String::from_utf8_lossy(&request);
            assert!(request.contains("authorization: Bearer token-b"));
            assert!(request.contains("chatgpt-account-id: acct-b"));
            assert!(request.contains("x-client-request-id: route-b"));
            assert!(!request.contains("authorization: Bearer token-a"));
            second
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                .await
                .unwrap();
        });

        let mut ctx_a = http_test_context();
        ctx_a.req_id = "route-a".into();
        ctx_a.session_id = Some(route_a.bind_lane("lane"));
        let err = match client
            .post_codex_bound_with_transport(
                &route_a,
                &buffered_test_request(),
                &ctx_a,
                None,
                crate::config::CodexTransport::Http,
            )
            .await
        {
            Err(err) => err,
            Ok(_) => panic!("expected the A-bound request to preserve its 401"),
        };
        assert_eq!(err.status, 401);
        no_retry_rx.await.unwrap();

        let route_b = client.conversation_route(false).await.unwrap();
        let bound_b = route_b.bind_lane("lane");
        assert_ne!(ctx_a.session_id.as_deref(), Some(bound_b.as_str()));
        let mut ctx_b = http_test_context();
        ctx_b.req_id = "route-b".into();
        ctx_b.session_id = Some(bound_b);
        let response = client
            .post_codex_bound_with_transport(
                &route_b,
                &buffered_test_request(),
                &ctx_b,
                None,
                crate::config::CodexTransport::Http,
            )
            .await
            .unwrap();
        assert_eq!(response.status, 200);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn bound_search_preserves_auth_snapshot_and_refreshes_only_next_route() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));
        client.auth_manager().set_test_auth(StoredAuth {
            access: "search-token-a".into(),
            refresh: "search-refresh-a".into(),
            account_id: Some("search-acct-a".into()),
            expires: u64::MAX,
        });
        let route_a = client.conversation_route(false).await.unwrap();
        client.auth_manager().set_test_auth(StoredAuth {
            access: "search-token-b".into(),
            refresh: "search-refresh-b".into(),
            account_id: Some("search-acct-b".into()),
            expires: u64::MAX,
        });

        let (no_retry_tx, no_retry_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.unwrap();
            let request = String::from_utf8_lossy(&read_http_request(&mut first).await).to_string();
            assert!(request.starts_with("POST /alpha/search "));
            assert!(request.contains("authorization: Bearer search-token-a"));
            assert!(request.contains("chatgpt-account-id: search-acct-a"));
            first
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err(),
                "the A-bound search must not retry with rotated auth"
            );
            no_retry_tx.send(()).unwrap();

            let (mut second, _) = listener.accept().await.unwrap();
            let request =
                String::from_utf8_lossy(&read_http_request(&mut second).await).to_string();
            assert!(request.contains("authorization: Bearer search-token-b"));
            assert!(request.contains("chatgpt-account-id: search-acct-b"));
            assert!(!request.contains("authorization: Bearer search-token-a"));
            let response = serde_json::to_vec(&serde_json::json!({
                "encrypted_output": "opaque",
                "output": "search output",
                "results": []
            }))
            .unwrap();
            second
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        response.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            second.write_all(&response).await.unwrap();
        });

        let request = super::super::search::SearchRequest {
            id: "search-session".to_string(),
            model: "gpt-5.6-luna".to_string(),
            reasoning: None,
            input: None,
            commands: super::super::search::SearchCommands {
                search_query: vec![super::super::search::SearchQuery {
                    q: "find Codex".to_string(),
                }],
            },
            settings: super::super::search::SearchSettings {
                filters: None,
                allowed_callers: vec!["direct"],
                external_web_access: true,
            },
            max_output_tokens: 2_500,
        };
        let error = client
            .post_search_bound(&route_a, &request, &http_test_context())
            .await
            .unwrap_err();
        assert_eq!(error.status, 401);
        no_retry_rx.await.unwrap();

        let route_b = client.conversation_route(false).await.unwrap();
        let response = client
            .post_search_bound(&route_b, &request, &http_test_context())
            .await
            .unwrap();
        assert_eq!(response.output, "search output");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn bound_buffered_in_band_unauthorized_refreshes_only_next_route() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));
        client.auth_manager().set_test_auth(StoredAuth {
            access: "buffered-token-a".into(),
            refresh: "buffered-refresh-a".into(),
            account_id: Some("buffered-acct-a".into()),
            expires: u64::MAX,
        });
        let route = client.conversation_route(false).await.unwrap();
        client.auth_manager().set_test_auth(StoredAuth {
            access: "buffered-token-b".into(),
            refresh: "buffered-refresh-b".into(),
            account_id: Some("buffered-acct-b".into()),
            expires: u64::MAX,
        });

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request =
                String::from_utf8_lossy(&read_http_request(&mut stream).await).to_string();
            assert!(request.contains("authorization: Bearer buffered-token-a"));
            let body = b"data: {\"type\":\"response.failed\",\"status_code\":401,\"response\":{\"error\":{\"message\":\"expired\"}}}\n\n";
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            stream.write_all(body).await.unwrap();
            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err(),
                "bound buffered request must not retry with rotated auth"
            );
        });

        let error = match client
            .post_codex_bound_with_transport(
                &route,
                &buffered_test_request(),
                &http_test_context(),
                None,
                crate::config::CodexTransport::Http,
            )
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("expected buffered in-band unauthorized failure"),
        };
        assert_eq!(error.status, 401);
        assert_eq!(error.origin, CodexErrorOrigin::BufferedHttp);
        assert_eq!(
            client.auth_manager().get_auth().await.unwrap().access,
            "buffered-token-b"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn bound_live_unauthorized_event_preserves_401_without_auth_retry() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = Arc::new(http_test_client(format!("http://{addr}/responses"), 100));
        client.auth_manager().set_test_auth(StoredAuth {
            access: "live-token-a".into(),
            refresh: "live-refresh-a".into(),
            account_id: Some("live-acct-a".into()),
            expires: u64::MAX,
        });
        let route = client.conversation_route(false).await.unwrap();
        client.auth_manager().set_test_auth(StoredAuth {
            access: "live-token-b".into(),
            refresh: "live-refresh-b".into(),
            account_id: Some("live-acct-b".into()),
            expires: u64::MAX,
        });

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
            assert!(
                websocket.next().await.is_some(),
                "missing Responses request"
            );
            websocket
                .send(Message::Text(
                    serde_json::json!({
                        "type": "response.failed",
                        "status_code": 401,
                        "response": {"error": {"status": 401, "message": "unauthorized"}}
                    })
                    .to_string(),
                ))
                .await
                .unwrap();
            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err(),
                "the bound live route must not reconnect with rotated auth"
            );
        });

        let mut ctx = http_test_context();
        ctx.req_id = "bound-live-401".into();
        ctx.session_id = Some(route.bind_lane("live-lane"));
        let mut events = client
            .stream_codex_websocket_events_bound(&route, &buffered_test_request(), &ctx, None)
            .await
            .unwrap();
        let item = tokio::time::timeout(Duration::from_secs(2), events.recv())
            .await
            .unwrap()
            .expect("missing unauthorized event");
        let err = match item {
            Err(err) => err,
            Ok(payload) => panic!("unauthorized payload leaked downstream: {payload}"),
        };
        assert_eq!(err.status, 401);
        assert_eq!(err.origin, CodexErrorOrigin::WebSocket);
        assert_eq!(
            client.auth_manager().get_auth().await.unwrap().access,
            "live-token-b"
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn image_request_refreshes_once_after_unauthorized() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/responses"
        )));
        let server_client = client.clone();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let request = read_http_request(&mut stream).await;
                let request = String::from_utf8_lossy(&request);
                if attempt == 0 {
                    assert!(request.contains("authorization: Bearer test"));
                    server_client.auth_manager().set_test_auth(StoredAuth {
                        access: "rotated".into(),
                        refresh: "rotated-refresh".into(),
                        account_id: Some("acct-rotated".into()),
                        expires: u64::MAX,
                    });
                    stream
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                } else {
                    assert!(request.contains("authorization: Bearer rotated"));
                    assert!(request.contains("chatgpt-account-id: acct-rotated"));
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                        )
                        .await
                        .unwrap();
                }
            }
        });

        let response = client
            .post_image_json(
                &format!("http://{addr}"),
                super::super::images::ImageOperation::Generation,
                &serde_json::json!({"model":"gpt-image-2","prompt":"draw"}),
                &http_test_context(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn image_request_does_not_retry_server_errors() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _request = read_http_request(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            assert!(
                tokio::time::timeout(Duration::from_millis(50), listener.accept())
                    .await
                    .is_err()
            );
        });

        let client = authenticated_http_test_client(format!("http://{addr}/responses"));
        let response = client
            .post_image_json(
                &format!("http://{addr}"),
                super::super::images::ImageOperation::Generation,
                &serde_json::json!({"model":"gpt-image-2","prompt":"draw"}),
                &http_test_context(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn image_request_uses_fixed_path_oauth_and_json_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let header_end = request
                .windows(4)
                .position(|part| part == b"\r\n\r\n")
                .unwrap();
            let headers = String::from_utf8_lossy(&request[..header_end]);
            assert!(headers.starts_with("POST /root/images/generations HTTP/1.1"));
            assert!(headers.contains("authorization: Bearer test"));
            assert!(headers.contains("chatgpt-account-id: acct"));
            let body: serde_json::Value =
                serde_json::from_slice(&request[header_end + 4..]).unwrap();
            assert_eq!(body["model"], "gpt-image-2");
            assert_eq!(body["prompt"], "draw a fox");
            let response = br#"{"created":1,"data":[{"b64_json":"aW1n"}]}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                response.len()
            );
            stream.write_all(head.as_bytes()).await.unwrap();
            stream.write_all(response).await.unwrap();
        });

        let client = authenticated_http_test_client(format!("http://{addr}/responses"));
        let response = client
            .post_image_json(
                &format!("http://{addr}/root"),
                super::super::images::ImageOperation::Generation,
                &serde_json::json!({"model":"gpt-image-2","prompt":"draw a fox"}),
                &http_test_context(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn native_responses_replaces_auth_and_preserves_json_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 16 * 1024];
            let read = stream.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..read]);
            assert!(request.contains("authorization: Bearer test"));
            assert!(request.contains("chatgpt-account-id: acct"));
            assert!(request.contains("accept: application/json"));
            assert!(request.contains(r#""extra":{"kept":true}"#));
            let body = br#"{"id":"resp_native","object":"response"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.write_all(body).await.unwrap();
        });

        let client = authenticated_http_test_client(format!("http://{addr}/v1/responses"));
        let response = client
            .post_native_responses(
                &serde_json::json!({
                    "model": "gpt-5.4",
                    "input": "hello",
                    "stream": false,
                    "extra": {"kept": true}
                }),
                &http_test_context(),
                false,
                false,
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(
            response.bytes().await.unwrap(),
            br#"{"id":"resp_native","object":"response"}"#.as_slice()
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn bound_native_sse_401_refreshes_next_route_without_changing_stream() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = authenticated_http_test_client(format!("http://{addr}/v1/responses"));
        client.auth_manager().set_test_auth(StoredAuth {
            access: "stream-token-a".into(),
            refresh: "stream-refresh-a".into(),
            account_id: Some("stream-acct-a".into()),
            expires: u64::MAX,
        });
        let route_a = client.conversation_route(false).await.unwrap();
        let refreshes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        client.native_rejection_refreshes = Some(refreshes.clone());
        let client = Arc::new(client);
        let server_client = client.clone();

        let chunks: Vec<&'static [u8]> = vec![
            b"data: {\"type\":\"response.failed\",\"status_co",
            b"de\":4",
            b"01,\"response\":{\"error\":{\"message\":\"expired\"}}}\r\n",
            b"\r\n",
            b"data: {\"type\":\"response.failed\",\"status_code\":401,\"response\":{\"error\":{\"message\":\"duplicate\"}}}\n\n",
        ];
        let expected = chunks.concat();
        let server_chunks = chunks.clone();
        let (prefix_tx, prefix_rx) = tokio::sync::oneshot::channel();
        let (continue_tx, continue_rx) = tokio::sync::oneshot::channel();
        let (no_retry_tx, no_retry_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.unwrap();
            let request = String::from_utf8_lossy(&read_http_request(&mut first).await).to_string();
            assert!(request.contains("authorization: Bearer stream-token-a"));
            assert!(request.contains("chatgpt-account-id: stream-acct-a"));
            server_client.auth_manager().set_test_auth(StoredAuth {
                access: "stream-token-b".into(),
                refresh: "stream-refresh-b".into(),
                account_id: Some("stream-acct-b".into()),
                expires: u64::MAX,
            });
            first
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nx-native-test: preserved\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            write_http_chunk(&mut first, server_chunks[0]).await;
            prefix_tx.send(()).unwrap();
            continue_rx.await.unwrap();
            for chunk in &server_chunks[1..] {
                write_http_chunk(&mut first, chunk).await;
            }
            first.write_all(b"0\r\n\r\n").await.unwrap();

            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err(),
                "the bound native stream must not retry after an in-band 401"
            );
            no_retry_tx.send(()).unwrap();

            let (mut second, _) = listener.accept().await.unwrap();
            let request =
                String::from_utf8_lossy(&read_http_request(&mut second).await).to_string();
            assert!(request.contains("authorization: Bearer stream-token-b"));
            assert!(request.contains("chatgpt-account-id: stream-acct-b"));
            assert!(!request.contains("authorization: Bearer stream-token-a"));
            second
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                )
                .await
                .unwrap();
        });

        let mut response = client
            .post_native_responses_bound(
                &route_a,
                &serde_json::json!({"model":"gpt-5.4","input":"hello","stream":true}),
                &http_test_context(),
                true,
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(response.headers()["x-native-test"], "preserved");
        assert_eq!(
            response.url().as_str(),
            format!("http://{addr}/v1/responses")
        );

        prefix_rx.await.unwrap();
        let first = tokio::time::timeout(Duration::from_millis(50), response.chunk())
            .await
            .expect("an incomplete event chunk must stream without waiting")
            .unwrap()
            .unwrap();
        assert_eq!(first.as_ref(), chunks[0]);
        continue_tx.send(()).unwrap();

        let mut received = first.to_vec();
        while let Some(chunk) = response.chunk().await.unwrap() {
            received.extend_from_slice(&chunk);
        }
        assert_eq!(received, expected);
        assert_eq!(
            refreshes.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "duplicate unauthorized events must refresh the rejected snapshot once"
        );
        no_retry_rx.await.unwrap();

        let route_b = client.conversation_route(false).await.unwrap();
        let response = client
            .post_native_responses_bound(
                &route_b,
                &serde_json::json!({"model":"gpt-5.4","input":"next"}),
                &http_test_context(),
                false,
            )
            .await
            .unwrap();
        assert_eq!(response.bytes().await.unwrap(), b"{}".as_slice());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn unbound_native_json_401_is_detected_across_chunks() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = authenticated_http_test_client(format!("http://{addr}/v1/responses"));
        client.auth_manager().set_test_auth(StoredAuth {
            access: "json-token-a".into(),
            refresh: "json-refresh-a".into(),
            account_id: Some("json-acct-a".into()),
            expires: u64::MAX,
        });
        let refreshes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        client.native_rejection_refreshes = Some(refreshes.clone());
        let client = Arc::new(client);
        let server_client = client.clone();
        let chunks: Vec<&'static [u8]> = vec![
            b"{\"type\":\"response.failed\",\"status_",
            b"code\":40",
            b"1,\"response\":{\"error\":{\"message\":\"expired\"}}}",
        ];
        let expected = chunks.concat();
        let server_chunks = chunks.clone();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request =
                String::from_utf8_lossy(&read_http_request(&mut stream).await).to_string();
            assert!(request.contains("authorization: Bearer json-token-a"));
            server_client.auth_manager().set_test_auth(StoredAuth {
                access: "json-token-b".into(),
                refresh: "json-refresh-b".into(),
                account_id: Some("json-acct-b".into()),
                expires: u64::MAX,
            });
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            for chunk in server_chunks {
                write_http_chunk(&mut stream, chunk).await;
            }
            stream.write_all(b"0\r\n\r\n").await.unwrap();
        });

        let response = client
            .post_native_responses(
                &serde_json::json!({"model":"gpt-5.4","input":"hello"}),
                &http_test_context(),
                false,
                false,
            )
            .await
            .unwrap();
        assert_eq!(response.bytes().await.unwrap(), expected);
        assert_eq!(refreshes.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(
            client.conversation_route(false).await.unwrap().auth.access,
            "json-token-b"
        );
        server.await.unwrap();
    }

    #[test]
    fn native_failure_detector_bounds_oversized_sse_frames() {
        let mut detector = NativeFailureDetector::new(true);
        assert!(!detector.observe(&vec![b'x'; MAX_NATIVE_FAILURE_EVENT_BYTES + 1]));
        assert!(detector.buffer.len() <= MAX_NATIVE_FAILURE_EVENT_BYTES);
        assert!(!detector.observe(b"\n\n"));
        assert!(detector.observe(
            b"data: {\"type\":\"response.failed\",\"status_code\":401,\"response\":{\"error\":{\"message\":\"expired\"}}}\n\n"
        ));
    }

    #[test]
    fn native_failure_detector_accepts_all_sse_blank_line_endings() {
        const EVENT: &[u8] = b"data: {\"type\":\"response.failed\",\"status_code\":401,\"response\":{\"error\":{\"message\":\"expired\"}}}";
        let delimiters: &[&[u8]] = &[
            b"\n\n",
            b"\n\r",
            b"\n\r\n",
            b"\r\r",
            b"\r\r\n",
            b"\r\n\n",
            b"\r\n\r",
            b"\r\n\r\n",
        ];

        for delimiter in delimiters {
            let mut frame = EVENT.to_vec();
            frame.extend_from_slice(delimiter);
            for split in 0..=frame.len() {
                let mut detector = NativeFailureDetector::new(true);
                let detected =
                    detector.observe(&frame[..split]) || detector.observe(&frame[split..]);
                assert!(
                    detected,
                    "delimiter {delimiter:?} was not detected with split at {split}"
                );
            }
        }
    }

    #[test]
    fn native_failure_detector_parses_bounded_json_only_at_eof() {
        let unauthorized =
            b"{\"type\":\"response.failed\",\"status_code\":401,\"response\":{\"error\":{\"message\":\"expired\"}}}";
        let mut detector = NativeFailureDetector::new(false);
        assert!(!detector.observe(&unauthorized[..32]));
        assert!(!detector.observe(&unauthorized[32..]));
        assert!(detector.finish());
        assert!(!detector.finish(), "a response is detected at most once");

        let mut invalid = NativeFailureDetector::new(false);
        assert!(!invalid.observe(unauthorized));
        assert!(!invalid.observe(b" trailing-garbage"));
        assert!(!invalid.finish(), "a valid JSON prefix is not a valid body");

        let mut interrupted = NativeFailureDetector::new(false);
        assert!(!interrupted.observe(unauthorized));
        interrupted.abort();
        assert!(
            !interrupted.finish(),
            "an interrupted body is not complete JSON"
        );

        let mut oversized = NativeFailureDetector::new(false);
        assert!(!oversized.observe(&vec![b'x'; MAX_NATIVE_FAILURE_EVENT_BYTES + 1]));
        assert!(oversized.buffer.len() <= MAX_NATIVE_FAILURE_EVENT_BYTES);
        assert!(!oversized.finish());
    }

    #[tokio::test]
    async fn bound_native_responses_validates_route_against_payload_metadata() {
        let client = authenticated_http_test_client("http://127.0.0.1:1/responses".to_string());
        let ctx = http_test_context();
        let full_route = client.conversation_route(false).await.unwrap();
        let full_error = match client
            .post_native_responses_bound(
                &full_route,
                &serde_json::json!({
                    "model":"gpt-5.4",
                    "input":"hello",
                    "client_metadata":{"lite":"true"}
                }),
                &ctx,
                false,
            )
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("Lite payload must not use a Full route"),
        };
        assert_eq!(full_error.status, 500);

        let lite_route = client.conversation_route(true).await.unwrap();
        let lite_error = match client
            .post_native_responses_bound(
                &lite_route,
                &serde_json::json!({"model":"gpt-5.6-sol","input":"hello"}),
                &ctx,
                false,
            )
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("Full payload must not use a Lite route"),
        };
        assert_eq!(lite_error.status, 500);
    }

    #[tokio::test]
    async fn bound_native_responses_preserves_route_identity_and_does_not_retry_401() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/v1/responses"
        )));
        client.auth_manager().set_test_auth(StoredAuth {
            access: "native-token-a".into(),
            refresh: "native-refresh-a".into(),
            account_id: Some("native-acct-a".into()),
            expires: u64::MAX,
        });
        let route_a = client.conversation_route(false).await.unwrap();
        let bound_a = route_a.bind_lane("native-raw-lane");
        client.auth_manager().set_test_auth(StoredAuth {
            access: "native-token-b".into(),
            refresh: "native-refresh-b".into(),
            account_id: Some("native-acct-b".into()),
            expires: u64::MAX,
        });

        let (no_retry_tx, no_retry_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut first).await;
            let request = String::from_utf8_lossy(&request);
            assert!(request.contains("authorization: Bearer native-token-a"));
            assert!(request.contains("chatgpt-account-id: native-acct-a"));
            assert!(request.contains(&format!("session_id: {bound_a}")));
            assert!(request.contains(&format!("x-codex-window-id: {bound_a}:0")));
            assert!(request.contains("x-client-request-id: native-route-a"));
            assert!(!request.contains("session_id: native-raw-lane"));
            first
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();

            assert!(
                tokio::time::timeout(Duration::from_millis(75), listener.accept())
                    .await
                    .is_err(),
                "the bound native request must not retry with rotated auth"
            );
            no_retry_tx.send(()).unwrap();

            let (mut second, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut second).await;
            let request = String::from_utf8_lossy(&request);
            assert!(request.contains("authorization: Bearer native-token-b"));
            assert!(request.contains("chatgpt-account-id: native-acct-b"));
            assert!(request.contains("x-client-request-id: native-route-b"));
            assert!(!request.contains("authorization: Bearer native-token-a"));
            second
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                )
                .await
                .unwrap();
        });

        let mut ctx_a = http_test_context();
        ctx_a.req_id = "native-route-a".into();
        ctx_a.session_id = Some(route_a.bind_lane("native-raw-lane"));
        let response = client
            .post_native_responses_bound(
                &route_a,
                &serde_json::json!({"model":"gpt-5.4","input":"hello"}),
                &ctx_a,
                false,
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
        no_retry_rx.await.unwrap();

        let route_b = client.conversation_route(false).await.unwrap();
        let bound_b = route_b.bind_lane("native-raw-lane");
        assert_ne!(ctx_a.session_id.as_deref(), Some(bound_b.as_str()));
        let mut ctx_b = http_test_context();
        ctx_b.req_id = "native-route-b".into();
        ctx_b.session_id = Some(bound_b);
        let response = client
            .post_native_responses_bound(
                &route_b,
                &serde_json::json!({"model":"gpt-5.4","input":"hello"}),
                &ctx_b,
                false,
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn native_responses_refreshes_once_before_returning_body() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = Arc::new(authenticated_http_test_client(format!(
            "http://{addr}/v1/responses"
        )));
        let server_client = client.clone();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0_u8; 16 * 1024];
                let read = stream.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..read]);
                if attempt == 0 {
                    assert!(request.contains("authorization: Bearer test"));
                    server_client.auth_manager().set_test_auth(StoredAuth {
                        access: "rotated".into(),
                        refresh: "rotated-refresh".into(),
                        account_id: Some("acct-rotated".into()),
                        expires: u64::MAX,
                    });
                    stream
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                        )
                        .await
                        .unwrap();
                } else {
                    assert!(request.contains("authorization: Bearer rotated"));
                    assert!(request.contains("chatgpt-account-id: acct-rotated"));
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                        )
                        .await
                        .unwrap();
                }
            }
        });

        let response = client
            .post_native_responses(
                &serde_json::json!({"model":"gpt-5.4","input":"hello"}),
                &http_test_context(),
                false,
                false,
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(response.bytes().await.unwrap(), b"{}".as_slice());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn native_responses_does_not_follow_redirects() {
        let source = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let source_addr = source.local_addr().unwrap();
        let target = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let target_addr = target.local_addr().unwrap();
        let source_server = tokio::spawn(async move {
            let (mut stream, _) = source.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            let response = format!(
                "HTTP/1.1 302 Found\r\nlocation: http://{target_addr}/stolen\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        let client = authenticated_http_test_client(format!("http://{source_addr}/v1/responses"));
        let response = client
            .post_native_responses(
                &serde_json::json!({"model":"gpt-5.4","input":"hello"}),
                &http_test_context(),
                false,
                false,
            )
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::FOUND);
        source_server.await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(50), target.accept())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn buffered_http_retries_retryable_status() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0_u8; 16 * 1024];
                assert!(stream.read(&mut request).await.unwrap() > 0);
                let (status, body): (&str, &[u8]) = if attempt == 0 {
                    ("503 Service Unavailable", b"retry")
                } else {
                    ("200 OK", b"data: keep\n\n")
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-length: {}\r\nretry-after: 0\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                stream.write_all(body).await.unwrap();
            }
        });

        let response = authenticated_http_test_client(format!("http://{addr}/responses"))
            .post_codex_with_transport(
                &buffered_test_request(),
                &http_test_context(),
                None,
                crate::config::CodexTransport::Http,
            )
            .await
            .unwrap();
        server.await.unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"data: keep\n\n");
    }

    #[tokio::test]
    async fn standalone_search_posts_json_to_alpha_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let header_end = request
                .windows(4)
                .position(|part| part == b"\r\n\r\n")
                .unwrap();
            let headers = String::from_utf8_lossy(&request[..header_end]);
            assert!(headers.starts_with("POST /alpha/search HTTP/1.1"));
            assert!(
                headers
                    .to_ascii_lowercase()
                    .contains("accept: application/json")
            );
            assert!(headers.contains("authorization: Bearer test"));
            let body: serde_json::Value =
                serde_json::from_slice(&request[header_end + 4..]).unwrap();
            assert_eq!(body["model"], "gpt-5.6-luna");
            assert!(body.get("reasoning").is_none());
            assert_eq!(body["commands"]["search_query"][0]["q"], "find Codex");

            let response = serde_json::to_vec(&serde_json::json!({
                "encrypted_output": "opaque",
                "output": "search output",
                "results": [{
                    "type": "text_result",
                    "ref_id": "turn0search0",
                    "url": "https://example.com",
                    "title": "Example"
                }]
            }))
            .unwrap();
            let response_headers = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                response.len()
            );
            stream.write_all(response_headers.as_bytes()).await.unwrap();
            stream.write_all(&response).await.unwrap();
        });

        let client = authenticated_http_test_client(format!("http://{addr}/responses"));
        let request = super::super::search::SearchRequest {
            id: "session".to_string(),
            model: "gpt-5.6-luna".to_string(),
            reasoning: None,
            input: None,
            commands: super::super::search::SearchCommands {
                search_query: vec![super::super::search::SearchQuery {
                    q: "find Codex".to_string(),
                }],
            },
            settings: super::super::search::SearchSettings {
                filters: None,
                allowed_callers: vec!["direct"],
                external_web_access: true,
            },
            max_output_tokens: 2_500,
        };
        let response = client
            .post_search(&request, &http_test_context())
            .await
            .unwrap();
        server.await.unwrap();
        assert_eq!(response.output, "search output");
        assert_eq!(response.results.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn auto_falls_back_to_http_after_statusful_websocket_handshake_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut websocket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 16 * 1024];
            let read = websocket.read(&mut request).await.unwrap();
            assert!(read > 0);
            assert!(
                String::from_utf8_lossy(&request[..read])
                    .to_ascii_lowercase()
                    .contains("upgrade: websocket")
            );
            websocket
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 13\r\nconnection: close\r\n\r\npolicy denied",
                )
                .await
                .unwrap();
            drop(websocket);

            let (mut http, _) = listener.accept().await.unwrap();
            let read = http.read(&mut request).await.unwrap();
            assert!(read > 0);
            assert!(String::from_utf8_lossy(&request[..read]).starts_with("POST "));
            let body = b"data: keep\n\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            http.write_all(response.as_bytes()).await.unwrap();
            http.write_all(body).await.unwrap();
        });

        let response = authenticated_http_test_client(format!("http://{addr}/responses"))
            .post_codex_with_transport(
                &buffered_test_request(),
                &http_test_context(),
                None,
                crate::config::CodexTransport::Auto,
            )
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"data: keep\n\n");
    }

    #[tokio::test]
    async fn over_budget_retry_after_stops_without_replay() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 16 * 1024];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            stream
                .write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 4\r\nretry-after: 120\r\nconnection: close\r\n\r\nstop",
                )
                .await
                .unwrap();
        });

        let error = match authenticated_http_test_client(format!("http://{addr}/responses"))
            .post_codex_with_transport(
                &buffered_test_request(),
                &http_test_context(),
                None,
                crate::config::CodexTransport::Http,
            )
            .await
        {
            Ok(_) => panic!("over-budget Retry-After should propagate"),
            Err(error) => error,
        };
        server.await.unwrap();
        assert_eq!(error.status, 503);
        assert_eq!(error.retry_after.as_deref(), Some("120"));
    }

    #[tokio::test]
    async fn buffered_http_rejects_non_retryable_error_status_before_sse_parsing() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 16 * 1024];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            let body = br#"{"error":{"message":"Model not found gpt-test"}}"#;
            let response = format!(
                "HTTP/1.1 404 Not Found\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.write_all(body).await.unwrap();
        });

        let result = authenticated_http_test_client(format!("http://{addr}/responses"))
            .post_codex_with_transport(
                &buffered_test_request(),
                &http_test_context(),
                None,
                crate::config::CodexTransport::Http,
            )
            .await;
        server.await.unwrap();
        let error = match result {
            Ok(_) => panic!("non-success HTTP status must not reach the SSE reducer"),
            Err(error) => error,
        };

        assert_eq!(error.status, 404);
        assert_eq!(error.detail.as_deref(), Some("Model not found gpt-test"));
        assert_eq!(error.origin, CodexErrorOrigin::BufferedHttp);
    }

    #[test]
    fn status_error_preserves_buffered_websocket_event_message() {
        let error = codex_status_error(CodexResponse {
            body: b"data: {\"type\":\"error\",\"error\":{\"status\":400,\"message\":\"bad request\"}}\n\n"
                .to_vec(),
            status: 400,
            headers: Vec::new(),
            transport: ActualTransport::WebSocket,
            socket_id: Some(1),
        });

        assert_eq!(error.status, 400);
        assert_eq!(error.detail.as_deref(), Some("bad request"));
        assert_eq!(error.origin, CodexErrorOrigin::BufferedWebSocket);
    }

    #[tokio::test]
    async fn active_http_body_can_exceed_header_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            for chunk in [b"a".as_slice(), b"b", b"c"] {
                stream.write_all(b"1\r\n").await.unwrap();
                stream.write_all(chunk).await.unwrap();
                stream.write_all(b"\r\n").await.unwrap();
                tokio::time::sleep(Duration::from_millis(45)).await;
            }
            stream.write_all(b"0\r\n\r\n").await.unwrap();
        });

        let response = http_test_client(format!("http://{addr}/responses"), 80)
            .attempt_post_http(&http_test_auth(), "{}", &http_test_context(), false)
            .await
            .expect("active body should not hit a whole-request timeout");
        server.await.unwrap();

        assert_eq!(response.body, b"abc");
    }

    #[tokio::test]
    async fn stalled_http_body_hits_idle_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 1\r\n\r\n")
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
        });

        let result = http_test_client(format!("http://{addr}/responses"), 30)
            .attempt_post_http(&http_test_auth(), "{}", &http_test_context(), false)
            .await;
        server.await.unwrap();
        let error = result.err().expect("stalled body should time out");

        assert!(error.message.contains("next Codex response body chunk"));
        assert_eq!(error.detail.as_deref(), Some("http_response_body"));
    }

    #[tokio::test]
    async fn reset_http_body_returns_transport_error() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 10\r\n\r\npartial")
                .await
                .unwrap();
        });

        let result = http_test_client(format!("http://{addr}/responses"), 100)
            .attempt_post_http(&http_test_auth(), "{}", &http_test_context(), false)
            .await;
        server.await.unwrap();
        let error = result.err().expect("truncated body should fail");

        assert!(
            error
                .message
                .contains("Transport error reading Codex response body")
        );
        assert_eq!(error.detail.as_deref(), Some("http_response_body"));
    }

    #[test]
    fn codex_error_display() {
        let err = CodexError {
            status: 429,
            message: "Rate limited".to_string(),
            detail: Some("body".to_string()),
            retry_after: Some("5".to_string()),
            origin: CodexErrorOrigin::Http,
        };
        let display = format!("{err}");
        assert!(display.contains("429"));
        assert!(display.contains("Rate limited"));
    }

    #[test]
    fn websocket_pre_request_502_is_retryable() {
        let err = CodexError {
            status: 502,
            message: "WebSocket connect error".to_string(),
            detail: Some("websocket_pre_request".to_string()),
            retry_after: Some("3".to_string()),
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(is_retryable_transport_error(&err));
    }

    #[test]
    fn proxy_tunnel_rejection_is_not_retried_or_used_for_http_fallback() {
        let err = CodexError {
            status: 0,
            message: "WebSocket proxy tunnel was rejected".to_string(),
            detail: Some(
                super::super::websocket::WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL.to_string(),
            ),
            retry_after: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        };

        assert!(!is_retryable_transport_error(&err));
        assert!(!should_fallback_to_http(&err));
    }

    #[test]
    fn websocket_pre_request_statusless_error_is_retryable() {
        let err = CodexError {
            status: 0,
            message: "WebSocket connect timeout after 15000ms".to_string(),
            detail: Some("websocket_pre_request".to_string()),
            retry_after: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(is_retryable_transport_error(&err));
    }

    #[test]
    fn websocket_pre_request_400_is_not_retryable() {
        let err = CodexError {
            status: 400,
            message: "WebSocket connect error".to_string(),
            detail: Some("websocket_pre_request".to_string()),
            retry_after: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(!is_retryable_transport_error(&err));
    }

    #[test]
    fn statusless_transport_error_matching_is_case_insensitive() {
        let err = CodexError {
            status: 0,
            message: "WebSocket protocol error: Connection reset without closing handshake"
                .to_string(),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(is_retryable_transport_error(&err));
    }

    #[test]
    fn image_headers_reuse_oauth_without_responses_beta_headers() {
        let auth = StoredAuth {
            access: "tok".into(),
            refresh: String::new(),
            account_id: Some("acct".into()),
            expires: u64::MAX,
        };
        let headers = build_codex_image_headers(&auth, &http_test_context()).unwrap();

        assert_eq!(
            headers.get(http::header::AUTHORIZATION).unwrap(),
            "Bearer tok"
        );
        assert_eq!(headers.get("chatgpt-account-id").unwrap(), "acct");
        assert_eq!(
            headers.get(http::header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        assert_eq!(
            headers.get(http::header::ACCEPT).unwrap(),
            "application/json"
        );
        assert!(headers.get("openai-beta").is_none());
        assert!(headers.get("x-codex-beta-features").is_none());
    }

    #[test]
    fn codex_headers_include_session_and_beta() {
        let auth = StoredAuth {
            access: "tok".into(),
            refresh: String::new(),
            account_id: Some("acct".into()),
            expires: u64::MAX,
        };
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: Some("s".into()),
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let headers = build_codex_headers(&auth, &ctx, false).unwrap();
        assert_eq!(
            headers.get("openai-beta").unwrap(),
            "responses=experimental"
        );
        assert_eq!(headers.get("session_id").unwrap(), "s");
        assert_eq!(
            headers.get("x-codex-beta-features").unwrap(),
            "remote_compaction_v2"
        );
    }

    #[test]
    fn codex_headers_include_responses_lite_when_requested() {
        let auth = StoredAuth {
            access: "tok".into(),
            refresh: String::new(),
            account_id: None,
            expires: u64::MAX,
        };
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let headers = build_codex_headers(&auth, &ctx, true).unwrap();
        assert_eq!(
            headers
                .get("x-openai-internal-codex-responses-lite")
                .unwrap(),
            "true"
        );
        assert_eq!(headers.get("originator").unwrap(), "codex_cli_rs");
        assert_eq!(default_user_agent(true), "codex_cli_rs");
    }

    #[test]
    fn codex_headers_omit_session_when_missing() {
        let auth = StoredAuth {
            access: "tok".into(),
            refresh: String::new(),
            account_id: None,
            expires: u64::MAX,
        };
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let headers = build_codex_headers(&auth, &ctx, false).unwrap();
        assert!(headers.get("session_id").is_none());
        assert!(headers.get("x-client-request-id").is_none());
    }

    #[test]
    fn codex_headers_return_error_for_invalid_session_header() {
        let auth = StoredAuth {
            access: "tok".into(),
            refresh: String::new(),
            account_id: None,
            expires: u64::MAX,
        };
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: Some("bad\nsession".into()),
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let err = build_codex_headers(&auth, &ctx, false).unwrap_err();
        assert_eq!(err.status, 500);
        assert!(err.message.contains("session_id"));
    }

    #[test]
    fn build_websocket_request_removes_stream() {
        let input = vec![
            super::super::translate::request::ResponsesInputItem::Message {
                role: "user".to_string(),
                content: vec![
                    super::super::translate::request::ResponsesContentPart::InputText {
                        text: "hello".to_string(),
                    },
                ],
            },
        ];
        let req = ResponsesRequest {
            model: "gpt-5.5".to_string(),
            instructions: None,
            input,
            tools: None,
            tool_choice: None,
            store: false,
            stream: true,
            parallel_tool_calls: true,
            include: None,
            client_metadata: None,
            service_tier: None,
            prompt_cache_key: None,
            text: super::super::translate::request::ResponsesText {
                verbosity: Some("low".to_string()),
                format: None,
            },
            reasoning: None,
        };
        let payload = build_websocket_request(&req, None);
        assert_eq!(
            payload.get("type").and_then(|v| v.as_str()),
            Some("response.create")
        );
        assert!(payload.get("stream").is_none());
        assert!(payload.get("previous_response_id").is_none());
    }

    #[test]
    fn websocket_pool_key_tracks_continuation_opt_in() {
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: Some("session".into()),
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let disabled = super::super::continuation::ContinuationCandidate {
            turn_id: None,
            previous_response_id: None,
            socket_id: None,
            input_delta: None,
            input_delta_count: 1,
            disabled_reason: Some("disabled".into()),
        };
        let first_enabled = super::super::continuation::ContinuationCandidate {
            turn_id: None,
            previous_response_id: None,
            socket_id: None,
            input_delta: None,
            input_delta_count: 1,
            disabled_reason: Some("missing_state".into()),
        };
        let append = super::super::continuation::ContinuationCandidate {
            turn_id: None,
            previous_response_id: Some("resp_1".into()),
            socket_id: Some(1),
            input_delta: None,
            input_delta_count: 1,
            disabled_reason: None,
        };

        let auth = StoredAuth {
            access: "token".into(),
            refresh: String::new(),
            account_id: Some("account".into()),
            expires: u64::MAX,
        };
        assert!(
            websocket_pool_key(
                &ctx,
                Some(&disabled),
                &auth,
                "https://example.test/responses",
                false,
            )
            .is_none()
        );
        let first_key = websocket_pool_key(
            &ctx,
            Some(&first_enabled),
            &auth,
            "https://example.test/responses",
            false,
        )
        .unwrap();
        let append_key = websocket_pool_key(
            &ctx,
            Some(&append),
            &auth,
            "https://example.test/responses",
            false,
        )
        .unwrap();
        assert_eq!(first_key, append_key);
        assert!(!first_key.as_str().contains("session"));
    }

    #[test]
    fn websocket_pool_reset_clears_initial_stale_state() {
        let missing_state = super::super::continuation::ContinuationCandidate {
            turn_id: None,
            previous_response_id: None,
            socket_id: None,
            input_delta: None,
            input_delta_count: 1,
            disabled_reason: Some("missing_state".into()),
        };
        let disabled = super::super::continuation::ContinuationCandidate {
            turn_id: None,
            previous_response_id: None,
            socket_id: None,
            input_delta: None,
            input_delta_count: 1,
            disabled_reason: Some("disabled".into()),
        };
        let prompt_changed = super::super::continuation::ContinuationCandidate {
            turn_id: None,
            previous_response_id: None,
            socket_id: None,
            input_delta: None,
            input_delta_count: 1,
            disabled_reason: Some("prompt_changed".into()),
        };

        assert!(should_reset_websocket_pool(Some(&missing_state)));
        assert!(!should_reset_websocket_pool(Some(&disabled)));
        assert!(should_reset_websocket_pool(Some(&prompt_changed)));
    }

    #[test]
    fn build_codex_headers_error_on_empty_access() {
        let auth = StoredAuth {
            access: "".into(),
            refresh: String::new(),
            account_id: None,
            expires: u64::MAX,
        };
        let ctx = RequestContext {
            req_id: "r".into(),
            session_id: None,
            session_seq: None,
            provider: "codex".into(),
            traffic: None,
            monitor: None,
        };
        let result = build_codex_headers(&auth, &ctx, false);
        assert!(
            result.is_ok(),
            "empty access should still produce valid Bearer header"
        );
    }

    #[test]
    fn codex_header_timeout_error_display() {
        let err = CodexHeaderTimeoutError { timeout_ms: 60000 };
        let display = format!("{err}");
        assert!(display.contains("60000"));
    }

    #[test]
    fn codex_transport_error_display() {
        let err = CodexTransportError {
            message: "connection reset".to_string(),
        };
        let display = format!("{err}");
        assert!(display.contains("connection reset"));
    }

    #[test]
    fn unauthorized_retry_distinguishes_auto_and_strict_websocket_handshakes() {
        let http_unauthorized = Ok(CodexResponse {
            body: Vec::new(),
            status: 401,
            headers: Vec::new(),
            transport: ActualTransport::Http,
            socket_id: None,
        });
        let websocket_unauthorized = Err(CodexError {
            status: 401,
            message: "WebSocket connect error".to_string(),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::WebSocket,
        });
        let forbidden = Err(CodexError {
            status: 403,
            message: "Forbidden".to_string(),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::WebSocket,
        });
        let rejected_handshake = Err(CodexError {
            status: 401,
            message: "WebSocket connect error".to_string(),
            detail: Some("policy denied".to_string()),
            retry_after: None,
            origin: CodexErrorOrigin::WebSocketHandshake,
        });
        let rejected_handshake_err = match &rejected_handshake {
            Err(error) => error,
            Ok(_) => panic!("expected rejected handshake"),
        };

        assert!(should_refresh_after_unauthorized(
            &http_unauthorized,
            false,
            crate::config::CodexTransport::Auto
        ));
        assert!(should_refresh_after_unauthorized(
            &websocket_unauthorized,
            false,
            crate::config::CodexTransport::Auto
        ));
        assert!(!should_refresh_after_unauthorized(
            &forbidden,
            false,
            crate::config::CodexTransport::Auto
        ));
        assert!(!should_refresh_after_unauthorized(
            &rejected_handshake,
            false,
            crate::config::CodexTransport::Auto
        ));
        assert!(should_refresh_after_unauthorized(
            &rejected_handshake,
            false,
            crate::config::CodexTransport::WebSocket
        ));
        assert!(!should_refresh_after_unauthorized(
            &http_unauthorized,
            true,
            crate::config::CodexTransport::Auto
        ));
        assert!(should_fallback_to_http(rejected_handshake_err));
    }

    #[test]
    fn informational_events_keep_live_continuation_retry_available() {
        assert!(!event_closes_live_retry_window(&serde_json::json!({
            "type": "codex.rate_limits",
            "rate_limits": {"limit_reached": false}
        })));
        assert!(!event_closes_live_retry_window(&serde_json::json!({
            "type": "keepalive"
        })));
        assert!(event_closes_live_retry_window(&serde_json::json!({
            "type": "response.created"
        })));
    }

    #[test]
    fn continuation_retry_requires_previous_response_id() {
        let append = super::super::continuation::ContinuationCandidate {
            turn_id: None,
            previous_response_id: Some("resp_1".into()),
            socket_id: Some(1),
            input_delta: None,
            input_delta_count: 1,
            disabled_reason: None,
        };
        let initial = super::super::continuation::ContinuationCandidate {
            turn_id: None,
            previous_response_id: None,
            socket_id: None,
            input_delta: None,
            input_delta_count: 1,
            disabled_reason: Some("missing_state".into()),
        };
        let timeout = CodexError {
            status: 0,
            message: "WebSocket response start timeout after 60000ms".to_string(),
            detail: Some(
                super::super::websocket::WEBSOCKET_RESPONSE_START_TIMEOUT_DETAIL.to_string(),
            ),
            retry_after: None,
            origin: CodexErrorOrigin::WebSocket,
        };
        let missing = CodexError {
            status: 0,
            message: "Previous response not found".to_string(),
            detail: Some("previous_response_not_found".to_string()),
            retry_after: None,
            origin: CodexErrorOrigin::WebSocket,
        };
        let idle = CodexError {
            status: 0,
            message: "WebSocket idle timeout after 60000ms".to_string(),
            detail: None,
            retry_after: None,
            origin: CodexErrorOrigin::WebSocket,
        };

        assert!(should_retry_without_continuation(&timeout, Some(&append)));
        assert!(should_retry_without_continuation(&missing, Some(&append)));
        assert!(!should_retry_without_continuation(&idle, Some(&append)));
        assert!(!should_retry_without_continuation(&timeout, Some(&initial)));
        assert!(!should_retry_without_continuation(&timeout, None));
    }
}
