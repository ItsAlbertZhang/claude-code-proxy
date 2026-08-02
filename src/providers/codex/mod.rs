pub mod auth;
pub mod chat_completions;
pub mod client;
pub mod compaction;
pub mod continuation;
pub mod count_tokens;
pub(crate) mod events;
pub mod images;
pub mod native;
pub mod request_summary;
pub mod search;
pub mod state;
pub mod transcription;
pub mod translate;
pub mod websocket;

use async_trait::async_trait;
use axum::Json;
use axum::body::Body;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use http::StatusCode;
use std::sync::Arc;
use std::time::Instant;

use crate::anthropic::error::json_error;
use crate::anthropic::schema::{CountTokensResponse, MessagesRequest};
use crate::anthropic::sse::parse_sse_events;
use crate::config;
use crate::logging::create_logger;
use crate::monitor::usage_from_anthropic_sse;
use crate::provider::{CliHandlers, Provider, RequestContext};
use crate::registry;
use crate::request_identity::RequestScope;
use crate::retry::{compute_backoff_delay, sleep};

use self::auth::browser_login::run_browser_login;
use self::auth::device::DeviceAuthClient;
use self::auth::manager::CodexAuthManager;
use self::auth::token_store::file_store;
use self::client::{AuthRejectionBudget, BufferedRetryState, CodexHttpClient};
use self::compaction::{
    CompactionError, CompactionLease, abort_compaction_attempt, activate_compaction,
    apply_compaction_replay, begin_compaction_with_permit, request_compaction,
    reserve_compaction_start, store_compaction,
};
use self::continuation::{
    ContinuationCandidate, abort_continuation, continuation_candidate_for_turn,
    record_continuation, reserve_continuation_turn,
};
use self::count_tokens::count_translated_tokens;
use self::translate::accumulate::accumulate_response_with_traffic_in_scope;
use self::translate::live_stream::LiveStreamTranslator;
use self::translate::model_allowlist::{
    assert_allowed_model, full_lane_web_search_model, resolve_model_request_with_config_override,
    uses_responses_lite,
};
use self::translate::reducer::finish_metadata_from_upstream_in_scope;
use self::translate::request::{
    TranslateOptions, has_hosted_web_search, is_compact_messages_request, translate_request,
};

const MAX_RETRYABLE_LIVE_STREAM_RETRIES: u32 = 10;
const MAX_EMPTY_COMPLETION_RETRIES: u32 = 10;
const EMPTY_CODEX_COMPLETION_DETAIL: &str = "empty_codex_completion";
use self::translate::stream::translate_stream_bytes_with_traffic_in_scope;

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

pub(crate) fn clear_lane_compactions(lane_token: &str) {
    compaction::clear_compactions_for_lane(lane_token);
}

pub struct CodexProvider {
    client: Arc<CodexHttpClient>,
    #[cfg(test)]
    server_compaction_override: Option<bool>,
}

impl Default for CodexProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexProvider {
    pub fn new() -> Self {
        Self {
            client: Arc::new(CodexHttpClient::new()),
            #[cfg(test)]
            server_compaction_override: None,
        }
    }

    #[cfg(test)]
    fn with_client(client: CodexHttpClient) -> Self {
        Self {
            client: Arc::new(client),
            server_compaction_override: None,
        }
    }

    #[cfg(test)]
    fn with_server_compaction_for_test(mut self) -> Self {
        self.server_compaction_override = Some(true);
        self
    }

    fn server_compaction_enabled(&self) -> bool {
        #[cfg(test)]
        if let Some(enabled) = self.server_compaction_override {
            return enabled;
        }
        config::codex_server_compaction()
    }
}

#[async_trait]
impl Provider for CodexProvider {
    fn name(&self) -> &'static str {
        "codex"
    }

    fn supported_models(&self) -> Vec<String> {
        let mut models: Vec<String> = registry::CODEX_MODELS
            .iter()
            .map(|m| m.to_string())
            .collect();
        for m in registry::CODEX_MODELS {
            models.push(format!("{m}-fast"));
        }
        models.sort_unstable();
        models.dedup();
        models
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &CODEX_CLI
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let message_id = format!("msg_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
        let want_stream = body.stream;
        let model = body.model.as_deref().unwrap_or("gpt-5.6-sol");

        let mut resolved =
            resolve_model_request_with_config_override(model, !body.bypass_provider_model_override);
        if let Err(e) = assert_allowed_model(&resolved.model) {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "Model \"{model}\" resolves to unsupported model \"{}\"",
                    e.model
                ),
            );
        }
        if search::is_standalone_search_request(&body) {
            if let Some(monitor) = ctx.monitor.as_ref() {
                monitor.model_resolved(&ctx.req_id, &resolved.model);
            }
            let lane_token = ctx.session_id.clone();
            let (base_search_request, query) = match search::build_search_request(
                &body,
                &resolved.model,
                ctx.session_id.as_deref(),
            ) {
                Ok(request) => request,
                Err(error) => {
                    return json_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        error.to_string(),
                    );
                }
            };
            let mut route = match self.client.conversation_route(false).await {
                Ok(route) => route,
                Err(error) => return map_codex_error_to_response(&error),
            };
            let log = create_logger("codex");
            let started_at = Instant::now();
            log.info(
                "codex_standalone_search_started",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), serde_json::json!(&ctx.req_id)),
                    ("model".to_string(), serde_json::json!(&resolved.model)),
                    ("stream".to_string(), serde_json::json!(want_stream)),
                ])),
            );
            if let Some(monitor) = ctx.monitor.as_ref() {
                monitor.upstream_started(&ctx.req_id);
            }
            let mut retry_state = BufferedRetryState::default();
            let mut rebind_available = true;
            let (search_response, search_request) = loop {
                let mut search_request = base_search_request.clone();
                let mut route_ctx = ctx.clone();
                if let Some(lane_token) = lane_token.as_deref() {
                    let bound_lane = route.bind_lane(lane_token);
                    search_request.id = bound_lane.clone();
                    route_ctx.session_id = Some(bound_lane);
                }
                match self
                    .client
                    .post_search_bound_with_retry_state(
                        &route,
                        &search_request,
                        &route_ctx,
                        &mut retry_state,
                    )
                    .await
                {
                    Ok(response) => break (response, search_request),
                    Err(error) => {
                        if error.status == 401 && rebind_available {
                            rebind_available = false;
                            if let Some(next_route) = self
                                .client
                                .refresh_conversation_route_after_rejection(&route)
                                .await
                            {
                                route = next_route;
                                continue;
                            }
                        }
                        log.warn(
                            "codex_standalone_search_failed",
                            Some(serde_json::Map::from_iter([
                                ("reqId".to_string(), serde_json::json!(&ctx.req_id)),
                                ("model".to_string(), serde_json::json!(&resolved.model)),
                                ("status".to_string(), serde_json::json!(error.status)),
                                (
                                    "ms".to_string(),
                                    serde_json::json!(started_at.elapsed().as_millis()),
                                ),
                            ])),
                        );
                        return map_codex_error_to_response(&error);
                    }
                }
            };
            log.info(
                "codex_standalone_search_completed",
                Some(serde_json::Map::from_iter([
                    ("reqId".to_string(), serde_json::json!(&ctx.req_id)),
                    ("model".to_string(), serde_json::json!(&resolved.model)),
                    (
                        "resultCount".to_string(),
                        serde_json::json!(search_response.results.as_ref().map(Vec::len)),
                    ),
                    (
                        "ms".to_string(),
                        serde_json::json!(started_at.elapsed().as_millis()),
                    ),
                ])),
            );
            let input_tokens = search::search_request_input_tokens(&search_request);
            let output_tokens = search::search_response_output_tokens(&search_response);
            if let Some(monitor) = ctx.monitor.as_ref() {
                monitor.usage_updated(&ctx.req_id, Some(input_tokens), Some(output_tokens));
            }
            return search::anthropic_search_response(
                &search_response,
                &query,
                &message_id,
                model,
                want_stream,
                input_tokens,
                ctx.traffic.as_deref(),
            );
        }
        let use_responses_lite = apply_model_lane_for_request(&mut resolved.model, &body);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &resolved.model);
        }
        let lane_token = ctx.session_id.clone();
        let read_rewrite_scope = lane_token.clone();
        let compact_boundary = is_compact_messages_request(&body);
        let server_compaction_enabled = self.server_compaction_enabled();
        let mut route = match self.client.conversation_route(use_responses_lite).await {
            Ok(route) => route,
            Err(error) => return map_codex_error_to_response(&error),
        };
        let auth_rejection_budget = Arc::new(AuthRejectionBudget::default());
        let mut route_rebuilt = false;
        let mut buffered_retry_state = BufferedRetryState::default();
        let mut live_start_attempt = 0_u32;
        let mut empty_completion_attempt = 0_u32;
        let mut upstream_started = false;
        let previous_response_id_enabled = self.client.previous_response_id_enabled();
        let logical_turn_id =
            reserve_continuation_turn(lane_token.as_deref(), previous_response_id_enabled);
        let compaction_start = (server_compaction_enabled && compact_boundary)
            .then(|| reserve_compaction_start(lane_token.as_deref(), &ctx.req_id))
            .flatten();

        'routes: loop {
            let mut ctx = ctx.clone();
            if let Some(lane_token) = lane_token.as_deref() {
                ctx.session_id = Some(route.bind_lane(lane_token));
            }

            let mut translated = match translate_request(
                &body,
                TranslateOptions {
                    session_id: ctx.session_id.clone(),
                    read_rewrite_scope: read_rewrite_scope.clone(),
                    service_tier: resolved.service_tier.clone(),
                    model: resolved.model.clone(),
                    use_responses_lite,
                },
            ) {
                Ok(t) => t,
                Err(e) => {
                    return json_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        e.to_string(),
                    );
                }
            };

            let mut compaction_lease = None;
            if !server_compaction_enabled && let Some(session_id) = ctx.session_id.as_deref() {
                compaction::clear_compaction(session_id);
            }
            if server_compaction_enabled && compact_boundary {
                if let Some(lease) = compaction_start.as_ref().and_then(|permit| {
                    begin_compaction_with_permit(
                        ctx.session_id.as_deref(),
                        &translated.model,
                        permit,
                    )
                }) {
                    log_compaction_event(
                        "server_compaction_triggered",
                        &ctx,
                        translated.input.len(),
                        None,
                    );
                    if let Some(monitor) = ctx.monitor.as_ref() {
                        monitor.compaction_started(&ctx.req_id);
                    }
                    let mut compaction_ctx = ctx.clone();
                    compaction_ctx.monitor = None;
                    match request_compaction(
                        self.client.as_ref(),
                        &route,
                        &translated,
                        &compaction_ctx,
                        &mut buffered_retry_state,
                    )
                    .await
                    {
                        Ok(native_history) => {
                            if store_compaction(&lease, &translated.model, native_history) {
                                log_compaction_event(
                                    "server_compaction_completed",
                                    &ctx,
                                    translated.input.len(),
                                    None,
                                );
                                compaction_lease = Some(lease);
                            } else {
                                abort_compaction_attempt(Some(&lease));
                                log_compaction_event(
                                    "server_compaction_failed",
                                    &ctx,
                                    translated.input.len(),
                                    Some(
                                        "compaction state was superseded or exceeded the in-memory limit",
                                    ),
                                );
                            }
                        }
                        Err(error) => {
                            abort_compaction_attempt(Some(&lease));
                            let error_message = error.to_string();
                            log_compaction_event(
                                "server_compaction_failed",
                                &ctx,
                                translated.input.len(),
                                Some(&error_message),
                            );
                            if let CompactionError::Upstream(error) = error
                                && error.status == 401
                            {
                                if auth_rejection_budget.try_claim()
                                    && let Some(next_route) = self
                                        .client
                                        .refresh_conversation_route_after_rejection(&route)
                                        .await
                                {
                                    route = next_route;
                                    route_rebuilt = true;
                                    continue 'routes;
                                }
                                return map_codex_error_to_response(&error);
                            }
                        }
                    }
                }
            } else if server_compaction_enabled
                && let Some(replay) =
                    apply_compaction_replay(ctx.session_id.as_deref(), &translated, &ctx.req_id)
            {
                translated = replay.request;
                compaction_lease = Some(replay.lease);
            }

            // Check continuation with the logical request ordering reserved before
            // any credential-derived route is selected or rebuilt.
            let mut continuation = continuation_candidate_for_turn(
                ctx.session_id.as_deref(),
                &translated,
                previous_response_id_enabled,
                logical_turn_id,
            );
            if route_rebuilt {
                force_full_context_continuation(&mut continuation, "auth_rebind_full_context");
            }
            let turn_id = continuation.turn_id;

            // Post to upstream with continuation
            let client = self.client.clone();
            if !upstream_started {
                if let Some(monitor) = ctx.monitor.as_ref() {
                    monitor.upstream_started(&ctx.req_id);
                }
                upstream_started = true;
            }
            if want_stream && matches!(self.client.transport(), config::CodexTransport::WebSocket) {
                let stream_request = translated.clone();
                match live_stream_response(
                    client,
                    &route,
                    message_id.clone(),
                    model,
                    ctx.clone(),
                    stream_request,
                    continuation,
                    compaction_lease.clone(),
                    read_rewrite_scope.clone(),
                    auth_rejection_budget.clone(),
                    &mut live_start_attempt,
                )
                .await
                {
                    LiveStreamOutcome::Response(response) => return response,
                    LiveStreamOutcome::Unauthorized { error } => {
                        if auth_rejection_budget.try_claim()
                            && let Some(next_route) = self
                                .client
                                .refresh_conversation_route_after_rejection(&route)
                                .await
                        {
                            route = next_route;
                            route_rebuilt = true;
                            continue 'routes;
                        }
                        return map_codex_error_to_response(&error);
                    }
                }
            }

            let mut continuation = Some(continuation);
            let upstream = loop {
                let response = match client
                    .post_codex_bound_with_retry_state(
                        &route,
                        &translated,
                        &ctx,
                        continuation.as_ref(),
                        &mut buffered_retry_state,
                    )
                    .await
                {
                    Ok(r) => r,
                    Err(error) => {
                        abort_compaction_attempt(compaction_lease.as_ref());
                        abort_continuation(ctx.session_id.as_deref(), turn_id);
                        if error.status == 401
                            && auth_rejection_budget.try_claim()
                            && let Some(next_route) = self
                                .client
                                .refresh_conversation_route_after_rejection(&route)
                                .await
                        {
                            route = next_route;
                            route_rebuilt = true;
                            continue 'routes;
                        }
                        return map_codex_error_to_response(&error);
                    }
                };
                if !is_empty_codex_success_completion(&response.body, read_rewrite_scope.as_deref())
                {
                    break response;
                }
                // A successful terminal event with no output would translate into
                // an empty end_turn; retry with full context instead.
                let error = empty_buffered_completion_error();
                drop_live_continuation_for_retry(&mut continuation);
                if empty_completion_attempt >= MAX_EMPTY_COMPLETION_RETRIES {
                    abort_compaction_attempt(compaction_lease.as_ref());
                    abort_continuation(ctx.session_id.as_deref(), turn_id);
                    return map_codex_error_to_response(&error);
                }
                let delay = compute_backoff_delay(empty_completion_attempt, None);
                if delay.exceeds_budget {
                    abort_compaction_attempt(compaction_lease.as_ref());
                    abort_continuation(ctx.session_id.as_deref(), turn_id);
                    return map_codex_error_to_response(&error);
                }
                empty_completion_attempt += 1;
                sleep(delay.wait_ms).await;
            };

            return if want_stream {
                let estimated_input_tokens = count_translated_tokens(&translated);
                let sse_bytes = match translate_stream_bytes_with_traffic_in_scope(
                    &upstream.body,
                    &message_id,
                    model,
                    estimated_input_tokens,
                    ctx.traffic.as_deref(),
                    read_rewrite_scope.as_deref(),
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        abort_compaction_attempt(compaction_lease.as_ref());
                        abort_continuation(ctx.session_id.as_deref(), turn_id);
                        return map_codex_failure_to_response(&format!(
                            "Stream translation error: {e}"
                        ));
                    }
                };
                if let Some(monitor) = ctx.monitor.as_ref() {
                    let (input_tokens, output_tokens) = usage_from_anthropic_sse(&sse_bytes);
                    monitor.stream_progress(
                        &ctx.req_id,
                        sse_bytes.len() as u64,
                        count_sse_events(&sse_bytes),
                        input_tokens,
                        output_tokens,
                    );
                }
                update_continuation_from_upstream(
                    ctx.session_id.as_deref(),
                    read_rewrite_scope.as_deref(),
                    turn_id,
                    &translated,
                    &upstream.body,
                    upstream.socket_id,
                    compaction_lease.as_ref(),
                );

                let headers = [
                    (http::header::CONTENT_TYPE, "text/event-stream"),
                    (http::header::CACHE_CONTROL, "no-cache"),
                    (http::header::CONNECTION, "keep-alive"),
                ];
                (headers, sse_bytes).into_response()
            } else {
                match accumulate_response_with_traffic_in_scope(
                    &upstream.body,
                    &message_id,
                    model,
                    ctx.traffic.as_deref(),
                    read_rewrite_scope.as_deref(),
                ) {
                    Ok(json) => {
                        if let Some(monitor) = ctx.monitor.as_ref() {
                            monitor.usage_updated(
                                &ctx.req_id,
                                json.pointer("/usage/input_tokens").and_then(|v| v.as_u64()),
                                json.pointer("/usage/output_tokens")
                                    .and_then(|v| v.as_u64()),
                            );
                        }
                        update_continuation_from_upstream(
                            ctx.session_id.as_deref(),
                            read_rewrite_scope.as_deref(),
                            turn_id,
                            &translated,
                            &upstream.body,
                            upstream.socket_id,
                            compaction_lease.as_ref(),
                        );
                        (StatusCode::OK, Json(json)).into_response()
                    }
                    Err(e) => {
                        abort_compaction_attempt(compaction_lease.as_ref());
                        abort_continuation(ctx.session_id.as_deref(), turn_id);
                        map_codex_failure_to_response(&format!("Accumulation error: {e}"))
                    }
                }
            };
        }
    }

    async fn handle_messages_scoped(
        &self,
        body: MessagesRequest,
        mut ctx: RequestContext,
        scope: RequestScope,
    ) -> Response {
        ctx.session_id = scope.lane_token("codex-conversation");
        self.handle_messages(body, ctx).await
    }

    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let model = body.model.as_deref().unwrap_or("gpt-5.6-sol");
        let mut resolved =
            resolve_model_request_with_config_override(model, !body.bypass_provider_model_override);
        if let Err(e) = assert_allowed_model(&resolved.model) {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "Model \"{model}\" resolves to unsupported model \"{}\"",
                    e.model
                ),
            );
        }
        let use_responses_lite = apply_model_lane_for_request(&mut resolved.model, &body);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &resolved.model);
        }

        let translated = match translate_request(
            &body,
            TranslateOptions {
                session_id: None,
                read_rewrite_scope: None,
                service_tier: resolved.service_tier.clone(),
                model: resolved.model.clone(),
                use_responses_lite,
            },
        ) {
            Ok(t) => t,
            Err(e) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    e.to_string(),
                );
            }
        };

        let tokens = count_translated_tokens(&translated);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.usage_updated(&ctx.req_id, Some(tokens), None);
        }
        (
            StatusCode::OK,
            Json(CountTokensResponse {
                input_tokens: tokens,
            }),
        )
            .into_response()
    }
}

/// Picks the upstream model and lane for a request. Hosted web_search must
/// run on the full Responses API (the lite lane rejects hosted tools), and
/// lite-only models like gpt-5.6-luna don't exist there, so such requests
/// are upgraded to a full-lane model. Returns whether to use the lite lane.
fn apply_model_lane_for_request(model: &mut String, body: &MessagesRequest) -> bool {
    if has_hosted_web_search(body) {
        *model = full_lane_web_search_model(model).to_string();
        return false;
    }
    uses_responses_lite(model)
}

fn count_sse_events(bytes: &[u8]) -> u64 {
    String::from_utf8_lossy(bytes).matches("event:").count() as u64
}

fn log_compaction_event(
    event: &str,
    ctx: &RequestContext,
    input_items: usize,
    error: Option<&str>,
) {
    let mut fields = serde_json::Map::new();
    fields.insert("reqId".into(), serde_json::json!(ctx.req_id));
    fields.insert("inputItems".into(), serde_json::json!(input_items));
    if let Some(error) = error {
        fields.insert("error".into(), serde_json::json!(error));
        create_logger("codex").warn(event, Some(fields));
    } else {
        create_logger("codex").info(event, Some(fields));
    }
}

fn abort_request_state(
    session_id: Option<&str>,
    turn_id: Option<u64>,
    compaction_lease: Option<&CompactionLease>,
    _request: &translate::request::ResponsesRequest,
) {
    abort_compaction_attempt(compaction_lease);
    abort_continuation(session_id, turn_id);
}

enum LiveStreamStart {
    Response(Response),
    Retry { error: client::CodexError },
    Unauthorized { error: client::CodexError },
}

enum LiveStreamOutcome {
    Response(Response),
    Unauthorized { error: client::CodexError },
}

#[allow(clippy::too_many_arguments)]
async fn live_stream_response(
    client: Arc<CodexHttpClient>,
    route: &client::CodexConversationRoute,
    message_id: String,
    model: &str,
    ctx: RequestContext,
    request_body: translate::request::ResponsesRequest,
    continuation: ContinuationCandidate,
    compaction_lease: Option<CompactionLease>,
    read_rewrite_scope: Option<String>,
    auth_rejection_budget: Arc<AuthRejectionBudget>,
    attempt: &mut u32,
) -> LiveStreamOutcome {
    let model = model.to_string();
    let turn_id = continuation.turn_id;
    let mut continuation = Some(continuation);

    loop {
        let upstream_events = match client
            .stream_codex_websocket_events_bound(route, &request_body, &ctx, continuation.as_ref())
            .await
        {
            Ok(events) => events,
            Err(error) if error.status == 401 => {
                abort_request_state(
                    ctx.session_id.as_deref(),
                    turn_id,
                    compaction_lease.as_ref(),
                    &request_body,
                );
                return LiveStreamOutcome::Unauthorized { error };
            }
            Err(err) if retryable_live_start_codex_error(&err) => {
                let dropped = drop_live_continuation_for_retry(&mut continuation);
                if dropped && is_missing_previous_response_error(&err) {
                    *attempt += 1;
                    continue;
                }
                if *attempt >= MAX_RETRYABLE_LIVE_STREAM_RETRIES {
                    abort_request_state(
                        ctx.session_id.as_deref(),
                        turn_id,
                        compaction_lease.as_ref(),
                        &request_body,
                    );
                    return LiveStreamOutcome::Response(map_codex_error_to_response(&err));
                }
                let delay = compute_backoff_delay(*attempt, err.retry_after.as_deref());
                if delay.exceeds_budget {
                    abort_request_state(
                        ctx.session_id.as_deref(),
                        turn_id,
                        compaction_lease.as_ref(),
                        &request_body,
                    );
                    return LiveStreamOutcome::Response(map_codex_error_to_response(&err));
                }
                *attempt += 1;
                sleep(delay.wait_ms).await;
                continue;
            }
            Err(err) => {
                abort_request_state(
                    ctx.session_id.as_deref(),
                    turn_id,
                    compaction_lease.as_ref(),
                    &request_body,
                );
                return LiveStreamOutcome::Response(map_codex_error_to_response(&err));
            }
        };

        match live_stream_response_once(
            upstream_events,
            message_id.clone(),
            &model,
            ctx.clone(),
            turn_id,
            request_body.clone(),
            compaction_lease.clone(),
            read_rewrite_scope.clone(),
            client.clone(),
            route.clone(),
            auth_rejection_budget.clone(),
        )
        .await
        {
            LiveStreamStart::Response(response) => {
                return LiveStreamOutcome::Response(response);
            }
            LiveStreamStart::Unauthorized { error } => {
                return LiveStreamOutcome::Unauthorized { error };
            }
            LiveStreamStart::Retry { error } => {
                let dropped = drop_live_continuation_for_retry(&mut continuation);
                if dropped && is_missing_previous_response_error(&error) {
                    *attempt += 1;
                    continue;
                }
                if *attempt >= MAX_RETRYABLE_LIVE_STREAM_RETRIES {
                    abort_request_state(
                        ctx.session_id.as_deref(),
                        turn_id,
                        compaction_lease.as_ref(),
                        &request_body,
                    );
                    return LiveStreamOutcome::Response(map_codex_error_to_response(&error));
                }
                let delay = compute_backoff_delay(*attempt, error.retry_after.as_deref());
                if delay.exceeds_budget {
                    abort_request_state(
                        ctx.session_id.as_deref(),
                        turn_id,
                        compaction_lease.as_ref(),
                        &request_body,
                    );
                    return LiveStreamOutcome::Response(map_codex_error_to_response(&error));
                }
                *attempt += 1;
                sleep(delay.wait_ms).await;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn live_stream_response_once(
    mut upstream_events: websocket::CodexWebSocketEventReceiver,
    message_id: String,
    model: &str,
    ctx: RequestContext,
    turn_id: Option<u64>,
    request_body: translate::request::ResponsesRequest,
    compaction_lease: Option<CompactionLease>,
    read_rewrite_scope: Option<String>,
    client: Arc<CodexHttpClient>,
    route: client::CodexConversationRoute,
    auth_rejection_budget: Arc<AuthRejectionBudget>,
) -> LiveStreamStart {
    let estimated_input_tokens = count_translated_tokens(&request_body);
    let mut translator = LiveStreamTranslator::with_estimated_input_tokens(
        message_id,
        model.to_string(),
        estimated_input_tokens,
    )
    .with_read_rewrite_scope(read_rewrite_scope.clone());
    let mut upstream_sse_body = Vec::new();
    // Keep protocol framing private until real output makes a transparent retry unsafe.
    // Every branch that consumes pending_chunk returns, so it is never flushed twice.
    let mut pending_chunk = Vec::new();
    let mut generation_started = false;

    while let Some(item) = upstream_events.recv().await {
        let payload = match item {
            Ok(payload) => payload,
            Err(error) if error.status == 401 => {
                abort_request_state(
                    ctx.session_id.as_deref(),
                    turn_id,
                    compaction_lease.as_ref(),
                    &request_body,
                );
                return LiveStreamStart::Unauthorized { error };
            }
            Err(err) => {
                if retryable_live_start_codex_error(&err) {
                    return LiveStreamStart::Retry { error: err };
                }
                abort_request_state(
                    ctx.session_id.as_deref(),
                    turn_id,
                    compaction_lease.as_ref(),
                    &request_body,
                );
                return LiveStreamStart::Response(map_codex_error_to_response(&err));
            }
        };
        if websocket::event_error_status(&payload) == Some(401) {
            let failure = events::classify_event_failure(&payload);
            let error = client::CodexError {
                status: 401,
                message: failure
                    .as_ref()
                    .map(|failure| failure.message.clone())
                    .unwrap_or_else(|| "Unauthorized".to_string()),
                detail: Some(payload.to_string()),
                retry_after: failure.and_then(|failure| failure.retry_after),
                origin: client::CodexErrorOrigin::WebSocket,
            };
            abort_request_state(
                ctx.session_id.as_deref(),
                turn_id,
                compaction_lease.as_ref(),
                &request_body,
            );
            return LiveStreamStart::Unauthorized { error };
        }
        if !generation_started && codex_generation_event(&payload) {
            if let Some(monitor) = ctx.monitor.as_ref() {
                monitor.generation_started(&ctx.req_id);
            }
            generation_started = true;
        }
        append_upstream_sse_payload(&mut upstream_sse_body, &payload);
        let (chunk, terminal) = match translate_live_stream_payload(&mut translator, &payload, None)
        {
            Ok(result) => result,
            Err(message) => {
                if retryable_live_start_payload(&payload, &message) {
                    let lower_message = message.to_ascii_lowercase();
                    let status = websocket::event_error_status(&payload).unwrap_or_else(|| {
                        let error = payload.get("error").or_else(|| {
                            payload.get("response").and_then(|value| value.get("error"))
                        });
                        let overloaded = error.is_some_and(|error| {
                            error.get("code").and_then(|value| value.as_str())
                                == Some("overloaded_error")
                                || error.get("type").and_then(|value| value.as_str())
                                    == Some("overloaded_error")
                        });
                        if payload.get("type").and_then(|value| value.as_str())
                            == Some("codex.rate_limits")
                            || lower_message.contains("rate limit")
                        {
                            429
                        } else if overloaded || lower_message.contains("overloaded") {
                            529
                        } else {
                            503
                        }
                    });
                    return LiveStreamStart::Retry {
                        error: client::CodexError {
                            status,
                            message: message.clone(),
                            detail: Some(message),
                            retry_after: retry_after_from_live_payload(&payload),
                            origin: client::CodexErrorOrigin::WebSocket,
                        },
                    };
                }
                abort_request_state(
                    ctx.session_id.as_deref(),
                    turn_id,
                    compaction_lease.as_ref(),
                    &request_body,
                );
                return LiveStreamStart::Response(map_codex_failure_to_response(&message));
            }
        };
        pending_chunk.extend_from_slice(&chunk);
        if terminal
            && is_codex_success_terminal_event(&payload)
            && !translator.has_semantic_output()
        {
            return LiveStreamStart::Retry {
                error: empty_live_completion_error(),
            };
        }
        if translator.has_semantic_output() && !pending_chunk.is_empty() {
            record_live_stream_downstream_capture(&ctx, &pending_chunk);
            record_live_stream_progress(&ctx, &pending_chunk);
            if terminal {
                update_continuation_from_upstream(
                    ctx.session_id.as_deref(),
                    read_rewrite_scope.as_deref(),
                    turn_id,
                    &request_body,
                    &upstream_sse_body,
                    upstream_events.socket_id(),
                    compaction_lease.as_ref(),
                );
                return LiveStreamStart::Response(single_live_stream_response(pending_chunk));
            }
            return LiveStreamStart::Response(remaining_live_stream_response(
                upstream_events,
                translator,
                pending_chunk,
                ctx,
                turn_id,
                request_body,
                upstream_sse_body,
                compaction_lease,
                read_rewrite_scope,
                client,
                route,
                auth_rejection_budget,
            ));
        }
        if terminal {
            update_continuation_from_upstream(
                ctx.session_id.as_deref(),
                read_rewrite_scope.as_deref(),
                turn_id,
                &request_body,
                &upstream_sse_body,
                upstream_events.socket_id(),
                compaction_lease.as_ref(),
            );
            if pending_chunk.is_empty() {
                return LiveStreamStart::Response(empty_live_stream_response());
            }
            record_live_stream_downstream_capture(&ctx, &pending_chunk);
            record_live_stream_progress(&ctx, &pending_chunk);
            return LiveStreamStart::Response(single_live_stream_response(pending_chunk));
        }
    }

    LiveStreamStart::Retry {
        error: client::CodexError {
            status: 0,
            message: "WebSocket connection closed before terminal Codex response event".to_string(),
            detail: Some(websocket::WEBSOCKET_MISSING_TERMINAL_DETAIL.to_string()),
            retry_after: None,
            origin: client::CodexErrorOrigin::WebSocket,
        },
    }
}

fn empty_live_completion_error() -> client::CodexError {
    client::CodexError {
        status: 503,
        message: "Codex completed without producing output".to_string(),
        detail: Some(EMPTY_CODEX_COMPLETION_DETAIL.to_string()),
        retry_after: None,
        origin: client::CodexErrorOrigin::WebSocket,
    }
}

fn codex_generation_event(payload: &serde_json::Value) -> bool {
    !matches!(
        payload.get("type").and_then(|value| value.as_str()),
        Some("codex.rate_limits" | "keepalive") | None
    )
}

fn translate_live_stream_payload(
    translator: &mut LiveStreamTranslator,
    payload: &serde_json::Value,
    traffic: Option<&crate::traffic::TrafficCapture>,
) -> Result<(Vec<u8>, bool), String> {
    let chunk = translator.accept(payload, traffic)?;
    let terminal = is_codex_terminal_event(payload) || translator.is_finished();
    Ok((chunk, terminal))
}

fn record_live_stream_downstream_capture(ctx: &RequestContext, chunk: &[u8]) {
    let Some(traffic) = ctx.traffic.as_ref() else {
        return;
    };
    for event in parse_sse_events(chunk) {
        let Ok(data) = serde_json::from_str::<serde_json::Value>(&event.data) else {
            continue;
        };
        traffic.write_json_event(
            "050-downstream-event",
            &serde_json::json!({
                "event": event.event.as_deref().unwrap_or("message"),
                "data": data,
            }),
        );
    }
}

fn record_live_stream_progress(ctx: &RequestContext, chunk: &[u8]) {
    if let Some(monitor) = ctx.monitor.as_ref() {
        let (input_tokens, output_tokens) = usage_from_anthropic_sse(chunk);
        monitor.stream_progress(
            &ctx.req_id,
            chunk.len() as u64,
            count_sse_events(chunk),
            input_tokens,
            output_tokens,
        );
    }
}

fn single_live_stream_response(chunk: Vec<u8>) -> Response {
    event_stream_response(futures_util::stream::once(async move {
        Ok::<Bytes, std::io::Error>(Bytes::from(chunk))
    }))
}

fn empty_live_stream_response() -> Response {
    event_stream_response(futures_util::stream::empty::<Result<Bytes, std::io::Error>>())
}

#[allow(clippy::too_many_arguments)]
fn remaining_live_stream_response(
    mut upstream_events: websocket::CodexWebSocketEventReceiver,
    mut translator: LiveStreamTranslator,
    first_chunk: Vec<u8>,
    ctx: RequestContext,
    turn_id: Option<u64>,
    request_body: translate::request::ResponsesRequest,
    mut upstream_sse_body: Vec<u8>,
    compaction_lease: Option<CompactionLease>,
    read_rewrite_scope: Option<String>,
    client: Arc<CodexHttpClient>,
    route: client::CodexConversationRoute,
    auth_rejection_budget: Arc<AuthRejectionBudget>,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    tokio::spawn(async move {
        if tx.send(Ok(Bytes::from(first_chunk))).await.is_err() {
            abort_request_state(
                ctx.session_id.as_deref(),
                turn_id,
                compaction_lease.as_ref(),
                &request_body,
            );
            return;
        }
        while let Some(item) = upstream_events.recv().await {
            match item {
                Ok(payload) => {
                    if websocket::event_error_status(&payload) == Some(401) {
                        client.refresh_conversation_auth_after_rejection_in_background(
                            &route,
                            auth_rejection_budget.clone(),
                        );
                        abort_request_state(
                            ctx.session_id.as_deref(),
                            turn_id,
                            compaction_lease.as_ref(),
                            &request_body,
                        );
                        let chunk = translator.error_chunk(
                            "Authentication failed",
                            "authentication_error",
                            ctx.traffic.as_deref(),
                        );
                        if !chunk.is_empty() {
                            record_live_stream_progress(&ctx, &chunk);
                            let _ = tx.send(Ok(Bytes::from(chunk))).await;
                        }
                        return;
                    }
                    append_upstream_sse_payload(&mut upstream_sse_body, &payload);
                    let (chunk, terminal) = match translate_live_stream_payload(
                        &mut translator,
                        &payload,
                        ctx.traffic.as_deref(),
                    ) {
                        Ok(result) => result,
                        Err(message) => {
                            abort_request_state(
                                ctx.session_id.as_deref(),
                                turn_id,
                                compaction_lease.as_ref(),
                                &request_body,
                            );
                            let chunk = translator.error_chunk(
                                &message,
                                "api_error",
                                ctx.traffic.as_deref(),
                            );
                            if !chunk.is_empty() {
                                record_live_stream_progress(&ctx, &chunk);
                                let _ = tx.send(Ok(Bytes::from(chunk))).await;
                            }
                            return;
                        }
                    };
                    if !chunk.is_empty() {
                        record_live_stream_progress(&ctx, &chunk);
                        if tx.send(Ok(Bytes::from(chunk))).await.is_err() {
                            abort_request_state(
                                ctx.session_id.as_deref(),
                                turn_id,
                                compaction_lease.as_ref(),
                                &request_body,
                            );
                            return;
                        }
                    }
                    if terminal {
                        update_continuation_from_upstream(
                            ctx.session_id.as_deref(),
                            read_rewrite_scope.as_deref(),
                            turn_id,
                            &request_body,
                            &upstream_sse_body,
                            upstream_events.socket_id(),
                            compaction_lease.as_ref(),
                        );
                        return;
                    }
                }
                Err(err) => {
                    abort_request_state(
                        ctx.session_id.as_deref(),
                        turn_id,
                        compaction_lease.as_ref(),
                        &request_body,
                    );
                    if err.status == 401 {
                        client.refresh_conversation_auth_after_rejection_in_background(
                            &route,
                            auth_rejection_budget.clone(),
                        );
                        let chunk = translator.error_chunk(
                            "Authentication failed",
                            "authentication_error",
                            ctx.traffic.as_deref(),
                        );
                        if !chunk.is_empty() {
                            record_live_stream_progress(&ctx, &chunk);
                            let _ = tx.send(Ok(Bytes::from(chunk))).await;
                        }
                        return;
                    }
                    let chunk =
                        translator.finish_after_closed_completed_tool_call(ctx.traffic.as_deref());
                    if !chunk.is_empty() {
                        record_live_stream_progress(&ctx, &chunk);
                        let _ = tx.send(Ok(Bytes::from(chunk))).await;
                        return;
                    }
                    let chunk = translator.error_chunk(
                        codex_error_message(&err),
                        codex_stream_error_type(&err),
                        ctx.traffic.as_deref(),
                    );
                    if !chunk.is_empty() {
                        record_live_stream_progress(&ctx, &chunk);
                        let _ = tx.send(Ok(Bytes::from(chunk))).await;
                    }
                    return;
                }
            }
        }

        abort_request_state(
            ctx.session_id.as_deref(),
            turn_id,
            compaction_lease.as_ref(),
            &request_body,
        );
        let chunk = translator.finish_after_closed_completed_tool_call(ctx.traffic.as_deref());
        if !chunk.is_empty() {
            record_live_stream_progress(&ctx, &chunk);
            let _ = tx.send(Ok(Bytes::from(chunk))).await;
            return;
        }
        let chunk = translator.error_chunk(
            "WebSocket connection closed before terminal Codex response event",
            "api_error",
            ctx.traffic.as_deref(),
        );
        if !chunk.is_empty() {
            record_live_stream_progress(&ctx, &chunk);
            let _ = tx.send(Ok(Bytes::from(chunk))).await;
        }
    });

    let stream = futures_util::stream::unfold(rx, |mut rx| async {
        rx.recv().await.map(|item| (item, rx))
    });
    event_stream_response(stream)
}

fn append_upstream_sse_payload(buffer: &mut Vec<u8>, payload: &serde_json::Value) {
    let text = payload.to_string();
    for line in text.lines() {
        buffer.extend_from_slice(b"data: ");
        buffer.extend_from_slice(line.as_bytes());
        buffer.push(b'\n');
    }
    buffer.push(b'\n');
}

fn event_stream_response<S>(stream: S) -> Response
where
    S: futures_util::Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
{
    let headers = [
        (http::header::CONTENT_TYPE, "text/event-stream"),
        (http::header::CACHE_CONTROL, "no-cache"),
        (http::header::CONNECTION, "keep-alive"),
    ];
    (headers, Body::from_stream(stream)).into_response()
}

fn empty_buffered_completion_error() -> client::CodexError {
    client::CodexError {
        status: 503,
        message: "Codex completed without producing output".to_string(),
        detail: Some(EMPTY_CODEX_COMPLETION_DETAIL.to_string()),
        retry_after: None,
        origin: match config::codex_transport() {
            config::CodexTransport::Http => client::CodexErrorOrigin::BufferedHttp,
            _ => client::CodexErrorOrigin::BufferedWebSocket,
        },
    }
}

/// True when the buffered upstream body ended in a successful terminal event
/// without ever producing semantic output (text, thinking, tool, web search).
fn is_empty_codex_success_completion(
    upstream_sse: &[u8],
    read_rewrite_scope: Option<&str>,
) -> bool {
    use self::translate::reducer::{ReducerEvent, TERM_COMPLETED, TERM_DONE};

    let Ok(events) =
        self::translate::reducer::reduce_upstream_bytes_in_scope(upstream_sse, read_rewrite_scope)
    else {
        return false;
    };
    let mut saw_success_terminal = false;
    for event in &events {
        match event {
            ReducerEvent::TextDelta { text, .. } if !text.is_empty() => return false,
            ReducerEvent::ThinkingStart { .. }
            | ReducerEvent::ToolStart { .. }
            | ReducerEvent::WebSearch { .. } => return false,
            ReducerEvent::Finish { terminal_type, .. }
                if terminal_type == TERM_COMPLETED || terminal_type == TERM_DONE =>
            {
                saw_success_terminal = true;
            }
            _ => {}
        }
    }
    saw_success_terminal
}

fn is_codex_terminal_event(payload: &serde_json::Value) -> bool {
    matches!(
        payload.get("type").and_then(|v| v.as_str()),
        Some("response.completed")
            | Some("response.incomplete")
            | Some("response.done")
            | Some("response.failed")
            | Some("response.error")
            | Some("error")
    )
}

fn is_codex_success_terminal_event(payload: &serde_json::Value) -> bool {
    matches!(
        payload.get("type").and_then(|v| v.as_str()),
        Some("response.completed") | Some("response.done")
    )
}

fn retryable_live_start_codex_error(err: &client::CodexError) -> bool {
    if err.origin == client::CodexErrorOrigin::WebSocketHandshake {
        if err.detail.as_deref() == Some(websocket::WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL) {
            return false;
        }
        return err.status == 0 || matches!(err.status, 429 | 500 | 502 | 503 | 504 | 529);
    }
    matches!(err.status, 429 | 500 | 502 | 503 | 504 | 529)
        || (err.status == 0 && retryable_live_message(codex_error_message(err)))
}

fn is_missing_previous_response_error(err: &client::CodexError) -> bool {
    err.detail.as_deref() == Some("previous_response_not_found")
}

fn drop_live_continuation_for_retry(continuation: &mut Option<ContinuationCandidate>) -> bool {
    if continuation
        .as_ref()
        .and_then(|candidate| candidate.previous_response_id.as_deref())
        .is_none()
    {
        return false;
    }

    if let Some(candidate) = continuation.as_mut() {
        force_full_context_continuation(candidate, "full_context_retry");
    }
    true
}

fn force_full_context_continuation(candidate: &mut ContinuationCandidate, reason: &str) {
    candidate.previous_response_id = None;
    candidate.socket_id = None;
    candidate.input_delta = None;
    candidate.disabled_reason = Some(reason.to_string());
}

fn retryable_live_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    [
        "overloaded",
        "rate limit",
        "you can retry your request",
        "temporarily unavailable",
        "timed out",
        "connection closed",
        "connection reset",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn retryable_live_start_payload(payload: &serde_json::Value, _message: &str) -> bool {
    events::classify_event_failure(payload).is_some_and(|failure| failure.retryable())
}

fn retry_after_from_live_payload(payload: &serde_json::Value) -> Option<String> {
    events::classify_event_failure(payload).and_then(|failure| failure.retry_after)
}

fn codex_stream_error_type(err: &client::CodexError) -> &'static str {
    match err.status {
        429 => "rate_limit_error",
        529 => "overloaded_error",
        _ if codex_error_message(err)
            .to_lowercase()
            .contains("overloaded") =>
        {
            "overloaded_error"
        }
        _ => "api_error",
    }
}

fn update_continuation_from_upstream(
    session_id: Option<&str>,
    read_rewrite_scope: Option<&str>,
    turn_id: Option<u64>,
    request_body: &translate::request::ResponsesRequest,
    upstream_body: &[u8],
    socket_id: Option<u64>,
    compaction_lease: Option<&CompactionLease>,
) {
    match finish_metadata_from_upstream_in_scope(upstream_body, read_rewrite_scope) {
        Ok(Some(finish)) if finish.continuation_eligible => {
            activate_compaction(compaction_lease, &request_body.model, &finish.output_items);
            record_continuation(
                session_id,
                turn_id,
                request_body,
                finish.response_id.as_deref(),
                socket_id,
                &finish.output_items,
            );
        }
        _ => {
            abort_compaction_attempt(compaction_lease);
            abort_continuation(session_id, turn_id);
        }
    }
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

fn map_codex_error_to_response(err: &client::CodexError) -> Response {
    let message = codex_error_message(err);
    if is_context_window_overflow(message) {
        return map_codex_failure_to_response(message);
    }
    if err.detail.as_deref() == Some(EMPTY_CODEX_COMPLETION_DETAIL) {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "api_error", &err.message);
    }

    match err.status {
        401 => json_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            err.detail.as_deref().unwrap_or("Authentication failed"),
        ),
        403 => json_error(
            StatusCode::FORBIDDEN,
            "permission_error",
            err.detail.as_deref().unwrap_or("Permission denied"),
        ),
        429 => {
            let response = json_error(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                &err.message,
            );
            if let Some(retry_after) = err.retry_after.as_deref() {
                ([(http::header::RETRY_AFTER, retry_after)], response).into_response()
            } else {
                response
            }
        }
        status @ (400..=599) => {
            let response = json_error(
                StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
                if status == 529 {
                    "overloaded_error"
                } else {
                    "api_error"
                },
                codex_error_message(err),
            );
            if let Some(retry_after) = err.retry_after.as_deref() {
                ([(http::header::RETRY_AFTER, retry_after)], response).into_response()
            } else {
                response
            }
        }
        _ => json_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            codex_error_message(err),
        ),
    }
}

fn map_codex_failure_to_response(message: &str) -> Response {
    if is_context_window_overflow(message) {
        json_error(StatusCode::PAYLOAD_TOO_LARGE, "request_too_large", message)
    } else {
        json_error(StatusCode::BAD_GATEWAY, "api_error", message)
    }
}

fn is_context_window_overflow(message: &str) -> bool {
    message.to_ascii_lowercase().contains("context window")
}

fn codex_error_message(err: &client::CodexError) -> &str {
    err.detail.as_deref().unwrap_or({
        if err.status == 0 {
            err.message.as_str()
        } else {
            "Upstream error"
        }
    })
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

pub(crate) struct CodexCli;

impl CliHandlers for CodexCli {
    fn login(&self) -> Result<(), anyhow::Error> {
        let tokens = run_browser_login()?;
        let store = file_store();
        let manager = CodexAuthManager::new(store);
        let saved = manager.persist_initial_tokens(&tokens)?;
        print!(
            "{}",
            format_auth_saved_output(&manager.store.auth_path(), saved.account_id.as_deref())
        );
        Ok(())
    }

    fn device(&self) -> Result<(), anyhow::Error> {
        let tokens = DeviceAuthClient::new().run()?;
        let store = file_store();
        let manager = CodexAuthManager::new(store);
        let saved = manager.persist_initial_tokens(&tokens)?;
        print!(
            "{}",
            format_auth_saved_output(&manager.store.auth_path(), saved.account_id.as_deref())
        );
        Ok(())
    }

    fn status(&self) -> Result<(), anyhow::Error> {
        let store = file_store();
        let stored = store.load_auth()?;
        match stored {
            Some(auth) => {
                println!(
                    "Account: {}",
                    auth.account_id.as_deref().unwrap_or("(none)")
                );
                println!("{}", format_expiry(auth.expires, now_ms()));
                println!("Storage: {}", store.auth_path());
                Ok(())
            }
            None => {
                anyhow::bail!("Not authenticated");
            }
        }
    }

    fn logout(&self) -> Result<(), anyhow::Error> {
        let store = file_store();
        store.clear_auth()?;
        println!("Logged out");
        Ok(())
    }
}

pub(crate) static CODEX_CLI: CodexCli = CodexCli;

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn format_expiry(expires: u64, now: u64) -> String {
    let remaining = (i128::from(expires) - i128::from(now)).div_euclid(1000);
    let iso = time::OffsetDateTime::from_unix_timestamp_nanos(i128::from(expires) * 1_000_000)
        .ok()
        .and_then(|dt| {
            let fmt = time::format_description::parse_borrowed::<2>(
                "[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:3]Z",
            )
            .ok()?;
            dt.format(&fmt).ok()
        })
        .unwrap_or_else(|| "invalid".to_string());
    format!("Expires: {iso} (in {remaining}s)")
}

fn format_auth_saved_output(auth_path: &str, account_id: Option<&str>) -> String {
    let mut out = format!("Auth saved in {auth_path}\n");
    if let Some(account_id) = account_id {
        out.push_str(&format!("Account: {account_id}\n"));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::result_large_err)]
mod tests {
    use super::*;
    use crate::monitor::{EndpointKind, MonitorHandle};
    use crate::providers::codex::auth::token_store::StoredAuth;
    use futures_util::{SinkExt, StreamExt};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };
    use tokio_tungstenite::tungstenite::Message;

    fn upstream_sse(events: &[serde_json::Value]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for event in events {
            bytes.extend_from_slice(format!("data: {event}\n\n").as_bytes());
        }
        bytes
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
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
                return request;
            }
        }
    }

    fn http_request_parts(request: &[u8]) -> (String, serde_json::Value) {
        let header_end = request
            .windows(4)
            .position(|part| part == b"\r\n\r\n")
            .unwrap();
        (
            String::from_utf8_lossy(&request[..header_end]).into_owned(),
            serde_json::from_slice(&request[header_end + 4..]).unwrap(),
        )
    }

    async fn next_websocket_request(
        websocket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    ) -> serde_json::Value {
        loop {
            let message = websocket.next().await.unwrap().unwrap();
            if let Message::Text(text) = message {
                let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                if value.get("type").and_then(serde_json::Value::as_str) == Some("response.create")
                {
                    return value;
                }
            }
        }
    }

    async fn send_websocket_events(
        websocket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        events: impl IntoIterator<Item = serde_json::Value>,
    ) {
        for event in events {
            websocket
                .send(Message::Text(event.to_string()))
                .await
                .unwrap();
        }
    }

    fn standalone_search_request() -> MessagesRequest {
        serde_json::from_value(serde_json::json!({
            "model": "gpt-5.4",
            "max_tokens": 1024,
            "stream": false,
            "messages": [{
                "role": "user",
                "content": "Perform a web search for the query: route binding"
            }],
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search"
            }],
            "tool_choice": {"type": "tool", "name": "web_search"}
        }))
        .unwrap()
    }

    fn search_context(req_id: &str, session_id: Option<&str>) -> RequestContext {
        RequestContext {
            req_id: req_id.to_string(),
            session_id: session_id.map(str::to_string),
            session_seq: None,
            provider: "codex".to_string(),
            traffic: None,
            monitor: None,
        }
    }

    #[tokio::test]
    async fn standalone_search_route_binds_stateful_identity_and_keeps_stateless_random() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for _ in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = read_http_request(&mut socket).await;
                let header_end = request
                    .windows(4)
                    .position(|part| part == b"\r\n\r\n")
                    .unwrap();
                let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
                let body: serde_json::Value =
                    serde_json::from_slice(&request[header_end + 4..]).unwrap();
                captured.push((headers, body));
                let response = br#"{"output":"search answer","results":[]}"#;
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    response.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(response).await.unwrap();
            }
            captured
        });
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        );
        client.auth_manager().set_test_auth(StoredAuth {
            access: "search-token".into(),
            refresh: String::new(),
            account_id: Some("search-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client);

        for (req_id, lane) in [
            ("stateful-search", Some("raw-search-lane")),
            ("stateless-search", None),
        ] {
            let response = provider
                .handle_messages(standalone_search_request(), search_context(req_id, lane))
                .await;
            assert_eq!(response.status(), StatusCode::OK);
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
        }

        let captured = server.await.unwrap();
        let (stateful_headers, stateful_body) = &captured[0];
        assert!(stateful_headers.starts_with("POST /v1/alpha/search HTTP/1.1"));
        let bound_id = stateful_body["id"].as_str().unwrap();
        assert_ne!(bound_id, "raw-search-lane");
        assert!(!stateful_headers.contains("raw-search-lane"));
        assert!(stateful_headers.contains(&format!("session_id: {bound_id}")));
        assert!(stateful_headers.contains(&format!("x-codex-window-id: {bound_id}:0")));
        assert!(stateful_headers.contains("x-client-request-id: stateful-search"));

        let (stateless_headers, stateless_body) = &captured[1];
        assert!(
            stateless_body["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("search-"))
        );
        assert!(!stateless_headers.contains("\nsession_id: "));
        assert!(!stateless_headers.contains("x-codex-window-id:"));
        assert!(!stateless_headers.contains("x-client-request-id:"));
    }

    #[tokio::test]
    async fn standalone_search_401_rebuilds_route_and_preserves_statelessness() {
        for (case, lane) in [("stateful", Some("search-raw-lane")), ("stateless", None)] {
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
                access: format!("search-{case}-a"),
                refresh: "refresh-a".into(),
                account_id: Some("search-account".into()),
                expires: u64::MAX,
            });
            let provider = CodexProvider::with_client(client);
            let server_client = provider.client.clone();
            let server = tokio::spawn(async move {
                let mut captured = Vec::new();
                for attempt in 0..2 {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    captured.push(http_request_parts(&read_http_request(&mut socket).await));
                    if attempt == 0 {
                        server_client.auth_manager().set_test_auth(StoredAuth {
                            access: format!("search-{case}-b"),
                            refresh: "refresh-b".into(),
                            account_id: Some("search-account".into()),
                            expires: u64::MAX,
                        });
                        socket
                            .write_all(
                                b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 5\r\nconnection: close\r\n\r\nstale",
                            )
                            .await
                            .unwrap();
                    } else {
                        let body = br#"{"output":"search answer","results":[]}"#;
                        let head = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            body.len()
                        );
                        socket.write_all(head.as_bytes()).await.unwrap();
                        socket.write_all(body).await.unwrap();
                    }
                }
                assert!(
                    tokio::time::timeout(std::time::Duration::from_millis(75), listener.accept())
                        .await
                        .is_err()
                );
                captured
            });

            let response = provider
                .handle_messages(
                    standalone_search_request(),
                    search_context(&format!("search-{case}"), lane),
                )
                .await;
            assert_eq!(response.status(), StatusCode::OK, "{case}");
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();

            assert_eq!(provider.client.route_rejection_refresh_count(), 1, "{case}");
            let captured = server.await.unwrap();
            assert_eq!(captured.len(), 2, "{case}");
            assert!(
                captured[0]
                    .0
                    .contains(&format!("authorization: Bearer search-{case}-a"))
            );
            assert!(
                captured[1]
                    .0
                    .contains(&format!("authorization: Bearer search-{case}-b"))
            );
            let mut body_a = captured[0].1.clone();
            let mut body_b = captured[1].1.clone();
            let id_a = body_a.as_object_mut().unwrap().remove("id").unwrap();
            let id_b = body_b.as_object_mut().unwrap().remove("id").unwrap();
            assert_eq!(body_a, body_b, "{case}");
            if lane.is_some() {
                assert_ne!(id_a, id_b);
                for ((headers, _), id) in captured.iter().zip([id_a, id_b]) {
                    let id = id.as_str().unwrap();
                    assert!(headers.contains(&format!("session_id: {id}")));
                    assert!(headers.contains(&format!("x-codex-window-id: {id}:0")));
                    assert!(headers.contains(&format!("x-client-request-id: search-{case}")));
                }
            } else {
                assert_eq!(id_a, id_b, "stateless retry changed the request id");
                for (headers, _) in &captured {
                    assert!(!headers.contains("\nsession_id: "));
                    assert!(!headers.contains("x-codex-window-id:"));
                    assert!(!headers.contains("x-client-request-id:"));
                }
            }
        }
    }

    #[tokio::test]
    async fn standalone_search_second_401_returns_route_b_without_third_send() {
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
            access: "search-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("search-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client);
        let server_client = provider.client.clone();
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(http_request_parts(&read_http_request(&mut socket).await));
                server_client.auth_manager().set_test_auth(StoredAuth {
                    access: if attempt == 0 { "search-b" } else { "search-c" }.into(),
                    refresh: format!("refresh-{}", attempt + 2),
                    account_id: Some("search-account".into()),
                    expires: u64::MAX,
                });
                let body = if attempt == 0 {
                    br#"{"error":{"message":"search route a"}}"#.as_slice()
                } else {
                    br#"{"error":{"message":"search route b"}}"#.as_slice()
                };
                let head = format!(
                    "HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(body).await.unwrap();
            }
            (
                tokio::time::timeout(std::time::Duration::from_millis(100), listener.accept())
                    .await,
                captured,
            )
        });

        let response = provider
            .handle_messages(
                standalone_search_request(),
                search_context("search-second-401", Some("search-second-lane")),
            )
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("search route b"));
        assert_eq!(provider.client.route_rejection_refresh_count(), 1);

        let (third, captured) = server.await.unwrap();
        assert!(third.is_err(), "search sent route C");
        assert_eq!(captured.len(), 2);
        assert!(captured[0].0.contains("authorization: Bearer search-a"));
        assert!(captured[1].0.contains("authorization: Bearer search-b"));
        let mut body_a = captured[0].1.clone();
        let mut body_b = captured[1].1.clone();
        let id_a = body_a.as_object_mut().unwrap().remove("id").unwrap();
        let id_b = body_b.as_object_mut().unwrap().remove("id").unwrap();
        assert_eq!(body_a, body_b);
        assert_ne!(id_a, id_b);
    }

    #[tokio::test]
    async fn standalone_search_changed_or_unknown_account_does_not_replay() {
        for (case, rejected_account, refreshed_account) in [
            (
                "changed",
                Some("search-account-a"),
                Some("search-account-b"),
            ),
            ("unknown", Some("search-account-a"), None),
        ] {
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
                access: format!("search-{case}-a"),
                refresh: "refresh-a".into(),
                account_id: rejected_account.map(str::to_string),
                expires: u64::MAX,
            });
            let provider = CodexProvider::with_client(client);
            let server_client = provider.client.clone();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let request = http_request_parts(&read_http_request(&mut socket).await);
                server_client.auth_manager().set_test_auth(StoredAuth {
                    access: format!("search-{case}-b"),
                    refresh: "refresh-b".into(),
                    account_id: refreshed_account.map(str::to_string),
                    expires: u64::MAX,
                });
                let body = format!(r#"{{"error":{{"message":"original search {case}"}}}}"#);
                let head = format!(
                    "HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(body.as_bytes()).await.unwrap();
                (
                    tokio::time::timeout(std::time::Duration::from_millis(100), listener.accept())
                        .await,
                    request,
                )
            });

            let response = provider
                .handle_messages(
                    standalone_search_request(),
                    search_context(&format!("search-{case}"), Some("search-account-lane")),
                )
                .await;
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{case}");
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert!(
                String::from_utf8_lossy(&body).contains(&format!("original search {case}")),
                "{case}"
            );
            assert_eq!(provider.client.route_rejection_refresh_count(), 1, "{case}");
            assert_eq!(
                provider
                    .client
                    .auth_manager()
                    .get_auth()
                    .await
                    .unwrap()
                    .access,
                format!("search-{case}-b"),
                "refreshed auth was not left for the next request"
            );
            let (second, request) = server.await.unwrap();
            assert!(second.is_err(), "{case} account replayed the request");
            assert!(
                request
                    .0
                    .contains(&format!("authorization: Bearer search-{case}-a"))
            );
        }
    }

    #[tokio::test]
    async fn buffered_messages_401_rebuilds_full_bound_request_once() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_transport(config::CodexTransport::Http);
        client.auth_manager().set_test_auth(StoredAuth {
            access: "messages-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("messages-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client);
        let server_client = provider.client.clone();
        let success = upstream_sse(&[
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_b"}
            }),
            serde_json::json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "route-b-only"
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message"}
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_b",
                    "status": "completed",
                    "usage": {"input_tokens": 3, "output_tokens": 1}
                }
            }),
        ]);
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(http_request_parts(&read_http_request(&mut socket).await));
                if attempt == 0 {
                    server_client.auth_manager().set_test_auth(StoredAuth {
                        access: "messages-b".into(),
                        refresh: "refresh-b".into(),
                        account_id: Some("messages-account".into()),
                        expires: u64::MAX,
                    });
                    socket
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 7\r\nconnection: close\r\n\r\nroute-a",
                        )
                        .await
                        .unwrap();
                } else {
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        success.len()
                    );
                    socket.write_all(head.as_bytes()).await.unwrap();
                    socket.write_all(&success).await.unwrap();
                }
            }
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(75), listener.accept())
                    .await
                    .is_err()
            );
            captured
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-5.4",
            "max_tokens": 256,
            "stream": false,
            "messages": [{"role":"user", "content":"original full context"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                request,
                search_context("messages-rebind", Some("messages-raw-lane")),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let downstream: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(downstream["content"][0]["text"], "route-b-only");

        let captured = server.await.unwrap();
        assert_eq!(captured.len(), 2);
        assert!(captured[0].0.contains("authorization: Bearer messages-a"));
        assert!(captured[1].0.contains("authorization: Bearer messages-b"));
        let key_a = captured[0].1["prompt_cache_key"].as_str().unwrap();
        let key_b = captured[1].1["prompt_cache_key"].as_str().unwrap();
        assert_ne!(key_a, key_b);
        assert!(captured[0].0.contains(&format!("session_id: {key_a}")));
        assert!(
            captured[0]
                .0
                .contains(&format!("x-codex-window-id: {key_a}:0"))
        );
        assert!(captured[1].0.contains(&format!("session_id: {key_b}")));
        assert!(
            captured[1]
                .0
                .contains(&format!("x-codex-window-id: {key_b}:0"))
        );
        assert_eq!(captured[0].1["input"], captured[1].1["input"]);
        assert!(captured[1].1.get("previous_response_id").is_none());
    }

    #[tokio::test]
    async fn buffered_messages_second_401_returns_route_b_without_third_send() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_transport(config::CodexTransport::Http);
        client.auth_manager().set_test_auth(StoredAuth {
            access: "messages-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("messages-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client);
        let server_client = provider.client.clone();
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(http_request_parts(&read_http_request(&mut socket).await));
                server_client.auth_manager().set_test_auth(StoredAuth {
                    access: if attempt == 0 {
                        "messages-b"
                    } else {
                        "messages-c"
                    }
                    .into(),
                    refresh: format!("refresh-{}", attempt + 2),
                    account_id: Some("messages-account".into()),
                    expires: u64::MAX,
                });
                let body = if attempt == 0 {
                    b"messages route a"
                } else {
                    b"messages route b"
                };
                let head = format!(
                    "HTTP/1.1 401 Unauthorized\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(body).await.unwrap();
            }
            (
                tokio::time::timeout(std::time::Duration::from_millis(100), listener.accept())
                    .await,
                captured,
            )
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":false,
            "messages":[{"role":"user","content":"unchanged full context"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                request,
                search_context("messages-second-401", Some("messages-second-lane")),
            )
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("messages route b"));
        assert_eq!(provider.client.route_rejection_refresh_count(), 1);

        let (third, captured) = server.await.unwrap();
        assert!(third.is_err(), "messages sent route C");
        assert_eq!(captured.len(), 2);
        assert!(captured[0].0.contains("authorization: Bearer messages-a"));
        assert!(captured[1].0.contains("authorization: Bearer messages-b"));
        assert_ne!(
            captured[0].1["prompt_cache_key"],
            captured[1].1["prompt_cache_key"]
        );
        assert_eq!(captured[0].1["input"], captured[1].1["input"]);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn compaction_401_rebinds_and_shares_transport_retry_budget() {
        let _compaction_guard = compaction::lock_compaction_registry_for_tests();
        compaction::clear_all_compactions_for_tests();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_transport(config::CodexTransport::Http);
        client.auth_manager().set_test_auth(StoredAuth {
            access: "compact-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("compact-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client).with_server_compaction_for_test();
        let server_client = provider.client.clone();
        let compacted = upstream_sse(&[
            serde_json::json!({
                "type":"response.output_item.done",
                "item":{"type":"compaction","encrypted_content":"encrypted-route-b"}
            }),
            serde_json::json!({
                "type":"response.completed",
                "response":{"id":"compact-b","status":"completed"}
            }),
        ]);
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..6 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(http_request_parts(&read_http_request(&mut socket).await));
                match attempt {
                    0..=2 => {
                        socket
                            .write_all(
                                b"HTTP/1.1 503 Service Unavailable\r\nretry-after: 0\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                            )
                            .await
                            .unwrap();
                    }
                    3 => {
                        server_client.auth_manager().set_test_auth(StoredAuth {
                            access: "compact-b".into(),
                            refresh: "refresh-b".into(),
                            account_id: Some("compact-account".into()),
                            expires: u64::MAX,
                        });
                        socket
                            .write_all(
                                b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 15\r\nconnection: close\r\n\r\ncompact route a",
                            )
                            .await
                            .unwrap();
                    }
                    4 => {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            compacted.len()
                        );
                        socket.write_all(head.as_bytes()).await.unwrap();
                        socket.write_all(&compacted).await.unwrap();
                    }
                    5 => {
                        let body = br#"{"error":{"message":"main exhausted"}}"#;
                        let head = format!(
                            "HTTP/1.1 503 Service Unavailable\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            body.len()
                        );
                        socket.write_all(head.as_bytes()).await.unwrap();
                        socket.write_all(body).await.unwrap();
                    }
                    _ => unreachable!(),
                }
            }
            (
                tokio::time::timeout(std::time::Duration::from_millis(100), listener.accept())
                    .await,
                captured,
            )
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":false,
            "system":"You are a helpful AI assistant tasked with summarizing conversations",
            "messages":[{"role":"user","content":"summarize this compact boundary"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                request,
                search_context("compact-rebind", Some("compact-raw-lane")),
            )
            .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("main exhausted"));
        assert_eq!(provider.client.route_rejection_refresh_count(), 1);

        let (seventh, captured) = server.await.unwrap();
        assert!(
            seventh.is_err(),
            "the primary send received a fresh retry budget after auth rebind"
        );
        assert_eq!(captured.len(), 6);
        for (attempt, (headers, body)) in captured.iter().enumerate() {
            let has_trigger = body["input"].as_array().is_some_and(|input| {
                input.iter().any(|item| {
                    item.get("type").and_then(serde_json::Value::as_str)
                        == Some("compaction_trigger")
                })
            });
            if attempt <= 4 {
                assert!(has_trigger, "attempt {attempt} was not remote compaction");
            } else {
                assert!(
                    !has_trigger,
                    "the primary request retained the compaction trigger"
                );
            }
            let expected_auth = if attempt <= 3 {
                "compact-a"
            } else {
                "compact-b"
            };
            assert!(
                headers.contains(&format!("authorization: Bearer {expected_auth}")),
                "attempt {attempt} used the wrong auth"
            );
        }
        let key_a = &captured[0].1["prompt_cache_key"];
        assert!(
            captured[..4]
                .iter()
                .all(|(_, body)| &body["prompt_cache_key"] == key_a)
        );
        let key_b = &captured[4].1["prompt_cache_key"];
        assert_ne!(key_a, key_b);
        assert_eq!(&captured[5].1["prompt_cache_key"], key_b);
        compaction::clear_all_compactions_for_tests();
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn buffered_websocket_401_invalidates_a_and_rebuilds_b_full_context() {
        let _websocket_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let _continuation_guard = continuation::lock_continuation_registry_for_tests();
        continuation::clear_all_continuations_for_tests();
        websocket::clear_codex_websocket_pool_for_tests();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_transport(config::CodexTransport::WebSocket)
        .with_test_previous_response_id(true);
        client.auth_manager().set_test_auth(StoredAuth {
            access: "pooled-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("pooled-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client);
        let server_client = provider.client.clone();
        let server = tokio::spawn(async move {
            let (first, _) = listener.accept().await.unwrap();
            let mut route_a = tokio_tungstenite::accept_async(first).await.unwrap();
            let seed_request = next_websocket_request(&mut route_a).await;
            assert!(seed_request.get("previous_response_id").is_none());
            send_websocket_events(
                &mut route_a,
                [
                    serde_json::json!({"type":"response.created","response":{"id":"resp_seed"}}),
                    serde_json::json!({
                        "type":"response.output_item.added",
                        "output_index":0,
                        "item":{"type":"message","id":"msg_seed","role":"assistant","content":[]}
                    }),
                    serde_json::json!({
                        "type":"response.output_text.delta",
                        "output_index":0,
                        "delta":"seed-answer"
                    }),
                    serde_json::json!({
                        "type":"response.output_item.done",
                        "output_index":0,
                        "item":{
                            "type":"message",
                            "id":"msg_seed",
                            "role":"assistant",
                            "content":[{"type":"output_text","text":"seed-answer"}]
                        }
                    }),
                    serde_json::json!({
                        "type":"response.completed",
                        "response":{
                            "id":"resp_seed",
                            "status":"completed",
                            "usage":{"input_tokens":2,"output_tokens":1}
                        }
                    }),
                ],
            )
            .await;

            let request_a = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                next_websocket_request(&mut route_a),
            )
            .await
            .expect("route A did not reuse its exact origin socket");
            assert_eq!(request_a["previous_response_id"], "resp_seed");
            assert!(
                request_a["input"]
                    .as_array()
                    .is_some_and(|input| !input.is_empty())
            );
            send_websocket_events(
                &mut route_a,
                [serde_json::json!({
                    "type":"response.created",
                    "response":{"id":"resp_route_a"}
                })],
            )
            .await;
            server_client.auth_manager().set_test_auth(StoredAuth {
                access: "pooled-b".into(),
                refresh: "refresh-b".into(),
                account_id: Some("pooled-account".into()),
                expires: u64::MAX,
            });
            send_websocket_events(
                &mut route_a,
                [serde_json::json!({
                    "type":"response.failed",
                    "status_code":401,
                    "response":{"error":{"status":401,"message":"pooled route A rejected"}}
                })],
            )
            .await;
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                loop {
                    match route_a.next().await {
                        None | Some(Err(_)) | Some(Ok(Message::Close(_))) => break,
                        Some(Ok(_)) => {}
                    }
                }
            })
            .await
            .expect("route A socket stayed live after invalidation");

            let (second, _) = listener.accept().await.unwrap();
            let second_headers = Arc::new(std::sync::Mutex::new(None));
            let callback_headers = second_headers.clone();
            let mut route_b = tokio_tungstenite::accept_hdr_async(
                second,
                move |request: &http::Request<()>, response| {
                    *callback_headers.lock().unwrap() = Some(request.headers().clone());
                    Ok(response)
                },
            )
            .await
            .unwrap();
            assert_eq!(
                second_headers.lock().unwrap().as_ref().unwrap()[http::header::AUTHORIZATION],
                "Bearer pooled-b"
            );
            let request_b = next_websocket_request(&mut route_b).await;
            assert!(request_b.get("previous_response_id").is_none());
            assert_ne!(request_a["prompt_cache_key"], request_b["prompt_cache_key"]);
            let route_a_key = request_a["prompt_cache_key"].as_str().unwrap().to_string();
            let route_b_key = request_b["prompt_cache_key"].as_str().unwrap().to_string();
            assert_eq!(
                request_b["input"],
                serde_json::json!([
                    {
                        "type":"message",
                        "role":"user",
                        "content":[{"type":"input_text","text":"seed"}]
                    },
                    {
                        "type":"message",
                        "role":"assistant",
                        "content":[{"type":"output_text","text":"seed-answer"}]
                    },
                    {
                        "type":"message",
                        "role":"user",
                        "content":[{"type":"input_text","text":"follow up"}]
                    }
                ]),
                "route B did not restore the exact full translated context"
            );
            send_websocket_events(
                &mut route_b,
                [
                    serde_json::json!({"type":"response.created","response":{"id":"resp_b"}}),
                    serde_json::json!({
                        "type":"response.output_item.added",
                        "output_index":0,
                        "item":{"type":"message","id":"msg_b","role":"assistant","content":[]}
                    }),
                    serde_json::json!({
                        "type":"response.output_text.delta",
                        "output_index":0,
                        "delta":"pooled-route-b-only"
                    }),
                    serde_json::json!({
                        "type":"response.output_item.done",
                        "output_index":0,
                        "item":{
                            "type":"message",
                            "id":"msg_b",
                            "role":"assistant",
                            "content":[{"type":"output_text","text":"pooled-route-b-only"}]
                        }
                    }),
                    serde_json::json!({
                        "type":"response.completed",
                        "response":{
                            "id":"resp_b",
                            "status":"completed",
                            "usage":{"input_tokens":5,"output_tokens":1}
                        }
                    }),
                ],
            )
            .await;
            let request_c = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                next_websocket_request(&mut route_b),
            )
            .await
            .expect("the next turn did not reuse route B's exact socket");
            assert_eq!(request_c["previous_response_id"], "resp_b");
            assert_eq!(request_c["prompt_cache_key"], request_b["prompt_cache_key"]);
            assert!(
                request_c["input"].to_string().contains("third turn"),
                "route B continuation omitted the next input delta"
            );
            send_websocket_events(
                &mut route_b,
                [
                    serde_json::json!({"type":"response.created","response":{"id":"resp_c"}}),
                    serde_json::json!({
                        "type":"response.output_item.added",
                        "output_index":0,
                        "item":{"type":"message","id":"msg_c","role":"assistant","content":[]}
                    }),
                    serde_json::json!({
                        "type":"response.output_text.delta",
                        "output_index":0,
                        "delta":"continued-on-route-b"
                    }),
                    serde_json::json!({
                        "type":"response.output_item.done",
                        "output_index":0,
                        "item":{
                            "type":"message",
                            "id":"msg_c",
                            "role":"assistant",
                            "content":[{"type":"output_text","text":"continued-on-route-b"}]
                        }
                    }),
                    serde_json::json!({
                        "type":"response.completed",
                        "response":{
                            "id":"resp_c",
                            "status":"completed",
                            "usage":{"input_tokens":7,"output_tokens":1}
                        }
                    }),
                ],
            )
            .await;
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(75), listener.accept())
                    .await
                    .is_err()
            );
            (route_a_key, route_b_key)
        });

        let seed: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":false,
            "messages":[{"role":"user","content":"seed"}]
        }))
        .unwrap();
        let seed_response = provider
            .handle_messages(seed, search_context("pooled-seed", Some("pooled-raw-lane")))
            .await;
        assert_eq!(seed_response.status(), StatusCode::OK);
        let seed_body = axum::body::to_bytes(seed_response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&seed_body).contains("seed-answer"));

        let follow_up: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":false,
            "messages":[
                {"role":"user","content":"seed"},
                {"role":"assistant","content":"seed-answer"},
                {"role":"user","content":"follow up"}
            ]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                follow_up,
                search_context("pooled-follow-up", Some("pooled-raw-lane")),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("pooled-route-b-only"));
        assert!(!body.contains("pooled route A rejected"));

        let third_turn: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":false,
            "messages":[
                {"role":"user","content":"seed"},
                {"role":"assistant","content":"seed-answer"},
                {"role":"user","content":"follow up"},
                {"role":"assistant","content":"pooled-route-b-only"},
                {"role":"user","content":"third turn"}
            ]
        }))
        .unwrap();
        let third_response = provider
            .handle_messages(
                third_turn,
                search_context("pooled-third", Some("pooled-raw-lane")),
            )
            .await;
        assert_eq!(third_response.status(), StatusCode::OK);
        let third_body = axum::body::to_bytes(third_response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&third_body).contains("continued-on-route-b"));
        let (route_a_key, route_b_key) = server.await.unwrap();
        assert!(
            !continuation::has_continuation_for_tests(&route_a_key),
            "route A continuation state survived rejection"
        );
        assert!(
            continuation::has_continuation_for_tests(&route_b_key),
            "route B continuation state was not published"
        );
        continuation::clear_all_continuations_for_tests();
        websocket::clear_codex_websocket_pool_for_tests();
    }

    #[tokio::test]
    async fn auto_handshake_401_survives_failed_same_token_http_fallback() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_transport(config::CodexTransport::Auto);
        client.auth_manager().set_test_auth(StoredAuth {
            access: "auto-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("auto-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client);
        let server_client = provider.client.clone();
        let server = tokio::spawn(async move {
            let (mut handshake_a, _) = listener.accept().await.unwrap();
            let request =
                String::from_utf8_lossy(&read_http_request(&mut handshake_a).await).to_string();
            assert!(request.contains("upgrade: websocket"));
            assert!(request.contains("authorization: Bearer auto-a"));
            server_client.auth_manager().set_test_auth(StoredAuth {
                access: "auto-b".into(),
                refresh: "refresh-b".into(),
                account_id: Some("auto-account".into()),
                expires: u64::MAX,
            });
            handshake_a
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 5\r\nconnection: close\r\n\r\nstale",
                )
                .await
                .unwrap();

            let (mut fallback_a, _) = listener.accept().await.unwrap();
            let request =
                String::from_utf8_lossy(&read_http_request(&mut fallback_a).await).to_string();
            assert!(request.starts_with("POST "));
            assert!(!request.contains("upgrade: websocket"));
            assert!(request.contains("authorization: Bearer auto-a"));
            fallback_a
                .write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 8\r\nconnection: close\r\n\r\nfallback",
                )
                .await
                .unwrap();

            let (route_b, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_hdr_async(
                route_b,
                |request: &http::Request<()>, response| {
                    assert_eq!(
                        request.headers()[http::header::AUTHORIZATION],
                        "Bearer auto-b"
                    );
                    Ok(response)
                },
            )
            .await
            .unwrap();
            let request = next_websocket_request(&mut websocket).await;
            assert!(request.get("previous_response_id").is_none());
            send_websocket_events(
                &mut websocket,
                [
                    serde_json::json!({"type":"response.created","response":{"id":"auto-b-response"}}),
                    serde_json::json!({
                        "type":"response.output_item.added",
                        "output_index":0,
                        "item":{"type":"message","id":"auto-b-message"}
                    }),
                    serde_json::json!({
                        "type":"response.output_text.delta",
                        "output_index":0,
                        "delta":"auto-route-b"
                    }),
                    serde_json::json!({
                        "type":"response.output_item.done",
                        "output_index":0,
                        "item":{"type":"message","id":"auto-b-message"}
                    }),
                    serde_json::json!({
                        "type":"response.completed",
                        "response":{"id":"auto-b-response","usage":{}}
                    }),
                ],
            )
            .await;
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(125), listener.accept())
                    .await
                    .is_err(),
                "auto route sent another request after the single route B attempt"
            );
        });

        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":false,
            "messages":[{"role":"user","content":"auto fallback auth"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(request, search_context("auto-auth", Some("auto-auth-lane")))
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8_lossy(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .into_owned();
        assert!(body.contains("auto-route-b"), "{body}");
        assert_eq!(provider.client.route_rejection_refresh_count(), 1);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn live_handshake_401_rebuilds_before_downstream_publication() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_transport(config::CodexTransport::WebSocket);
        client.auth_manager().set_test_auth(StoredAuth {
            access: "live-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("live-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client);
        let server_client = provider.client.clone();
        let server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.unwrap();
            let first_request =
                String::from_utf8_lossy(&read_http_request(&mut first).await).into_owned();
            assert!(first_request.contains("authorization: Bearer live-a"));
            let session_a = first_request
                .lines()
                .find_map(|line| line.strip_prefix("session_id: "))
                .unwrap()
                .to_string();
            server_client.auth_manager().set_test_auth(StoredAuth {
                access: "live-b".into(),
                refresh: "refresh-b".into(),
                account_id: Some("live-account".into()),
                expires: u64::MAX,
            });
            first
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 5\r\nconnection: close\r\n\r\nstale",
                )
                .await
                .unwrap();

            let (second, _) = listener.accept().await.unwrap();
            let captured_headers = Arc::new(std::sync::Mutex::new(None));
            let callback_headers = captured_headers.clone();
            let mut websocket = tokio_tungstenite::accept_hdr_async(
                second,
                move |request: &http::Request<()>, response| {
                    *callback_headers.lock().unwrap() = Some(request.headers().clone());
                    Ok(response)
                },
            )
            .await
            .unwrap();
            let headers = captured_headers.lock().unwrap().clone().unwrap();
            assert_eq!(headers[http::header::AUTHORIZATION], "Bearer live-b");
            let session_b = headers["session_id"].to_str().unwrap().to_string();
            assert_ne!(session_a, session_b);
            assert_eq!(headers["x-codex-window-id"], format!("{session_b}:0"));
            let request = websocket
                .next()
                .await
                .unwrap()
                .unwrap()
                .into_text()
                .unwrap();
            let request: serde_json::Value = serde_json::from_str(&request).unwrap();
            assert_eq!(request["prompt_cache_key"], session_b);
            assert!(request.get("previous_response_id").is_none());
            for event in [
                serde_json::json!({"type":"response.created","response":{"id":"resp_b"}}),
                serde_json::json!({
                    "type":"response.output_item.added",
                    "output_index":0,
                    "item":{"type":"message","id":"msg_b"}
                }),
                serde_json::json!({
                    "type":"response.output_text.delta",
                    "output_index":0,
                    "delta":"live-route-b"
                }),
                serde_json::json!({
                    "type":"response.output_item.done",
                    "output_index":0,
                    "item":{"type":"message"}
                }),
                serde_json::json!({
                    "type":"response.completed",
                    "response":{
                        "id":"resp_b",
                        "status":"completed",
                        "usage":{"input_tokens":3,"output_tokens":1}
                    }
                }),
            ] {
                websocket
                    .send(Message::Text(event.to_string()))
                    .await
                    .unwrap();
            }
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(75), listener.accept())
                    .await
                    .is_err()
            );
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":true,
            "messages":[{"role":"user","content":"live full context"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                request,
                search_context("live-handshake-rebind", Some("live-raw-lane")),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8_lossy(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .into_owned();
        assert!(body.contains("live-route-b"));
        assert!(!body.contains("401"));
        assert!(!body.to_ascii_lowercase().contains("unauthorized"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn live_response_created_then_401_rebuilds_without_leaking_unauthorized() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_transport(config::CodexTransport::WebSocket);
        client.auth_manager().set_test_auth(StoredAuth {
            access: "created-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("created-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client);
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "created-rebind",
            Some("created-raw-lane".into()),
            None,
            EndpointKind::Messages,
        );
        let server_monitor = monitor.clone();
        let server_client = provider.client.clone();
        let server = tokio::spawn(async move {
            let (first, _) = listener.accept().await.unwrap();
            let first_headers = Arc::new(std::sync::Mutex::new(None));
            let callback_headers = first_headers.clone();
            let mut first = tokio_tungstenite::accept_hdr_async(
                first,
                move |request: &http::Request<()>, response| {
                    *callback_headers.lock().unwrap() = Some(request.headers().clone());
                    Ok(response)
                },
            )
            .await
            .unwrap();
            assert_eq!(
                first_headers.lock().unwrap().as_ref().unwrap()[http::header::AUTHORIZATION],
                "Bearer created-a"
            );
            let request_a = first.next().await.unwrap().unwrap().into_text().unwrap();
            let request_a: serde_json::Value = serde_json::from_str(&request_a).unwrap();
            first
                .send(Message::Text(
                    serde_json::json!({"type":"response.created","response":{"id":"route-a"}})
                        .to_string(),
                ))
                .await
                .unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                loop {
                    if server_monitor.snapshot().active.iter().any(|request| {
                        request.request_id == "created-rebind"
                            && request.generation_started_at.is_some()
                    }) {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("response.created was not observed before the 401");
            server_client.auth_manager().set_test_auth(StoredAuth {
                access: "created-b".into(),
                refresh: "refresh-b".into(),
                account_id: Some("created-account".into()),
                expires: u64::MAX,
            });
            first
                .send(Message::Text(
                    serde_json::json!({
                        "type":"response.failed",
                        "status_code":401,
                        "response":{"error":{"status":401,"message":"route-a unauthorized"}}
                    })
                    .to_string(),
                ))
                .await
                .unwrap();
            drop(first);

            let (second, _) = listener.accept().await.unwrap();
            let second_headers = Arc::new(std::sync::Mutex::new(None));
            let callback_headers = second_headers.clone();
            let mut second = tokio_tungstenite::accept_hdr_async(
                second,
                move |request: &http::Request<()>, response| {
                    *callback_headers.lock().unwrap() = Some(request.headers().clone());
                    Ok(response)
                },
            )
            .await
            .unwrap();
            assert_eq!(
                second_headers.lock().unwrap().as_ref().unwrap()[http::header::AUTHORIZATION],
                "Bearer created-b"
            );
            let request_b = second.next().await.unwrap().unwrap().into_text().unwrap();
            let request_b: serde_json::Value = serde_json::from_str(&request_b).unwrap();
            assert_ne!(request_a["prompt_cache_key"], request_b["prompt_cache_key"]);
            assert_eq!(request_a["input"], request_b["input"]);
            assert!(request_b.get("previous_response_id").is_none());
            for event in [
                serde_json::json!({"type":"response.created","response":{"id":"route-b"}}),
                serde_json::json!({
                    "type":"response.output_item.added",
                    "output_index":0,
                    "item":{"type":"message","id":"msg_b"}
                }),
                serde_json::json!({
                    "type":"response.output_text.delta",
                    "output_index":0,
                    "delta":"created-route-b"
                }),
                serde_json::json!({
                    "type":"response.output_item.done",
                    "output_index":0,
                    "item":{"type":"message"}
                }),
                serde_json::json!({
                    "type":"response.completed",
                    "response":{
                        "id":"route-b",
                        "status":"completed",
                        "usage":{"input_tokens":3,"output_tokens":1}
                    }
                }),
            ] {
                second.send(Message::Text(event.to_string())).await.unwrap();
            }
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(75), listener.accept())
                    .await
                    .is_err()
            );
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":true,
            "messages":[{"role":"user","content":"created full context"}]
        }))
        .unwrap();
        let mut ctx = search_context("created-rebind", Some("created-raw-lane"));
        ctx.monitor = Some(monitor);
        let response = provider.handle_messages(request, ctx).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8_lossy(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .into_owned();
        assert!(body.contains("created-route-b"));
        assert!(!body.contains("route-a unauthorized"));
        assert!(!body.contains("401"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn live_completed_tool_call_then_401_emits_auth_error_not_success_stop() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_transport(config::CodexTransport::WebSocket);
        client.auth_manager().set_test_auth(StoredAuth {
            access: "tool-auth-a".into(),
            refresh: String::new(),
            account_id: Some("tool-auth-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client);
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
            websocket.next().await.unwrap().unwrap();
            for event in [
                serde_json::json!({"type":"response.created","response":{"id":"tool-route-a"}}),
                serde_json::json!({
                    "type":"response.output_item.added",
                    "output_index":0,
                    "item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":""}
                }),
                serde_json::json!({
                    "type":"response.function_call_arguments.delta",
                    "output_index":0,
                    "delta":"{}"
                }),
                serde_json::json!({
                    "type":"response.output_item.done",
                    "output_index":0,
                    "item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":"{}"}
                }),
                serde_json::json!({
                    "type":"response.failed",
                    "status_code":401,
                    "response":{"error":{"status":401,"message":"tool route unauthorized"}}
                }),
            ] {
                websocket
                    .send(Message::Text(event.to_string()))
                    .await
                    .unwrap();
            }
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(125), listener.accept())
                    .await
                    .is_err(),
                "completed tool call was replayed after authentication failure"
            );
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":true,
            "messages":[{"role":"user","content":"read a file"}],
            "tools":[{"name":"Read","description":"read","input_schema":{"type":"object"}}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(request, search_context("tool-auth", Some("tool-auth-lane")))
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8_lossy(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .into_owned();
        assert!(body.contains(r#""type":"tool_use""#), "{body}");
        assert!(body.contains("event: error"), "{body}");
        assert!(body.contains("authentication_error"), "{body}");
        assert!(body.contains("Authentication failed"), "{body}");
        assert!(!body.contains(r#""stop_reason":"tool_use""#), "{body}");
        assert!(!body.contains("event: message_stop"), "{body}");
        assert!(!body.contains("tool route unauthorized"), "{body}");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn live_post_semantic_401_does_not_replay_and_refreshes_next_request() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let token_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let token_address = token_listener.local_addr().unwrap();
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            format!("http://{address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_auth_token_endpoint(format!("http://{token_address}/oauth/token"))
        .with_test_transport(config::CodexTransport::WebSocket);
        client.auth_manager().set_test_auth(StoredAuth {
            access: "semantic-a".into(),
            refresh: "refresh-a".into(),
            account_id: Some("semantic-account".into()),
            expires: u64::MAX,
        });
        let provider = CodexProvider::with_client(client);
        let token_server = tokio::spawn(async move {
            let (mut socket, _) = token_listener.accept().await.unwrap();
            let request =
                String::from_utf8_lossy(&read_http_request(&mut socket).await).to_string();
            assert!(request.contains("refresh_token=refresh-a"));
            let body =
                br#"{"access_token":"semantic-b","refresh_token":"refresh-b","expires_in":3600}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(body).await.unwrap();
        });
        let (no_replay_tx, no_replay_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_hdr_async(
                stream,
                |request: &tokio_tungstenite::tungstenite::handshake::server::Request,
                 response: tokio_tungstenite::tungstenite::handshake::server::Response| {
                    assert_eq!(request.headers()["authorization"], "Bearer semantic-a");
                    Ok(response)
                },
            )
            .await
            .unwrap();
            websocket.next().await.unwrap().unwrap();
            for event in [
                serde_json::json!({"type":"response.created","response":{"id":"route-a"}}),
                serde_json::json!({
                    "type":"response.output_item.added",
                    "output_index":0,
                    "item":{"type":"message","id":"msg_a"}
                }),
                serde_json::json!({
                    "type":"response.output_text.delta",
                    "output_index":0,
                    "delta":"published-route-a"
                }),
            ] {
                websocket
                    .send(Message::Text(event.to_string()))
                    .await
                    .unwrap();
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            websocket
                .send(Message::Text(
                    serde_json::json!({
                        "type":"response.failed",
                        "status_code":401,
                        "response":{"error":{"status":401,"message":"late unauthorized"}}
                    })
                    .to_string(),
                ))
                .await
                .unwrap();
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(125), listener.accept())
                    .await
                    .is_err(),
                "post-publication 401 replayed the request"
            );
            no_replay_tx.send(()).unwrap();

            let (stream, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_hdr_async(
                stream,
                |request: &tokio_tungstenite::tungstenite::handshake::server::Request,
                 response: tokio_tungstenite::tungstenite::handshake::server::Response| {
                    assert_eq!(request.headers()["authorization"], "Bearer semantic-b");
                    Ok(response)
                },
            )
            .await
            .unwrap();
            websocket.next().await.unwrap().unwrap();
            for event in [
                serde_json::json!({"type":"response.created","response":{"id":"route-b"}}),
                serde_json::json!({
                    "type":"response.output_item.added",
                    "output_index":0,
                    "item":{"type":"message","id":"msg_b"}
                }),
                serde_json::json!({
                    "type":"response.output_text.delta",
                    "output_index":0,
                    "delta":"next-route-b"
                }),
                serde_json::json!({
                    "type":"response.output_item.done",
                    "output_index":0,
                    "item":{"type":"message","id":"msg_b"}
                }),
                serde_json::json!({
                    "type":"response.completed",
                    "response":{"id":"route-b","usage":{}}
                }),
            ] {
                websocket
                    .send(Message::Text(event.to_string()))
                    .await
                    .unwrap();
            }
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":true,
            "messages":[{"role":"user","content":"semantic boundary"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                request,
                search_context("semantic-boundary", Some("semantic-raw-lane")),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8_lossy(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .into_owned();
        assert!(body.contains("published-route-a"));
        assert_eq!(body.matches("event: message_start").count(), 1, "{body}");
        assert!(body.contains("event: error"), "{body}");
        assert!(body.contains("authentication_error"), "{body}");
        assert!(body.contains("Authentication failed"), "{body}");
        for forbidden in [
            "response.failed",
            "status_code",
            "\"status\":401",
            "late unauthorized",
            "Unauthorized",
        ] {
            assert!(!body.contains(forbidden), "leaked {forbidden:?}: {body}");
        }
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if provider
                    .client
                    .auth_manager()
                    .get_auth()
                    .await
                    .unwrap()
                    .access
                    == "semantic-b"
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("post-semantic auth refresh did not prepare the next request");
        assert_eq!(provider.client.route_rejection_refresh_count(), 1);
        no_replay_rx.await.unwrap();

        let next_request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":true,
            "messages":[{"role":"user","content":"next request"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                next_request,
                search_context("semantic-next", Some("semantic-next-lane")),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8_lossy(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .into_owned();
        assert!(body.contains("next-route-b"), "{body}");
        server.await.unwrap();
        token_server.await.unwrap();
    }

    #[test]
    fn terminal_only_completed_upstream_is_empty_completion() {
        let body = upstream_sse(&[serde_json::json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "incomplete_details": null, "usage": {"input_tokens": 5, "output_tokens": 0}}
        })]);
        assert!(is_empty_codex_success_completion(&body, None));
    }

    #[test]
    fn terminal_only_done_upstream_is_empty_completion() {
        let body = upstream_sse(&[serde_json::json!({
            "type": "response.done",
            "response": {"id": "resp_1", "usage": {}}
        })]);
        assert!(is_empty_codex_success_completion(&body, None));
    }

    #[test]
    fn empty_message_item_is_empty_completion() {
        let body = upstream_sse(&[
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1"}
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message"}
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {"id": "resp_1", "usage": {}}
            }),
        ]);
        assert!(is_empty_codex_success_completion(&body, None));
    }

    #[test]
    fn upstream_with_text_is_not_empty_completion() {
        let body = upstream_sse(&[
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1"}
            }),
            serde_json::json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "hello"
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message"}
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {"id": "resp_1", "usage": {}}
            }),
        ]);
        assert!(!is_empty_codex_success_completion(&body, None));
    }

    #[test]
    fn upstream_with_tool_call_is_not_empty_completion() {
        let body = upstream_sse(&[
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": ""}
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{}"}
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {"id": "resp_1", "usage": {}}
            }),
        ]);
        assert!(!is_empty_codex_success_completion(&body, None));
    }

    #[test]
    fn terminal_only_incomplete_upstream_is_not_empty_completion() {
        let body = upstream_sse(&[serde_json::json!({
            "type": "response.incomplete",
            "response": {"id": "resp_1", "incomplete_details": {"reason": "max_output_tokens"}, "usage": {}}
        })]);
        assert!(!is_empty_codex_success_completion(&body, None));
    }

    #[test]
    fn upstream_without_terminal_event_is_not_empty_completion() {
        assert!(!is_empty_codex_success_completion(&upstream_sse(&[]), None));
    }

    fn request_with_tools(tools: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(serde_json::json!({
            "model": "gpt-5.6-luna",
            "messages": [{"role":"user", "content":"find it"}],
            "tools": tools
        }))
        .unwrap()
    }

    #[test]
    fn web_search_requests_leave_lite_lane_and_upgrade_luna() {
        let body = request_with_tools(serde_json::json!([
            {"type":"web_search_20250305", "name":"web_search"}
        ]));
        for (resolved, expected) in [
            ("gpt-5.6-luna", "gpt-5.6-sol"),
            ("gpt-5.6-sol", "gpt-5.6-sol"),
            ("gpt-5.6-terra", "gpt-5.6-terra"),
            ("gpt-5.4", "gpt-5.4"),
        ] {
            let mut model = resolved.to_string();
            let lite = apply_model_lane_for_request(&mut model, &body);
            assert!(!lite, "{resolved} with web_search must use the full lane");
            assert_eq!(model, expected);
        }
    }

    #[test]
    fn requests_without_web_search_keep_model_and_lite_lane() {
        let body = request_with_tools(serde_json::json!([
            {"name":"Bash", "input_schema":{}}
        ]));
        for (resolved, lite_expected) in [
            ("gpt-5.6-luna", true),
            ("gpt-5.6-sol", true),
            ("gpt-5.4", false),
        ] {
            let mut model = resolved.to_string();
            let lite = apply_model_lane_for_request(&mut model, &body);
            assert_eq!(model, resolved, "model must not change without web_search");
            assert_eq!(lite, lite_expected);
        }
    }

    #[test]
    fn generation_timing_ignores_control_events() {
        assert!(!codex_generation_event(&serde_json::json!({
            "type": "codex.rate_limits"
        })));
        assert!(!codex_generation_event(&serde_json::json!({
            "type": "keepalive"
        })));
        assert!(codex_generation_event(&serde_json::json!({
            "type": "response.created"
        })));
    }

    #[test]
    fn live_stream_progress_records_terminal_usage() {
        let monitor = crate::monitor::MonitorHandle::new(10);
        monitor.request_started(
            "request",
            None,
            None,
            crate::monitor::EndpointKind::Messages,
        );
        let ctx = RequestContext {
            req_id: "request".to_string(),
            session_id: None,
            session_seq: None,
            provider: "codex".to_string(),
            traffic: None,
            monitor: Some(monitor.clone()),
        };
        let chunk = b"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":12,\"output_tokens\":48}}\n\n";

        record_live_stream_progress(&ctx, chunk);

        let state = monitor.snapshot();
        assert_eq!(state.active[0].input_tokens, Some(12));
        assert_eq!(state.active[0].output_tokens, Some(48));
    }

    #[test]
    fn supported_models_includes_fast_variants() {
        let provider = CodexProvider::new();
        let models = provider.supported_models();
        assert!(models.contains(&"gpt-5.6-sol".to_string()));
        assert!(models.contains(&"gpt-5.6-sol-fast".to_string()));
        assert!(models.contains(&"gpt-5.6-terra".to_string()));
        assert!(models.contains(&"gpt-5.6-luna".to_string()));
        assert!(models.contains(&"gpt-5.4".to_string()));
        assert!(models.contains(&"gpt-5.4-mini".to_string()));
    }

    #[test]
    fn format_auth_saved_output_with_account() {
        assert_eq!(
            format_auth_saved_output("/tmp/auth.json", Some("acct_1")),
            "Auth saved in /tmp/auth.json\nAccount: acct_1\n"
        );
    }

    #[test]
    fn format_auth_saved_output_without_account() {
        assert_eq!(
            format_auth_saved_output("/tmp/auth.json", None),
            "Auth saved in /tmp/auth.json\n"
        );
    }

    #[test]
    fn format_expiry_with_future_expiry() {
        // 2100-01-01T00:00:00Z in ms
        let expires = 4102444800000;
        let now = 4102444790000; // 10s before
        let output = format_expiry(expires, now);
        assert!(output.starts_with("Expires: 2100-01-01T00:00:00.000Z (in "));
        assert!(output.ends_with("s)"));
    }

    #[test]
    fn format_expiry_with_past_expiry() {
        // 2000-01-01T00:00:00Z in ms
        let expires = 946684800000;
        let now = 946684810000; // 10s after
        let output = format_expiry(expires, now);
        assert!(output.starts_with("Expires: 2000-01-01T00:00:00.000Z (in -"));
    }

    #[tokio::test]
    async fn live_upstream_status_and_retry_after_are_preserved() {
        let err = client::CodexError {
            status: 422,
            message: "invalid request".to_string(),
            detail: Some("invalid request".to_string()),
            retry_after: Some("7".to_string()),
            origin: client::CodexErrorOrigin::WebSocketHandshake,
        };
        let response = map_codex_error_to_response(&err);
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            response.headers().get(http::header::RETRY_AFTER).unwrap(),
            "7"
        );
    }

    #[tokio::test]
    async fn statusless_codex_error_returns_source_message() {
        let err = client::CodexError {
            status: 0,
            message: "WebSocket connect error: HTTP error: 502 Bad Gateway".to_string(),
            detail: None,
            retry_after: None,
            origin: client::CodexErrorOrigin::WebSocket,
        };

        let response = map_codex_error_to_response(&err);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            body.pointer("/error/message").and_then(|v| v.as_str()),
            Some("WebSocket connect error: HTTP error: 502 Bad Gateway")
        );
    }

    #[tokio::test]
    async fn empty_live_completion_maps_to_explicit_service_unavailable() {
        let err = empty_live_completion_error();

        assert_eq!(err.status, 503);
        assert_eq!(err.detail.as_deref(), Some(EMPTY_CODEX_COMPLETION_DETAIL));
        assert_eq!(
            map_codex_error_to_response(&err).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[test]
    fn live_start_statusless_websocket_handshake_error_is_retryable() {
        let err = client::CodexError {
            status: 0,
            message: "WebSocket connect timeout after 15000ms".to_string(),
            detail: None,
            retry_after: None,
            origin: client::CodexErrorOrigin::WebSocketHandshake,
        };

        assert!(retryable_live_start_codex_error(&err));
    }

    #[test]
    fn live_start_proxy_tunnel_rejection_is_not_retryable() {
        let err = client::CodexError {
            status: 0,
            message: "WebSocket proxy tunnel was rejected".to_string(),
            detail: Some(websocket::WEBSOCKET_PROXY_TUNNEL_REJECTED_DETAIL.to_string()),
            retry_after: None,
            origin: client::CodexErrorOrigin::WebSocketHandshake,
        };

        assert!(!retryable_live_start_codex_error(&err));
    }

    #[test]
    fn live_start_payload_retry_detection_covers_rate_limit_and_overload() {
        assert!(retryable_live_start_payload(
            &serde_json::json!({
                "type": "codex.rate_limits",
                "rate_limits": {"limit_reached": true}
            }),
            "rate limit reached",
        ));
        assert!(retryable_live_start_payload(
            &serde_json::json!({
                "type": "response.failed",
                "response": {"error": {"type": "overloaded_error", "message": "overloaded"}}
            }),
            "overloaded",
        ));
        assert!(!retryable_live_start_payload(
            &serde_json::json!({
                "type": "response.failed",
                "response": {"error": {"message": "bad request"}}
            }),
            "bad request",
        ));
    }
}
