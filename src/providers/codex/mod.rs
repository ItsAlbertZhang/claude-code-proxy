pub(crate) mod admission;
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
pub(crate) mod state;
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
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

use crate::anthropic::error::json_error;
use crate::anthropic::schema::{CountTokensResponse, MessagesRequest};
use crate::anthropic::sse::parse_sse_events;
use crate::config;
use crate::logging::create_logger;
use crate::monitor::{CodexRecoveryCause, usage_from_anthropic_sse};
use crate::provider::{
    CliHandlers, Provider, RequestContext, ScopedRequestContext, compatible_explicit_identity,
    legacy_scope,
};
use crate::registry;
use crate::request_identity::{
    ConversationIdentity, LaneDomain, OpaqueLane, RequestPurpose, RequestScope,
};
use crate::retry::{compute_backoff_delay, sleep};

use self::auth::browser_login::run_browser_login;
use self::auth::device::DeviceAuthClient;
use self::auth::manager::CodexAuthManager;
use self::auth::token_store::file_store;
use self::client::{AuthRejectionBudget, BufferedRetryState, CodexHttpClient};
use self::compaction::{
    CompactionLease, CompactionStartPermit, abort_compaction_for_route,
    activate_compaction_for_route, apply_compaction_replay_for_route, begin_compaction_for_route,
    prepare_compaction_request, request_compaction_bound_result, reserve_compaction_start,
    store_compaction_for_route,
};
#[cfg(test)]
use self::continuation::continuation_candidate_for_owner;
use self::continuation::{
    ContinuationPublication, ContinuationReservation, abort_continuation_for_owner,
    record_continuation_for_owner, reserve_continuation_for_owner,
};
use self::count_tokens::count_translated_tokens;
use self::state::{CodexBoundRoute, ProtocolLane};
use self::translate::accumulate::accumulate_response_scoped;
use self::translate::live_stream::LiveStreamTranslator;
use self::translate::model_allowlist::{
    assert_allowed_model, full_lane_web_search_model, resolve_model_request_with_config_override,
    uses_responses_lite_with_full_lane,
};
use self::translate::reducer::finish_metadata_from_upstream_scoped;
use self::translate::request::{
    ServiceTier, TranslateOptions, has_hosted_web_search, is_compact_messages_request,
    translate_request_scoped,
};

const MAX_RETRYABLE_LIVE_STREAM_RETRIES: u32 = 10;
const MAX_EMPTY_COMPLETION_RETRIES: u32 = 10;
const EMPTY_CODEX_COMPLETION_DETAIL: &str = "empty_codex_completion";

fn merge_claude_fast_service_tier(
    model_tier: Option<ServiceTier>,
    claude_fast_intent: bool,
) -> Option<ServiceTier> {
    if claude_fast_intent {
        Some(ServiceTier::Priority)
    } else {
        model_tier
    }
}

fn service_tier_label(service_tier: &ServiceTier) -> &'static str {
    match service_tier {
        ServiceTier::Priority => "priority",
        ServiceTier::Flex => "flex",
    }
}
use self::translate::stream::translate_stream_bytes_scoped;

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

pub(crate) fn clear_conversation_state(identity: &ConversationIdentity) {
    continuation::clear_continuation_for_owner(Some(identity));
    websocket::invalidate_codex_websocket_pool_owner(identity);
    let lane = RequestScope::from_conversation_identity(
        Some(identity.clone()),
        RequestPurpose::Conversation,
    )
    .provider_lane(LaneDomain::CodexConversation);
    if let Some(lane) = lane {
        compaction::clear_compactions_for_lane(lane);
    }
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

impl CodexProvider {
    async fn handle_messages_inner(
        &self,
        body: MessagesRequest,
        scoped: ScopedRequestContext,
        claude_fast_intent: bool,
    ) -> Response {
        let (ctx, scope) = scoped.into_parts();
        let conversation_identity = scope.conversational_lane().cloned();
        let legacy_compaction_session = match conversation_identity.as_ref() {
            Some(ConversationIdentity::Main(session))
                if ctx.session_id.as_deref() == Some(session.as_str()) =>
            {
                Some(session.clone())
            }
            _ => None,
        };
        let codex_lane = scope.provider_lane(LaneDomain::CodexConversation);
        let read_lane = scope.provider_lane(LaneDomain::CodexReadRewrite);
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
                monitor.codex_request_lane(&ctx.req_id, false);
            }
            // Generate the stateless ID once. Stateful requests replace it with the
            // current route's opaque conversation key on every route attempt.
            let (base_search_request, query) =
                match search::build_search_request(&body, &resolved.model, None) {
                    Ok(request) => request,
                    Err(error) => {
                        return json_error(
                            StatusCode::BAD_REQUEST,
                            "invalid_request_error",
                            error.to_string(),
                        );
                    }
                };
            let mut route = match self
                .client
                .bind_conversation_route(codex_lane, ProtocolLane::ResponsesFull)
                .await
            {
                Ok(route) => route,
                Err(error) => return map_codex_error_to_response(&error),
            };
            let auth_rejection_budget = Arc::new(AuthRejectionBudget::default());
            let mut buffered_retry_state = BufferedRetryState::default();
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
            let (search_response, search_request) = loop {
                let mut search_request = base_search_request.clone();
                if let Some(id) = route.conversation_key_encoded() {
                    search_request.id = id;
                }
                match self
                    .client
                    .post_search_bound_with_retry_state(
                        &route,
                        &search_request,
                        &ctx,
                        &mut buffered_retry_state,
                    )
                    .await
                {
                    Ok(response) => break (response, search_request),
                    Err(error) => {
                        if error.status == 401 && auth_rejection_budget.try_claim() {
                            let Some(next_route) = self
                                .client
                                .refresh_conversation_route_after_rejection(&route)
                                .await
                                .into_route()
                            else {
                                return map_codex_error_to_response(&error);
                            };
                            route = next_route;
                            continue;
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
        resolved.service_tier =
            merge_claude_fast_service_tier(resolved.service_tier, claude_fast_intent);
        let full_lane = config::codex_full_lane();
        let use_responses_lite =
            apply_model_lane_for_request(&mut resolved.model, &body, full_lane);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &resolved.model);
            monitor.codex_request_lane(&ctx.req_id, use_responses_lite);
        }

        let original_translated = match translate_request_scoped(
            &body,
            TranslateOptions {
                // Route binding owns all upstream conversation identity. Never
                // translate a raw session or Agent identifier into a cache key.
                session_id: None,
                service_tier: resolved.service_tier.clone(),
                model: resolved.model.clone(),
                use_responses_lite,
            },
            read_lane,
        ) {
            Ok(translated) => translated,
            Err(error) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    error.to_string(),
                );
            }
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.codex_acceleration_resolved(
                &ctx.req_id,
                claude_fast_intent.then_some("fast"),
                original_translated
                    .service_tier
                    .as_ref()
                    .map(service_tier_label),
            );
        }

        let compact_boundary = is_compact_messages_request(&body);
        let server_compaction_enabled = self.server_compaction_enabled();
        if !server_compaction_enabled && let Some(lane) = codex_lane {
            compaction::clear_compactions_for_lane(lane);
        }

        let client = self.client.clone();
        // Only conversational Messages requests that may use a Responses
        // WebSocket participate in owner admission. Validation, translation,
        // and the standalone-search fast path have already completed.
        let mut conversation_admission = match client.configured_transport() {
            config::CodexTransport::WebSocket | config::CodexTransport::Auto => {
                admission::admit_messages(&scope).await
            }
            config::CodexTransport::Http => None,
        };

        // Reserve the owner generation before route/auth awaits. Canonical
        // append-only evaluation is deferred until route binding and any
        // portable-summary replay have produced the actual upstream request.
        let previous_response_id_enabled = config::codex_previous_response_id();
        let owner_continuation = reserve_continuation_for_owner(
            conversation_identity.as_ref(),
            previous_response_id_enabled,
        )
        .with_previous_id_metrics(ctx.monitor.clone(), &ctx.req_id);
        let compaction_start_permit = (server_compaction_enabled && compact_boundary)
            .then(|| reserve_compaction_start(codex_lane))
            .flatten();
        let mut logical_continuation = owner_continuation.clone();
        let mut cleanup =
            LiveRequestStateCleanup::new(owner_continuation.clone(), None, compaction_start_permit);

        let mut route = match client
            .bind_conversation_route(
                codex_lane,
                ProtocolLane::from_uses_responses_lite(use_responses_lite),
            )
            .await
        {
            Ok(route) => route,
            Err(error) => {
                cleanup.abort();
                return map_codex_error_to_response(&error);
            }
        };
        let auth_rejection_budget = Arc::new(AuthRejectionBudget::default());
        let mut buffered_retry_state = BufferedRetryState::default();
        let mut route_rebuilt = false;
        let mut live_start_attempt = 0_u32;
        let mut empty_completion_attempt = 0_u32;
        let mut upstream_started = false;

        // A compact boundary first runs one hidden, ownerful CompactionTrigger
        // turn. The caller-visible Claude plaintext summary is detached work
        // performed only after that hidden terminal has been published.
        let hidden_compaction_permit = cleanup.take_compaction_start_permit();
        if compact_boundary && let Some(permit) = hidden_compaction_permit.as_ref() {
            'hidden_routes: loop {
                let translated = bind_messages_request_to_route(&original_translated, &route);
                let prepared = prepare_compaction_request(&translated);
                let mut hidden_continuation = owner_continuation
                    .evaluate_hidden_compaction(prepared.request())
                    .bind_route(&route);
                if route_rebuilt {
                    hidden_continuation =
                        hidden_continuation.full_context_retry(CodexRecoveryCause::AuthRejection);
                }
                cleanup.replace_continuation(hidden_continuation.clone());
                log_compaction_event(
                    "server_compaction_triggered",
                    &ctx,
                    prepared.request().input.len(),
                    None,
                );
                if let Some(monitor) = ctx.monitor.as_ref() {
                    monitor.compaction_started(&ctx.req_id);
                }
                let Some(build_lease) =
                    begin_compaction_for_route(permit, &route, &translated.model)
                else {
                    abort_continuation_for_owner(&hidden_continuation);
                    log_compaction_event(
                        "server_compaction_failed",
                        &ctx,
                        prepared.request().input.len(),
                        Some("compaction route lease could not be acquired"),
                    );
                    break 'hidden_routes;
                };
                cleanup.replace_compaction_lease(Some(build_lease.clone()));
                let mut compaction_ctx = ctx.clone();
                compaction_ctx.monitor = None;
                match request_compaction_bound_result(
                    client.as_ref(),
                    &route,
                    prepared,
                    &compaction_ctx,
                    &hidden_continuation,
                    &mut buffered_retry_state,
                )
                .await
                {
                    Ok(bound) => {
                        if let Some(publication) = bound.continuation_publication() {
                            if record_continuation_for_owner(
                                &hidden_continuation,
                                publication.request,
                                Some(publication.response_id),
                                Some(publication.socket_id),
                                publication.output_items,
                            ) == ContinuationPublication::Rejected
                            {
                                websocket::invalidate_codex_websocket_pool_for_reservation(
                                    &hidden_continuation,
                                );
                            }
                        } else {
                            // HTTP can install compacted history, but cannot
                            // fabricate a socket-bound continuation baseline.
                            abort_continuation_for_owner(&hidden_continuation);
                        }
                        if store_compaction_for_route(&build_lease, bound.compacted_history()) {
                            log_compaction_event(
                                "server_compaction_completed",
                                &ctx,
                                translated.input.len(),
                                None,
                            );
                        } else {
                            cleanup.abort_compaction();
                            log_compaction_event(
                                "server_compaction_failed",
                                &ctx,
                                translated.input.len(),
                                Some("compaction state exceeded the in-memory limit"),
                            );
                        }
                        break 'hidden_routes;
                    }
                    Err(error) => {
                        cleanup.abort_compaction();
                        log_compaction_event(
                            "server_compaction_failed",
                            &ctx,
                            translated.input.len(),
                            Some(&error.to_string()),
                        );
                        if let compaction::CompactionError::Upstream(upstream) = error {
                            if upstream.is_in_band_auth_rejection() {
                                client.refresh_conversation_auth_after_rejection_in_background(
                                    &route,
                                    auth_rejection_budget.clone(),
                                );
                                cleanup.abort();
                                return map_codex_error_to_response(&upstream);
                            }
                            if upstream.is_replayable_auth_rejection() {
                                let Some(next_route) = rebuild_route_after_unauthorized(
                                    client.as_ref(),
                                    &route,
                                    auth_rejection_budget.as_ref(),
                                )
                                .await
                                else {
                                    cleanup.abort();
                                    return map_codex_error_to_response(&upstream);
                                };
                                route = next_route;
                                route_rebuilt = true;
                                continue 'hidden_routes;
                            }
                        }
                        abort_continuation_for_owner(&hidden_continuation);
                        break 'hidden_routes;
                    }
                }
            }

            logical_continuation = ContinuationReservation::detached(
                original_translated.input.len(),
                "claude_plaintext_summary",
            );
            cleanup.replace_continuation(logical_continuation.clone());
            route = route.auxiliary();
            route_rebuilt = false;
        }

        'routes: loop {
            let mut translated = bind_messages_request_to_route(&original_translated, &route);

            if server_compaction_enabled && !compact_boundary {
                if let Some(replay) = apply_compaction_replay_for_route(&route, &translated) {
                    translated = replay.request;
                    cleanup.replace_compaction_lease(Some(replay.lease));
                } else if let Some(replay) = compaction::apply_compaction_replay(
                    legacy_compaction_session.as_deref(),
                    &translated,
                ) {
                    translated = replay;
                }
            }

            let mut request_continuation = logical_continuation
                .evaluate(&translated)
                .bind_route(&route);
            if route_rebuilt {
                request_continuation =
                    request_continuation.full_context_retry(CodexRecoveryCause::AuthRejection);
            }
            cleanup.replace_continuation(request_continuation.clone());

            if !upstream_started {
                if let Some(monitor) = ctx.monitor.as_ref() {
                    monitor.upstream_started(&ctx.req_id);
                }
                upstream_started = true;
            }
            if want_stream {
                match live_stream_route_attempt(
                    client.clone(),
                    &route,
                    message_id.clone(),
                    model,
                    ctx.clone(),
                    translated.clone(),
                    request_continuation.clone(),
                    cleanup.compaction_lease().cloned(),
                    read_lane,
                    auth_rejection_budget.clone(),
                    &mut live_start_attempt,
                    &mut conversation_admission,
                )
                .await
                {
                    LiveRouteOutcome::Response(response) => {
                        cleanup.disarm();
                        return response;
                    }
                    LiveRouteOutcome::Unauthorized(error) => {
                        cleanup.abort_compaction();
                        if compact_boundary {
                            cleanup.disarm();
                            return map_codex_error_to_response(&error);
                        }
                        let Some(next_route) = rebuild_route_after_unauthorized(
                            client.as_ref(),
                            &route,
                            auth_rejection_budget.as_ref(),
                        )
                        .await
                        else {
                            cleanup.abort();
                            return map_codex_error_to_response(&error);
                        };
                        route = next_route;
                        route_rebuilt = true;
                        continue 'routes;
                    }
                }
            }

            let mut active_continuation = Some(request_continuation.clone());
            let upstream = loop {
                let response = match client
                    .post_codex_bound_with_retry_state(
                        &route,
                        &translated,
                        &ctx,
                        active_continuation.as_ref(),
                        &mut buffered_retry_state,
                    )
                    .await
                {
                    Ok(response) => response,
                    Err(error) => {
                        if error.is_in_band_auth_rejection() {
                            client.refresh_conversation_auth_after_rejection_in_background(
                                &route,
                                auth_rejection_budget.clone(),
                            );
                            cleanup.abort();
                            return map_codex_error_to_response(&error);
                        }
                        if error.is_replayable_auth_rejection() {
                            cleanup.abort_compaction();
                            if compact_boundary {
                                cleanup.disarm();
                                return map_codex_error_to_response(&error);
                            }
                            if websocket::invalidate_codex_websocket_pool_for_reservation(
                                &request_continuation,
                            ) {
                                request_continuation
                                    .record_socket_cause(CodexRecoveryCause::AuthRejection);
                            }
                            let Some(next_route) = rebuild_route_after_unauthorized(
                                client.as_ref(),
                                &route,
                                auth_rejection_budget.as_ref(),
                            )
                            .await
                            else {
                                cleanup.abort();
                                return map_codex_error_to_response(&error);
                            };
                            route = next_route;
                            route_rebuilt = true;
                            continue 'routes;
                        }
                        cleanup.abort();
                        return map_codex_error_to_response(&error);
                    }
                };
                if !is_empty_codex_success_completion(&response.body) {
                    break response;
                }
                // A successful terminal event with no output would translate into
                // an empty end_turn; retry with full context instead.
                let error = empty_buffered_completion_error();
                drop_live_continuation_for_retry(
                    &mut active_continuation,
                    CodexRecoveryCause::EmptyCompletion,
                );
                if empty_completion_attempt >= MAX_EMPTY_COMPLETION_RETRIES {
                    cleanup.abort();
                    return map_codex_error_to_response(&error);
                }
                let delay = compute_backoff_delay(empty_completion_attempt, None);
                if delay.exceeds_budget {
                    cleanup.abort();
                    return map_codex_error_to_response(&error);
                }
                empty_completion_attempt += 1;
                sleep(delay.wait_ms).await;
            };

            return if want_stream {
                let estimated_input_tokens = count_translated_tokens(&translated);
                let sse_bytes = match translate_stream_bytes_scoped(
                    &upstream.body,
                    &message_id,
                    model,
                    estimated_input_tokens,
                    ctx.traffic.as_deref(),
                    read_lane,
                ) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        cleanup.abort();
                        return map_codex_failure_to_response(&format!(
                            "Stream translation error: {error}"
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
                    &request_continuation,
                    &translated,
                    &upstream.body,
                    upstream.socket_id,
                    cleanup.compaction_lease(),
                    read_lane,
                );
                cleanup.disarm();

                let headers = [
                    (http::header::CONTENT_TYPE, "text/event-stream"),
                    (http::header::CACHE_CONTROL, "no-cache"),
                    (http::header::CONNECTION, "keep-alive"),
                ];
                (headers, sse_bytes).into_response()
            } else {
                match accumulate_response_scoped(
                    &upstream.body,
                    &message_id,
                    model,
                    ctx.traffic.as_deref(),
                    read_lane,
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
                            &request_continuation,
                            &translated,
                            &upstream.body,
                            upstream.socket_id,
                            cleanup.compaction_lease(),
                            read_lane,
                        );
                        cleanup.disarm();
                        (StatusCode::OK, Json(json)).into_response()
                    }
                    Err(error) => {
                        cleanup.abort();
                        map_codex_failure_to_response(&format!("Accumulation error: {error}"))
                    }
                }
            };
        }
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
        let scope = legacy_scope(&ctx, RequestPurpose::Conversation);
        self.handle_messages_inner(body, ScopedRequestContext::new(ctx, scope), false)
            .await
    }

    async fn handle_messages_with_conversation_identity(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
        conversation_identity: Option<ConversationIdentity>,
    ) -> Response {
        let identity = compatible_explicit_identity(&ctx, conversation_identity);
        let scope =
            RequestScope::from_conversation_identity(identity, RequestPurpose::Conversation);
        self.handle_messages_inner(body, ScopedRequestContext::new(ctx, scope), false)
            .await
    }

    async fn handle_messages_with_claude_fast_intent(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
        conversation_identity: Option<ConversationIdentity>,
        claude_fast_intent: bool,
    ) -> Response {
        let identity = compatible_explicit_identity(&ctx, conversation_identity);
        let scope =
            RequestScope::from_conversation_identity(identity, RequestPurpose::Conversation);
        self.handle_messages_inner(
            body,
            ScopedRequestContext::new(ctx, scope),
            claude_fast_intent,
        )
        .await
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
        let full_lane = config::codex_full_lane();
        let use_responses_lite =
            apply_model_lane_for_request(&mut resolved.model, &body, full_lane);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, &resolved.model);
            monitor.codex_request_lane(&ctx.req_id, use_responses_lite);
        }

        let translated = match translate_request_scoped(
            &body,
            TranslateOptions {
                session_id: None,
                service_tier: resolved.service_tier.clone(),
                model: resolved.model.clone(),
                use_responses_lite,
            },
            None,
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
fn apply_model_lane_for_request(
    model: &mut String,
    body: &MessagesRequest,
    full_lane: bool,
) -> bool {
    if has_hosted_web_search(body) {
        *model = full_lane_web_search_model(model).to_string();
        return false;
    }
    uses_responses_lite_with_full_lane(model, full_lane)
}

fn bind_messages_request_to_route(
    original: &translate::request::ResponsesRequest,
    route: &CodexBoundRoute,
) -> translate::request::ResponsesRequest {
    let mut request = original.clone();
    request.prompt_cache_key = route.namespace_prompt_cache_key("messages");
    request
}

async fn rebuild_route_after_unauthorized(
    client: &CodexHttpClient,
    rejected: &CodexBoundRoute,
    budget: &AuthRejectionBudget,
) -> Option<CodexBoundRoute> {
    if !budget.try_claim() {
        return None;
    }
    client
        .refresh_conversation_route_after_rejection(rejected)
        .await
        .into_route()
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
    continuation: &ContinuationReservation,
    compaction_lease: Option<&CompactionLease>,
) {
    if let Some(lease) = compaction_lease {
        abort_compaction_for_route(lease);
    }
    abort_continuation_for_owner(continuation);
}

struct LiveRequestStateCleanup {
    continuation: ContinuationReservation,
    compaction_lease: Option<CompactionLease>,
    compaction_start_permit: Option<CompactionStartPermit>,
    armed: bool,
}

impl LiveRequestStateCleanup {
    fn new(
        continuation: ContinuationReservation,
        compaction_lease: Option<CompactionLease>,
        compaction_start_permit: Option<CompactionStartPermit>,
    ) -> Self {
        Self {
            continuation,
            compaction_lease,
            compaction_start_permit,
            armed: true,
        }
    }

    fn replace_continuation(&mut self, continuation: ContinuationReservation) {
        self.continuation = continuation;
    }

    fn take_compaction_start_permit(&mut self) -> Option<CompactionStartPermit> {
        self.compaction_start_permit.take()
    }

    fn compaction_lease(&self) -> Option<&CompactionLease> {
        self.compaction_lease.as_ref()
    }

    fn replace_compaction_lease(&mut self, lease: Option<CompactionLease>) {
        self.compaction_lease = lease;
    }

    fn abort_compaction(&mut self) {
        if let Some(lease) = self.compaction_lease.take() {
            abort_compaction_for_route(&lease);
        }
    }

    fn abort(&mut self) {
        if self.armed {
            abort_request_state(&self.continuation, self.compaction_lease.as_ref());
            self.armed = false;
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for LiveRequestStateCleanup {
    fn drop(&mut self) {
        if self.armed {
            abort_request_state(&self.continuation, self.compaction_lease.as_ref());
        }
    }
}

enum LiveStreamStart {
    Response(Response),
    Retry {
        error: client::CodexError,
        full_context_retry_attempted: bool,
    },
    Unauthorized(client::CodexError),
}

enum LiveRouteOutcome {
    Response(Response),
    Unauthorized(client::CodexError),
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
async fn live_stream_response(
    client: Arc<CodexHttpClient>,
    route: CodexBoundRoute,
    message_id: String,
    model: &str,
    ctx: RequestContext,
    request_body: translate::request::ResponsesRequest,
    continuation: ContinuationReservation,
    compaction_lease: Option<CompactionLease>,
    compaction_start_permit: Option<CompactionStartPermit>,
    read_lane: Option<OpaqueLane>,
) -> Response {
    let mut cleanup = LiveRequestStateCleanup::new(
        continuation.clone(),
        compaction_lease.clone(),
        compaction_start_permit,
    );
    let mut attempt = 0;
    let mut conversation_admission = None;
    let outcome = live_stream_route_attempt(
        client,
        &route,
        message_id,
        model,
        ctx,
        request_body,
        continuation,
        compaction_lease,
        read_lane,
        Arc::new(AuthRejectionBudget::default()),
        &mut attempt,
        &mut conversation_admission,
    )
    .await;
    match outcome {
        LiveRouteOutcome::Response(response) => {
            cleanup.disarm();
            response
        }
        LiveRouteOutcome::Unauthorized(error) => {
            cleanup.abort();
            map_codex_error_to_response(&error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn live_stream_route_attempt(
    client: Arc<CodexHttpClient>,
    route: &CodexBoundRoute,
    message_id: String,
    model: &str,
    ctx: RequestContext,
    request_body: translate::request::ResponsesRequest,
    request_continuation: ContinuationReservation,
    compaction_lease: Option<CompactionLease>,
    read_lane: Option<OpaqueLane>,
    auth_rejection_budget: Arc<AuthRejectionBudget>,
    attempt: &mut u32,
    conversation_admission: &mut Option<tokio::sync::OwnedMutexGuard<()>>,
) -> LiveRouteOutcome {
    let model = model.to_string();
    let mut continuation = Some(request_continuation.clone());

    loop {
        let start = match client.configured_transport() {
            config::CodexTransport::Http => {
                client
                    .stream_codex_http_events_bound(
                        route,
                        &request_body,
                        &ctx,
                        auth_rejection_budget.clone(),
                    )
                    .await
            }
            config::CodexTransport::WebSocket => {
                client
                    .stream_codex_websocket_events_bound(
                        route,
                        &request_body,
                        &ctx,
                        continuation.as_ref(),
                    )
                    .await
            }
            config::CodexTransport::Auto => {
                client
                    .stream_codex_auto_events_bound(
                        route,
                        &request_body,
                        &ctx,
                        continuation.as_ref(),
                        auth_rejection_budget.clone(),
                    )
                    .await
            }
        };
        let upstream_events = match start {
            Ok(events) => events,
            Err(error) if error.status == 401 => {
                return LiveRouteOutcome::Unauthorized(error);
            }
            Err(error) if retryable_live_start_codex_error(&error) => {
                let cause = client::continuation_retry_cause(&error)
                    .unwrap_or(CodexRecoveryCause::TransportFailure);
                let dropped = drop_live_continuation_for_retry(&mut continuation, cause);
                if dropped && is_missing_previous_response_error(&error) {
                    *attempt += 1;
                    continue;
                }
                if *attempt >= MAX_RETRYABLE_LIVE_STREAM_RETRIES {
                    abort_request_state(&request_continuation, compaction_lease.as_ref());
                    return LiveRouteOutcome::Response(map_codex_error_to_response(&error));
                }
                let delay = compute_backoff_delay(*attempt, error.retry_after.as_deref());
                if delay.exceeds_budget {
                    abort_request_state(&request_continuation, compaction_lease.as_ref());
                    return LiveRouteOutcome::Response(map_codex_error_to_response(&error));
                }
                *attempt += 1;
                sleep(delay.wait_ms).await;
                continue;
            }
            Err(error) => {
                abort_request_state(&request_continuation, compaction_lease.as_ref());
                return LiveRouteOutcome::Response(map_codex_error_to_response(&error));
            }
        };

        match live_stream_response_once(
            upstream_events,
            client.clone(),
            route.clone(),
            auth_rejection_budget.clone(),
            message_id.clone(),
            &model,
            ctx.clone(),
            request_continuation.clone(),
            request_body.clone(),
            compaction_lease.clone(),
            read_lane,
            conversation_admission,
        )
        .await
        {
            LiveStreamStart::Response(response) => {
                return LiveRouteOutcome::Response(response);
            }
            LiveStreamStart::Unauthorized(error) => {
                return LiveRouteOutcome::Unauthorized(error);
            }
            LiveStreamStart::Retry {
                error,
                full_context_retry_attempted,
            } => {
                let cause = client::continuation_retry_cause(&error)
                    .unwrap_or(CodexRecoveryCause::TransportFailure);
                let dropped = drop_live_continuation_for_retry(&mut continuation, cause);
                if full_context_retry_attempted && client::is_continuation_retry_error(&error) {
                    abort_request_state(&request_continuation, compaction_lease.as_ref());
                    return LiveRouteOutcome::Response(map_codex_error_to_response(&error));
                }
                if dropped && is_missing_previous_response_error(&error) {
                    *attempt += 1;
                    continue;
                }
                if *attempt >= MAX_RETRYABLE_LIVE_STREAM_RETRIES {
                    abort_request_state(&request_continuation, compaction_lease.as_ref());
                    return LiveRouteOutcome::Response(map_codex_error_to_response(&error));
                }
                let delay = compute_backoff_delay(*attempt, error.retry_after.as_deref());
                if delay.exceeds_budget {
                    abort_request_state(&request_continuation, compaction_lease.as_ref());
                    return LiveRouteOutcome::Response(map_codex_error_to_response(&error));
                }
                *attempt += 1;
                sleep(delay.wait_ms).await;
            }
        }
    }
}

fn provider_retry(
    upstream_events: &websocket::CodexWebSocketEventStream,
    error: client::CodexError,
) -> LiveStreamStart {
    let full_context_retry_attempted = upstream_events.used_full_context_retry();
    upstream_events.mark_provider_retry_handoff();
    LiveStreamStart::Retry {
        error,
        full_context_retry_attempted,
    }
}

#[allow(clippy::too_many_arguments)]
async fn live_stream_response_once(
    mut upstream_events: websocket::CodexWebSocketEventStream,
    client: Arc<CodexHttpClient>,
    route: CodexBoundRoute,
    auth_rejection_budget: Arc<AuthRejectionBudget>,
    message_id: String,
    model: &str,
    ctx: RequestContext,
    request_continuation: ContinuationReservation,
    request_body: translate::request::ResponsesRequest,
    compaction_lease: Option<CompactionLease>,
    read_lane: Option<OpaqueLane>,
    conversation_admission: &mut Option<tokio::sync::OwnedMutexGuard<()>>,
) -> LiveStreamStart {
    let estimated_input_tokens = count_translated_tokens(&request_body);
    let mut translator = LiveStreamTranslator::with_stable_read_lane(
        message_id,
        model.to_string(),
        estimated_input_tokens,
        read_lane,
    );
    let mut upstream_sse_body = Vec::new();
    // Keep protocol framing private until real output makes a transparent retry unsafe.
    // Every branch that consumes pending_chunk returns, so it is never flushed twice.
    let mut pending_chunk = Vec::new();
    let mut generation_started = false;

    while let Some(item) = upstream_events.recv().await {
        let payload = match item {
            Ok(payload) => payload,
            Err(error) if error.status == 401 => {
                let rejection = handle_live_auth_rejection(
                    &upstream_events,
                    &mut translator,
                    pending_chunk,
                    &ctx,
                    &client,
                    &route,
                    auth_rejection_budget,
                    request_continuation,
                    compaction_lease,
                    error,
                );
                upstream_events.cancel_and_wait().await;
                return rejection;
            }
            Err(error) => {
                if retryable_live_start_codex_error(&error) {
                    let retry = provider_retry(&upstream_events, error);
                    upstream_events.cancel_and_wait().await;
                    return retry;
                }
                abort_request_state(&request_continuation, compaction_lease.as_ref());
                upstream_events.cancel_and_wait().await;
                return LiveStreamStart::Response(map_codex_error_to_response(&error));
            }
        };
        if let Some(failure) = events::failure_with_status(&payload, 401) {
            let error = client::CodexError {
                status: 401,
                message: failure.message.clone(),
                detail: Some(failure.message),
                retry_after: failure.retry_after,
                origin: client::CodexErrorOrigin::WebSocket,
            };
            let rejection = handle_live_auth_rejection(
                &upstream_events,
                &mut translator,
                pending_chunk,
                &ctx,
                &client,
                &route,
                auth_rejection_budget,
                request_continuation,
                compaction_lease,
                error,
            );
            upstream_events.cancel_and_wait().await;
            return rejection;
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
                    let retry = provider_retry(
                        &upstream_events,
                        client::CodexError {
                            status,
                            message: message.clone(),
                            detail: Some(message),
                            retry_after: retry_after_from_live_payload(&payload),
                            origin: client::CodexErrorOrigin::WebSocket,
                        },
                    );
                    upstream_events.cancel_and_wait().await;
                    return retry;
                }
                abort_request_state(&request_continuation, compaction_lease.as_ref());
                upstream_events.cancel_and_wait().await;
                return LiveStreamStart::Response(map_codex_failure_to_response(&message));
            }
        };
        pending_chunk.extend_from_slice(&chunk);
        if terminal
            && is_codex_success_terminal_event(&payload)
            && !translator.has_semantic_output()
        {
            let retry = provider_retry(&upstream_events, empty_live_completion_error());
            upstream_events.wait_for_completion().await;
            return retry;
        }
        if translator.has_semantic_output() && !pending_chunk.is_empty() {
            record_live_stream_downstream_capture(&ctx, &pending_chunk);
            record_live_stream_progress(&ctx, &pending_chunk);
            if terminal {
                upstream_events.wait_for_completion().await;
                update_continuation_from_upstream(
                    &request_continuation,
                    &request_body,
                    &upstream_sse_body,
                    upstream_events.socket_id(),
                    compaction_lease.as_ref(),
                    read_lane,
                );
                return LiveStreamStart::Response(single_live_stream_response(pending_chunk));
            }
            return LiveStreamStart::Response(remaining_live_stream_response(
                upstream_events,
                translator,
                pending_chunk,
                ctx,
                request_continuation,
                request_body,
                upstream_sse_body,
                compaction_lease,
                read_lane,
                client,
                route,
                auth_rejection_budget,
                conversation_admission.take(),
            ));
        }
        if terminal {
            upstream_events.wait_for_completion().await;
            update_continuation_from_upstream(
                &request_continuation,
                &request_body,
                &upstream_sse_body,
                upstream_events.socket_id(),
                compaction_lease.as_ref(),
                read_lane,
            );
            if pending_chunk.is_empty() {
                return LiveStreamStart::Response(empty_live_stream_response());
            }
            record_live_stream_downstream_capture(&ctx, &pending_chunk);
            record_live_stream_progress(&ctx, &pending_chunk);
            return LiveStreamStart::Response(single_live_stream_response(pending_chunk));
        }
    }

    upstream_events.wait_for_completion().await;
    provider_retry(
        &upstream_events,
        client::CodexError {
            status: 0,
            message: "WebSocket connection closed before terminal Codex response event".to_string(),
            detail: Some(websocket::WEBSOCKET_MISSING_TERMINAL_DETAIL.to_string()),
            retry_after: None,
            origin: client::CodexErrorOrigin::WebSocket,
        },
    )
}

fn invalidate_live_route_socket(continuation: &ContinuationReservation, socket_id: Option<u64>) {
    let removed = if socket_id.is_some() {
        websocket::invalidate_codex_websocket_pool_socket(continuation, socket_id)
    } else {
        websocket::invalidate_codex_websocket_pool_for_reservation(continuation)
    };
    if removed {
        continuation.record_socket_cause(CodexRecoveryCause::AuthRejection);
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_live_auth_rejection(
    upstream_events: &websocket::CodexWebSocketEventStream,
    translator: &mut LiveStreamTranslator,
    mut pending_chunk: Vec<u8>,
    ctx: &RequestContext,
    client: &Arc<CodexHttpClient>,
    route: &CodexBoundRoute,
    auth_rejection_budget: Arc<AuthRejectionBudget>,
    request_continuation: ContinuationReservation,
    compaction_lease: Option<CompactionLease>,
    error: client::CodexError,
) -> LiveStreamStart {
    invalidate_live_route_socket(&request_continuation, upstream_events.socket_id());
    if !translator.has_semantic_output() {
        upstream_events.mark_provider_retry_handoff();
        return LiveStreamStart::Unauthorized(error);
    }

    abort_request_state(&request_continuation, compaction_lease.as_ref());
    client.refresh_conversation_auth_after_rejection_in_background(route, auth_rejection_budget);
    let chunk = translator.error_chunk(
        "Authentication failed",
        "authentication_error",
        ctx.traffic.as_deref(),
    );
    pending_chunk.extend_from_slice(&chunk);
    if !pending_chunk.is_empty() {
        record_live_stream_downstream_capture(ctx, &pending_chunk);
        record_live_stream_progress(ctx, &pending_chunk);
    }
    LiveStreamStart::Response(single_live_stream_response(pending_chunk))
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
    if payload.get("type").and_then(|value| value.as_str()) == Some("keepalive")
        && payload
            .get("_ccp_http_silence")
            .and_then(|value| value.as_bool())
            == Some(true)
    {
        return Ok((translator.ping_chunk(traffic), false));
    }
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
    mut upstream_events: websocket::CodexWebSocketEventStream,
    mut translator: LiveStreamTranslator,
    first_chunk: Vec<u8>,
    ctx: RequestContext,
    request_continuation: ContinuationReservation,
    request_body: translate::request::ResponsesRequest,
    mut upstream_sse_body: Vec<u8>,
    compaction_lease: Option<CompactionLease>,
    read_lane: Option<OpaqueLane>,
    client: Arc<CodexHttpClient>,
    route: CodexBoundRoute,
    auth_rejection_budget: Arc<AuthRejectionBudget>,
    conversation_admission: Option<tokio::sync::OwnedMutexGuard<()>>,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    tokio::spawn(async move {
        let _conversation_admission = conversation_admission;
        if tx.send(Ok(Bytes::from(first_chunk))).await.is_err() {
            cancel_downstream_live_stream(
                upstream_events,
                &request_continuation,
                compaction_lease.as_ref(),
            )
            .await;
            return;
        }
        loop {
            let item = tokio::select! {
                biased;
                _ = tx.closed() => {
                    cancel_downstream_live_stream(
                        upstream_events,
                        &request_continuation,
                        compaction_lease.as_ref(),
                    )
                    .await;
                    return;
                }
                item = upstream_events.recv() => item,
            };
            let Some(item) = item else {
                break;
            };
            match item {
                Ok(payload) => {
                    if events::failure_with_status(&payload, 401).is_some() {
                        invalidate_live_route_socket(
                            &request_continuation,
                            upstream_events.socket_id(),
                        );
                        abort_request_state(&request_continuation, compaction_lease.as_ref());
                        upstream_events.wait_for_completion().await;
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
                    append_upstream_sse_payload(&mut upstream_sse_body, &payload);
                    let (chunk, terminal) = match translate_live_stream_payload(
                        &mut translator,
                        &payload,
                        ctx.traffic.as_deref(),
                    ) {
                        Ok(result) => result,
                        Err(message) => {
                            abort_request_state(&request_continuation, compaction_lease.as_ref());
                            upstream_events.cancel_and_wait().await;
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
                            cancel_downstream_live_stream(
                                upstream_events,
                                &request_continuation,
                                compaction_lease.as_ref(),
                            )
                            .await;
                            return;
                        }
                    }
                    if terminal {
                        upstream_events.wait_for_completion().await;
                        update_continuation_from_upstream(
                            &request_continuation,
                            &request_body,
                            &upstream_sse_body,
                            upstream_events.socket_id(),
                            compaction_lease.as_ref(),
                            read_lane,
                        );
                        return;
                    }
                }
                Err(err) => {
                    if err.status == 401 {
                        invalidate_live_route_socket(
                            &request_continuation,
                            upstream_events.socket_id(),
                        );
                        abort_request_state(&request_continuation, compaction_lease.as_ref());
                        upstream_events.wait_for_completion().await;
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
                    abort_request_state(&request_continuation, compaction_lease.as_ref());
                    upstream_events.wait_for_completion().await;
                    let chunk =
                        translator.finish_after_closed_completed_tool_call(ctx.traffic.as_deref());
                    if !chunk.is_empty() {
                        record_live_stream_progress(&ctx, &chunk);
                        let _ = tx.send(Ok(Bytes::from(chunk))).await;
                        return;
                    }
                    let error_type = codex_stream_error_type(&err);
                    let chunk = translator.error_chunk(
                        codex_error_message(&err),
                        error_type,
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

        upstream_events.wait_for_completion().await;
        abort_request_state(&request_continuation, compaction_lease.as_ref());
        let chunk = translator.finish_after_closed_completed_tool_call(ctx.traffic.as_deref());
        if !chunk.is_empty() {
            record_live_stream_progress(&ctx, &chunk);
            let _ = tx.send(Ok(Bytes::from(chunk))).await;
            return;
        }
        let chunk = translator.error_chunk(
            "Upstream event stream closed before terminal Codex response event",
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

async fn cancel_downstream_live_stream(
    upstream_events: websocket::CodexWebSocketEventStream,
    continuation: &ContinuationReservation,
    compaction_lease: Option<&CompactionLease>,
) {
    abort_request_state(continuation, compaction_lease);
    let socket_id = upstream_events.cancel_and_wait_socket_id().await;
    if websocket::invalidate_codex_websocket_pool_socket(continuation, socket_id) {
        continuation.record_socket_cause(CodexRecoveryCause::Cancelled);
    }
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
fn is_empty_codex_success_completion(upstream_sse: &[u8]) -> bool {
    use self::translate::reducer::{ReducerEvent, TERM_COMPLETED, TERM_DONE};

    let Ok(events) = self::translate::reducer::reduce_upstream_bytes(upstream_sse) else {
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
    if err.origin == client::CodexErrorOrigin::Http {
        // The HTTP event stream owns its bounded retry budget. Retrying the
        // exhausted error here would multiply attempts across both layers.
        return false;
    }
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
    matches!(
        err.detail.as_deref(),
        Some("previous_response_not_found")
            | Some(websocket::WEBSOCKET_CONTINUATION_SOCKET_MISSING_DETAIL)
    )
}

fn drop_live_continuation_for_retry(
    continuation: &mut Option<ContinuationReservation>,
    cause: CodexRecoveryCause,
) -> bool {
    if continuation
        .as_ref()
        .and_then(|reservation| reservation.candidate().previous_response_id.as_deref())
        .is_none()
    {
        return false;
    }

    if let Some(reservation) = continuation.as_ref() {
        *continuation = Some(reservation.full_context_retry(cause));
    }
    true
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

#[allow(clippy::too_many_arguments)]
fn update_continuation_from_upstream(
    continuation: &ContinuationReservation,
    request_body: &translate::request::ResponsesRequest,
    upstream_body: &[u8],
    socket_id: Option<u64>,
    compaction_lease: Option<&CompactionLease>,
    read_lane: Option<OpaqueLane>,
) {
    match finish_metadata_from_upstream_scoped(upstream_body, read_lane) {
        Ok(Some(finish)) if finish.continuation_eligible => {
            if let Some(lease) = compaction_lease {
                activate_compaction_for_route(lease, &finish.output_items);
            }
            record_continuation_for_owner(
                continuation,
                request_body,
                finish.response_id.as_deref(),
                socket_id,
                &finish.output_items,
            );
        }
        _ => abort_request_state(continuation, compaction_lease),
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
mod tests {
    use futures_util::{SinkExt, StreamExt};
    use http_body_util::BodyExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio_tungstenite::tungstenite::Message;

    use super::*;

    #[test]
    fn claude_fast_promotes_only_the_derived_service_tier() {
        assert_eq!(merge_claude_fast_service_tier(None, false), None);
        assert_eq!(
            merge_claude_fast_service_tier(Some(ServiceTier::Flex), false),
            Some(ServiceTier::Flex)
        );
        assert_eq!(
            merge_claude_fast_service_tier(None, true),
            Some(ServiceTier::Priority)
        );
        assert_eq!(
            merge_claude_fast_service_tier(Some(ServiceTier::Flex), true),
            Some(ServiceTier::Priority)
        );
        assert_eq!(
            merge_claude_fast_service_tier(Some(ServiceTier::Priority), true),
            Some(ServiceTier::Priority)
        );
    }

    #[test]
    fn existing_fast_alias_stays_priority_without_claude_fast() {
        let resolved = resolve_model_request_with_config_override("gpt-5.5-fast", false);
        assert_eq!(resolved.model, "gpt-5.5");
        assert_eq!(
            merge_claude_fast_service_tier(resolved.service_tier, false),
            Some(ServiceTier::Priority)
        );
    }

    #[test]
    fn normal_model_uses_priority_when_claude_fast_is_detected() {
        let resolved = resolve_model_request_with_config_override("gpt-5.5", false);
        assert_eq!(resolved.model, "gpt-5.5");
        assert_eq!(
            merge_claude_fast_service_tier(resolved.service_tier, true),
            Some(ServiceTier::Priority)
        );
    }

    fn live_test_request(text: &str) -> translate::request::ResponsesRequest {
        translate::request::ResponsesRequest {
            model: "gpt-5.6-sol".to_string(),
            instructions: None,
            input: vec![translate::request::ResponsesInputItem::Message {
                role: "user".to_string(),
                content: vec![translate::request::ResponsesContentPart::InputText {
                    text: text.to_string(),
                }],
            }],
            tools: None,
            tool_choice: None,
            store: false,
            stream: true,
            parallel_tool_calls: true,
            include: None,
            client_metadata: None,
            service_tier: None,
            prompt_cache_key: None,
            text: translate::request::ResponsesText {
                verbosity: None,
                format: None,
            },
            reasoning: None,
        }
    }

    fn live_test_context(session_id: &str) -> RequestContext {
        RequestContext {
            req_id: format!("request-{session_id}"),
            session_id: Some(session_id.to_string()),
            session_seq: None,
            provider: "codex".to_string(),
            traffic: None,
            monitor: None,
        }
    }

    fn authenticated_live_test_client(base_url: String) -> Arc<CodexHttpClient> {
        authenticated_live_test_client_with_transport(base_url, config::CodexTransport::WebSocket)
    }

    fn authenticated_live_test_client_with_transport(
        base_url: String,
        transport: config::CodexTransport,
    ) -> Arc<CodexHttpClient> {
        let client = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            base_url,
            1_000,
            1_000,
            0,
        )
        .with_test_transport(transport);
        client
            .auth_manager()
            .set_test_auth(auth::token_store::StoredAuth {
                access: "test".to_string(),
                refresh: String::new(),
                expires: u64::MAX,
                account_id: Some("acct".to_string()),
            });
        Arc::new(client)
    }

    async fn bind_live_test_route(
        client: &CodexHttpClient,
        owner: &ConversationIdentity,
        continuation: &ContinuationReservation,
    ) -> (CodexBoundRoute, ContinuationReservation) {
        let lane = RequestScope::from_conversation_identity(
            Some(owner.clone()),
            RequestPurpose::Conversation,
        )
        .provider_lane(LaneDomain::CodexConversation);
        let route = client
            .bind_conversation_route(lane, ProtocolLane::ResponsesFull)
            .await
            .unwrap();
        let continuation = continuation.bind_route(&route);
        (route, continuation)
    }

    async fn next_live_websocket_request(
        websocket: &mut tokio_tungstenite::WebSocketStream<TcpStream>,
    ) -> serde_json::Value {
        loop {
            match websocket.next().await {
                Some(Ok(Message::Ping(payload))) => {
                    websocket.send(Message::Pong(payload)).await.unwrap();
                }
                Some(Ok(Message::Text(text))) => return serde_json::from_str(&text).unwrap(),
                other => panic!("unexpected WebSocket request frame: {other:?}"),
            }
        }
    }

    async fn emit_live_event(
        websocket: &mut tokio_tungstenite::WebSocketStream<TcpStream>,
        event: &serde_json::Value,
    ) {
        websocket
            .send(Message::Text(event.to_string()))
            .await
            .unwrap();
    }

    async fn read_http_request(socket: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = socket.read(&mut buffer).await.unwrap();
            assert!(read > 0, "request ended before its body was complete");
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

    fn captured_header<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
        headers.lines().find_map(|line| {
            let (header_name, value) = line.split_once(':')?;
            header_name
                .eq_ignore_ascii_case(name)
                .then_some(value.trim())
        })
    }

    fn messages_test_context(req_id: &str, session_id: Option<&str>) -> RequestContext {
        RequestContext {
            req_id: req_id.to_string(),
            session_id: session_id.map(str::to_string),
            session_seq: None,
            provider: "codex".to_string(),
            traffic: None,
            monitor: None,
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

    fn upstream_sse(events: &[serde_json::Value]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for event in events {
            bytes.extend_from_slice(format!("data: {event}\n\n").as_bytes());
        }
        bytes
    }

    const COMPACTION_SUMMARY: &str =
        "portable provider summary with enough detail to anchor replay safely";

    fn compaction_output() -> Vec<translate::request::ResponsesInputItem> {
        serde_json::from_value(serde_json::json!([{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": COMPACTION_SUMMARY}]
        }]))
        .unwrap()
    }

    fn compaction_replay_request() -> translate::request::ResponsesRequest {
        let mut request = live_test_request("unused");
        request.input = serde_json::from_value(serde_json::json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":COMPACTION_SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]))
        .unwrap();
        request
    }

    fn native_compaction(tag: &str) -> Vec<translate::request::ResponsesInputItem> {
        vec![translate::request::ResponsesInputItem::Compaction {
            encrypted_content: tag.to_string(),
        }]
    }

    #[test]
    fn terminal_only_completed_upstream_is_empty_completion() {
        let body = upstream_sse(&[serde_json::json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "incomplete_details": null, "usage": {"input_tokens": 5, "output_tokens": 0}}
        })]);
        assert!(is_empty_codex_success_completion(&body));
    }

    #[test]
    fn terminal_only_done_upstream_is_empty_completion() {
        let body = upstream_sse(&[serde_json::json!({
            "type": "response.done",
            "response": {"id": "resp_1", "usage": {}}
        })]);
        assert!(is_empty_codex_success_completion(&body));
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
        assert!(is_empty_codex_success_completion(&body));
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
        assert!(!is_empty_codex_success_completion(&body));
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
        assert!(!is_empty_codex_success_completion(&body));
    }

    #[test]
    fn terminal_only_incomplete_upstream_is_not_empty_completion() {
        let body = upstream_sse(&[serde_json::json!({
            "type": "response.incomplete",
            "response": {"id": "resp_1", "incomplete_details": {"reason": "max_output_tokens"}, "usage": {}}
        })]);
        assert!(!is_empty_codex_success_completion(&body));
    }

    #[test]
    fn upstream_without_terminal_event_is_not_empty_completion() {
        assert!(!is_empty_codex_success_completion(&upstream_sse(&[])));
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
        for full_lane in [false, true] {
            for (resolved, expected) in [
                ("gpt-5.6-luna", "gpt-5.6-sol"),
                ("gpt-5.6-sol", "gpt-5.6-sol"),
                ("gpt-5.6-terra", "gpt-5.6-terra"),
                ("gpt-5.4", "gpt-5.4"),
            ] {
                let mut model = resolved.to_string();
                let lite = apply_model_lane_for_request(&mut model, &body, full_lane);
                assert!(!lite, "{resolved} with web_search must use the full lane");
                assert_eq!(model, expected);
            }
        }
    }

    #[test]
    fn requests_without_web_search_apply_full_lane_flag() {
        let body = request_with_tools(serde_json::json!([
            {"name":"Bash", "input_schema":{}}
        ]));
        for (resolved, full_lane, lite_expected) in [
            ("gpt-5.6-luna", false, true),
            ("gpt-5.6-luna", true, true),
            ("gpt-5.6-sol", false, true),
            ("gpt-5.6-sol", true, false),
            ("gpt-5.6-terra", false, true),
            ("gpt-5.6-terra", true, false),
            ("gpt-5.4", false, false),
            ("gpt-5.4", true, false),
        ] {
            let mut model = resolved.to_string();
            let lite = apply_model_lane_for_request(&mut model, &body, full_lane);
            assert_eq!(model, resolved, "model must not change without web_search");
            assert_eq!(
                lite, lite_expected,
                "model={resolved}, full_lane={full_lane}"
            );
        }
    }

    #[tokio::test]
    async fn standalone_search_401_rebuilds_stateful_route_and_preserves_stateless_id() {
        for (case, session_id) in [("stateful", Some("raw-search-lane")), ("stateless", None)] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let client = CodexHttpClient::new_for_test(
                reqwest::Client::builder().no_proxy().build().unwrap(),
                format!("http://{address}/v1/responses"),
                1_000,
                1_000,
                0,
            );
            client
                .auth_manager()
                .set_test_auth(auth::token_store::StoredAuth {
                    access: format!("search-{case}-a"),
                    refresh: "refresh-a".into(),
                    expires: u64::MAX,
                    account_id: Some("search-account".into()),
                });
            let provider = CodexProvider::with_client(client);
            let server_client = provider.client.clone();
            let server = tokio::spawn(async move {
                let mut captured = Vec::new();
                for attempt in 0..2 {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    captured.push(http_request_parts(&read_http_request(&mut socket).await));
                    if attempt == 0 {
                        server_client
                            .auth_manager()
                            .set_test_auth(auth::token_store::StoredAuth {
                                access: format!("search-{case}-b"),
                                refresh: "refresh-b".into(),
                                expires: u64::MAX,
                                account_id: Some("search-account".into()),
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
                    tokio::time::timeout(std::time::Duration::from_millis(100), listener.accept())
                        .await
                        .is_err(),
                    "search sent a third route attempt"
                );
                captured
            });

            let response = provider
                .handle_messages(
                    standalone_search_request(),
                    messages_test_context(&format!("search-{case}"), session_id),
                )
                .await;
            assert_eq!(response.status(), StatusCode::OK, "{case}");
            let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();

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
            if session_id.is_some() {
                assert_ne!(id_a, id_b);
                for ((headers, _), id) in captured.iter().zip([id_a, id_b]) {
                    let id = id.as_str().unwrap();
                    for name in ["session_id", "x-client-request-id", "x-codex-window-id"] {
                        assert!(headers.contains(&format!("{name}: {id}")));
                    }
                    assert!(!headers.contains("raw-search-lane"));
                }
            } else {
                assert_eq!(id_a, id_b, "stateless search changed its generated id");
                assert!(
                    id_a.as_str().is_some_and(|id| id.starts_with("search-")),
                    "stateless search id was not generated"
                );
                for (headers, _) in &captured {
                    assert!(!headers.contains("\nsession_id: "));
                    assert!(!headers.contains("x-client-request-id:"));
                    assert!(!headers.contains("x-codex-window-id:"));
                }
            }
        }
    }

    #[tokio::test]
    async fn buffered_messages_401_rebuilds_full_route_bound_request_once() {
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
        client
            .auth_manager()
            .set_test_auth(auth::token_store::StoredAuth {
                access: "messages-a".into(),
                refresh: "refresh-a".into(),
                expires: u64::MAX,
                account_id: Some("messages-account".into()),
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
                    server_client
                        .auth_manager()
                        .set_test_auth(auth::token_store::StoredAuth {
                            access: "messages-b".into(),
                            refresh: "refresh-b".into(),
                            expires: u64::MAX,
                            account_id: Some("messages-account".into()),
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
                tokio::time::timeout(std::time::Duration::from_millis(100), listener.accept())
                    .await
                    .is_err(),
                "Messages sent a third route attempt"
            );
            captured
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-5.4",
            "max_tokens": 256,
            "stream": false,
            "messages": [{"role": "user", "content": "original full context"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                request,
                messages_test_context("messages-rebind", Some("messages-raw-lane")),
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
        assert!(captured[0].0.contains("authorization: Bearer messages-a"));
        assert!(captured[1].0.contains("authorization: Bearer messages-b"));
        let key_a = captured[0].1["prompt_cache_key"].as_str().unwrap();
        let key_b = captured[1].1["prompt_cache_key"].as_str().unwrap();
        assert_ne!(key_a, key_b);
        let mut route_keys = Vec::new();
        for ((headers, _), prompt_key) in captured.iter().zip([key_a, key_b]) {
            let route_key = captured_header(headers, "session_id").unwrap();
            assert_eq!(
                captured_header(headers, "x-client-request-id"),
                Some(route_key)
            );
            assert_eq!(
                captured_header(headers, "x-codex-window-id"),
                Some(route_key)
            );
            assert_ne!(route_key, prompt_key);
            assert!(!headers.contains("messages-raw-lane"));
            route_keys.push(route_key.to_string());
        }
        assert_ne!(route_keys[0], route_keys[1]);
        assert_eq!(captured[0].1["input"], captured[1].1["input"]);
        assert!(captured[1].1.get("previous_response_id").is_none());
    }

    #[tokio::test]
    async fn buffered_messages_second_401_returns_route_b_without_route_c() {
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
        client
            .auth_manager()
            .set_test_auth(auth::token_store::StoredAuth {
                access: "bounded-a".into(),
                refresh: "refresh-a".into(),
                expires: u64::MAX,
                account_id: Some("bounded-account".into()),
            });
        let provider = CodexProvider::with_client(client);
        let server_client = provider.client.clone();
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(http_request_parts(&read_http_request(&mut socket).await));
                server_client
                    .auth_manager()
                    .set_test_auth(auth::token_store::StoredAuth {
                        access: if attempt == 0 {
                            "bounded-b"
                        } else {
                            "bounded-c"
                        }
                        .into(),
                        refresh: format!("refresh-{}", attempt + 2),
                        expires: u64::MAX,
                        account_id: Some("bounded-account".into()),
                    });
                let body = if attempt == 0 {
                    b"route-a-rejected".as_slice()
                } else {
                    b"route-b-rejected".as_slice()
                };
                let head = format!(
                    "HTTP/1.1 401 Unauthorized\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(body).await.unwrap();
            }
            let third =
                tokio::time::timeout(std::time::Duration::from_millis(125), listener.accept())
                    .await;
            (third, captured)
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-5.4",
            "max_tokens": 256,
            "stream": false,
            "messages": [{"role": "user", "content": "bounded retry"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                request,
                messages_test_context("bounded-retry", Some("bounded-raw-lane")),
            )
            .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("route-b-rejected"));

        let (third, captured) = server.await.unwrap();
        assert!(third.is_err(), "Messages sent route C");
        assert_eq!(captured.len(), 2);
        assert!(captured[0].0.contains("authorization: Bearer bounded-a"));
        assert!(captured[1].0.contains("authorization: Bearer bounded-b"));
        assert_ne!(
            captured[0].1["prompt_cache_key"],
            captured[1].1["prompt_cache_key"]
        );
        assert_eq!(captured[0].1["input"], captured[1].1["input"]);
    }

    #[tokio::test]
    async fn compaction_401_rebinds_permit_and_primary_send_to_route_b() {
        let _compaction_guard = compaction::lock_compaction_registry_for_async_tests().await;
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
        client
            .auth_manager()
            .set_test_auth(auth::token_store::StoredAuth {
                access: "compact-a".into(),
                refresh: "refresh-a".into(),
                expires: u64::MAX,
                account_id: Some("compact-account".into()),
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
        let success = upstream_sse(&[
            serde_json::json!({
                "type":"response.output_item.added",
                "output_index":0,
                "item":{"type":"message","id":"main-b"}
            }),
            serde_json::json!({
                "type":"response.output_text.delta",
                "output_index":0,
                "delta":"route-b-compacted"
            }),
            serde_json::json!({
                "type":"response.output_item.done",
                "output_index":0,
                "item":{"type":"message","id":"main-b"}
            }),
            serde_json::json!({
                "type":"response.completed",
                "response":{"id":"main-b","status":"completed","usage":{}}
            }),
        ]);
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for attempt in 0..3 {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(http_request_parts(&read_http_request(&mut socket).await));
                match attempt {
                    0 => {
                        server_client
                            .auth_manager()
                            .set_test_auth(auth::token_store::StoredAuth {
                                access: "compact-b".into(),
                                refresh: "refresh-b".into(),
                                expires: u64::MAX,
                                account_id: Some("compact-account".into()),
                            });
                        socket
                            .write_all(
                                b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 15\r\nconnection: close\r\n\r\ncompact-route-a",
                            )
                            .await
                            .unwrap();
                    }
                    1 => {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            compacted.len()
                        );
                        socket.write_all(head.as_bytes()).await.unwrap();
                        socket.write_all(&compacted).await.unwrap();
                    }
                    2 => {
                        let head = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            success.len()
                        );
                        socket.write_all(head.as_bytes()).await.unwrap();
                        socket.write_all(&success).await.unwrap();
                    }
                    _ => unreachable!(),
                }
            }
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(100), listener.accept())
                    .await
                    .is_err(),
                "compaction request sent route C"
            );
            captured
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":false,
            "system":"You are a helpful AI assistant tasked with summarizing conversations",
            "messages":[{"role":"user","content":"summarize this boundary"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                request,
                messages_test_context("compact-rebind", Some("compact-raw-lane")),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("route-b-compacted"));

        let captured = server.await.unwrap();
        assert_eq!(captured.len(), 3);
        assert!(captured[0].0.contains("authorization: Bearer compact-a"));
        assert!(captured[1].0.contains("authorization: Bearer compact-b"));
        assert!(captured[2].0.contains("authorization: Bearer compact-b"));
        let has_trigger = |body: &serde_json::Value| {
            body["input"].as_array().is_some_and(|input| {
                input
                    .iter()
                    .any(|item| item["type"] == "compaction_trigger")
            })
        };
        assert!(has_trigger(&captured[0].1));
        assert!(has_trigger(&captured[1].1));
        assert!(!has_trigger(&captured[2].1));
        assert_ne!(
            captured[0].1["prompt_cache_key"],
            captured[1].1["prompt_cache_key"]
        );
        assert!(captured[2].1["prompt_cache_key"].is_null());
        for header in ["session_id", "x-client-request-id", "x-codex-window-id"] {
            assert!(
                captured_header(&captured[2].0, header).is_none(),
                "detached summary leaked {header}"
            );
        }
        compaction::clear_all_compactions_for_tests();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "current_thread")]
    async fn read_correction_survives_route_rollover_and_stays_agent_scoped() {
        let _read_guard = translate::read_rewrite::READ_REWRITE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
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
        client
            .auth_manager()
            .set_test_auth(auth::token_store::StoredAuth {
                access: "read-route-a".into(),
                refresh: "read-refresh-a".into(),
                expires: u64::MAX,
                account_id: Some("read-account".into()),
            });
        let provider = CodexProvider::with_client(client);
        let read_call = "call_read_after_rollover";
        let first_response = upstream_sse(&[
            serde_json::json!({
                "type":"response.output_item.added",
                "output_index":0,
                "item":{"type":"function_call","call_id":read_call,"name":"Read"}
            }),
            serde_json::json!({
                "type":"response.function_call_arguments.delta",
                "output_index":0,
                "delta":"{\"file_path\":\"/tmp/route-read\",\"offset\":1300007,\"limit\":20}"
            }),
            serde_json::json!({
                "type":"response.output_item.done",
                "output_index":0,
                "item":{
                    "type":"function_call",
                    "call_id":read_call,
                    "name":"Read",
                    "arguments":"{\"file_path\":\"/tmp/route-read\",\"offset\":1300007,\"limit\":20}"
                }
            }),
            serde_json::json!({
                "type":"response.completed",
                "response":{"id":"resp_read_a","status":"completed","usage":{}}
            }),
        ]);
        let success = upstream_sse(&[
            serde_json::json!({
                "type":"response.output_item.added",
                "output_index":0,
                "item":{"type":"message","id":"msg_read_done"}
            }),
            serde_json::json!({
                "type":"response.output_text.delta",
                "output_index":0,
                "delta":"done"
            }),
            serde_json::json!({
                "type":"response.output_item.done",
                "output_index":0,
                "item":{"type":"message","id":"msg_read_done"}
            }),
            serde_json::json!({
                "type":"response.completed",
                "response":{"id":"resp_read_done","status":"completed","usage":{}}
            }),
        ]);
        let server = tokio::spawn(async move {
            let mut captured = Vec::new();
            for response_body in [&first_response, &success, &success] {
                let (mut socket, _) = listener.accept().await.unwrap();
                captured.push(http_request_parts(&read_http_request(&mut socket).await));
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    response_body.len()
                );
                socket.write_all(head.as_bytes()).await.unwrap();
                socket.write_all(response_body).await.unwrap();
            }
            captured
        });

        let scope = |agent: &str, req_id: &str| {
            let mut context = messages_test_context(req_id, Some("raw-shared-session"));
            context.session_id = Some("raw-context-must-not-own-read-state".to_string());
            ScopedRequestContext::new(
                context,
                RequestScope::from_conversation_identity(
                    Some(ConversationIdentity::Agent(
                        "raw-shared-session".to_string(),
                        agent.to_string(),
                    )),
                    RequestPurpose::Conversation,
                ),
            )
        };
        let initial: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":false,
            "messages":[{"role":"user","content":"read the file"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages_inner(initial, scope("agent-a", "read-first"), false)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let downstream: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(downstream["content"][0]["name"], "Read");
        assert!(downstream["content"][0]["input"].get("offset").is_none());

        provider
            .client
            .auth_manager()
            .set_test_auth(auth::token_store::StoredAuth {
                access: "read-route-b".into(),
                refresh: "read-refresh-b".into(),
                expires: u64::MAX,
                account_id: Some("read-account".into()),
            });
        let result_request = || {
            serde_json::from_value::<MessagesRequest>(serde_json::json!({
                "model":"gpt-5.4",
                "max_tokens":256,
                "stream":false,
                "messages":[
                    {"role":"assistant","content":[{
                        "type":"tool_use",
                        "id":read_call,
                        "name":"Read",
                        "input":{"file_path":"/tmp/route-read","limit":20}
                    }]},
                    {"role":"user","content":[{
                        "type":"tool_result",
                        "tool_use_id":read_call,
                        "content":[{"type":"text","text":"1\tcontent"}]
                    }]}
                ]
            }))
            .unwrap()
        };
        for (agent, req_id) in [("agent-a", "read-same-lane"), ("agent-b", "read-sibling")] {
            let response = provider
                .handle_messages_inner(result_request(), scope(agent, req_id), false)
                .await;
            assert_eq!(response.status(), StatusCode::OK);
            let _ = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
        }

        let captured = server.await.unwrap();
        assert!(captured[0].0.contains("authorization: Bearer read-route-a"));
        assert!(captured[1].0.contains("authorization: Bearer read-route-b"));
        assert_ne!(
            captured_header(&captured[0].0, "session_id"),
            captured_header(&captured[1].0, "session_id")
        );
        let same_lane_output = captured[1].1["input"][1]["output"]
            .as_str()
            .expect("same-lane tool output");
        assert!(same_lane_output.contains("Proxy Read offset note:"));
        assert!(same_lane_output.contains("1300007"));
        assert!(same_lane_output.contains("/tmp/route-read"));
        let sibling_output = captured[2].1["input"][1]["output"]
            .as_str()
            .expect("sibling tool output");
        assert!(!sibling_output.contains("Proxy Read offset note:"));
        for (headers, _) in captured {
            for raw in [
                "raw-shared-session",
                "raw-context-must-not-own-read-state",
                "agent-a",
                "agent-b",
            ] {
                assert!(!headers.contains(raw));
            }
        }
    }

    #[tokio::test]
    async fn cancellation_during_auth_refresh_aborts_reserved_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let session_id = "cancel-during-auth-refresh";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));

        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_address = upstream_listener.local_addr().unwrap();
        let oauth_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let oauth_address = oauth_listener.local_addr().unwrap();
        let auth_manager = CodexAuthManager::new_for_test(
            file_store(),
            format!("http://{oauth_address}/oauth/token"),
        );
        auth_manager.set_test_auth(auth::token_store::StoredAuth {
            access: "cancel-a".into(),
            refresh: "cancel-refresh".into(),
            expires: u64::MAX,
            account_id: Some("cancel-account".into()),
        });
        let client = CodexHttpClient::new_for_test_with_auth_manager(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            auth_manager,
            format!("http://{upstream_address}/v1/responses"),
            1_000,
            1_000,
            0,
        )
        .with_test_transport(config::CodexTransport::Http);
        let provider = CodexProvider::with_client(client);

        let upstream = tokio::spawn(async move {
            let (mut socket, _) = upstream_listener.accept().await.unwrap();
            let _ = read_http_request(&mut socket).await;
            socket
                .write_all(
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 5\r\nconnection: close\r\n\r\nstale",
                )
                .await
                .unwrap();
        });
        let (refresh_started_tx, refresh_started_rx) = tokio::sync::oneshot::channel();
        let (release_refresh_tx, release_refresh_rx) = tokio::sync::oneshot::channel();
        let oauth = tokio::spawn(async move {
            let (mut socket, _) = oauth_listener.accept().await.unwrap();
            let request = read_http_request(&mut socket).await;
            assert!(String::from_utf8_lossy(&request).contains("refresh_token=cancel-refresh"));
            refresh_started_tx.send(()).unwrap();
            let _ = release_refresh_rx.await;
            let body = br#"{"access_token":"cancel-b","refresh_token":"cancel-refresh-b","expires_in":3600}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(head.as_bytes()).await;
            let _ = socket.write_all(body).await;
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":false,
            "messages":[{"role":"user","content":"cancel while refreshing"}]
        }))
        .unwrap();
        let response_task = tokio::spawn(async move {
            provider
                .handle_messages(
                    request,
                    messages_test_context("cancel-refresh", Some(session_id)),
                )
                .await
        });

        tokio::time::timeout(std::time::Duration::from_secs(2), refresh_started_rx)
            .await
            .expect("auth refresh did not start")
            .expect("auth refresh acknowledgement dropped");
        assert!(!response_task.is_finished());
        response_task.abort();
        assert!(response_task.await.unwrap_err().is_cancelled());
        let _ = release_refresh_tx.send(());
        upstream.await.unwrap();
        oauth.await.unwrap();

        assert!(
            !continuation::has_continuation_owner_state_for_tests(&owner),
            "canceled request left stale continuation owner state"
        );
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

    #[tokio::test]
    async fn live_stream_response_emits_downstream_frames_before_terminal_event() {
        use http_body_util::BodyExt as _;

        let request_body = live_test_request("incremental HTTP");
        let client = authenticated_live_test_client("http://127.0.0.1:1/responses".to_string());
        let owner = ConversationIdentity::Main("incremental-http".to_string());
        let continuation = ContinuationReservation::for_owner_turn(Some(&owner), Some(1));
        let (route, continuation) =
            bind_live_test_route(client.as_ref(), &owner, &continuation).await;
        let ctx = RequestContext {
            req_id: "incremental-http".to_string(),
            session_id: None,
            session_seq: None,
            provider: "codex".to_string(),
            traffic: None,
            monitor: None,
        };
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        tx.send(Ok(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_up"}
        })))
        .await
        .unwrap();
        tx.send(Ok(serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "first"
        })))
        .await
        .unwrap();

        let (rx, _) = websocket::CodexWebSocketEventStream::pending(rx);
        let mut conversation_admission = None;
        let response = match live_stream_response_once(
            rx,
            client,
            route,
            Arc::new(AuthRejectionBudget::default()),
            "msg_test".to_string(),
            "claude-opus-4-8",
            ctx,
            continuation,
            request_body,
            None,
            None,
            &mut conversation_admission,
        )
        .await
        {
            LiveStreamStart::Response(response) => response,
            LiveStreamStart::Retry { error, .. } | LiveStreamStart::Unauthorized(error) => {
                panic!("unexpected retry: {error}")
            }
        };
        let mut body = response.into_body();
        let first = tokio::time::timeout(Duration::from_millis(200), body.frame())
            .await
            .expect("initial downstream frame must be available immediately")
            .unwrap()
            .unwrap()
            .into_data()
            .unwrap();
        let first = String::from_utf8(first.to_vec()).unwrap();
        assert!(first.contains("event: message_start"));
        assert!(first.contains("event: content_block_start"));
        assert!(first.contains("event: content_block_delta"));

        tx.send(Ok(serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "second"
        })))
        .await
        .unwrap();
        let second = tokio::time::timeout(Duration::from_millis(200), body.frame())
            .await
            .expect("text delta must arrive before the terminal event")
            .unwrap()
            .unwrap()
            .into_data()
            .unwrap();
        assert!(
            String::from_utf8(second.to_vec())
                .unwrap()
                .contains("event: content_block_delta")
        );

        for payload in [
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message"}
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "incomplete_details": null,
                    "usage": {"input_tokens": 1, "output_tokens": 1}
                }
            }),
        ] {
            tx.send(Ok(payload)).await.unwrap();
        }
        drop(tx);
        while let Some(frame) = body.frame().await {
            frame.unwrap();
        }
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
    fn exhausted_http_stream_error_is_not_retried_by_provider() {
        let err = client::CodexError {
            status: 503,
            message: "Codex HTTP stream exhausted its retry budget".to_string(),
            detail: Some("http_response_body".to_string()),
            retry_after: None,
            origin: client::CodexErrorOrigin::Http,
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

    async fn run_live_failure_case(
        session_id: &str,
        event: serde_json::Value,
        expected_attempts: usize,
    ) -> StatusCode {
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = live_test_request("one");
        let continuation = continuation_candidate_for_owner(Some(&owner), &request, true);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for _ in 0..expected_attempts {
                let (socket, _) = listener.accept().await.unwrap();
                let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
                let _ = next_live_websocket_request(&mut websocket).await;
                emit_live_event(&mut websocket, &event).await;
                drop(websocket);
            }
        });
        let client = authenticated_live_test_client(format!("http://{addr}/responses"));
        let lane = RequestScope::from_conversation_identity(
            Some(owner.clone()),
            RequestPurpose::Conversation,
        )
        .provider_lane(LaneDomain::CodexConversation);
        let route = client
            .bind_conversation_route(lane, ProtocolLane::ResponsesFull)
            .await
            .unwrap();
        let continuation = continuation.bind_route(&route);
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            live_stream_response(
                client,
                route,
                "message".to_string(),
                &request.model,
                live_test_context(session_id),
                request.clone(),
                continuation.clone(),
                None,
                None,
                None,
            ),
        )
        .await
        .expect("live failure case timed out");
        tokio::time::timeout(std::time::Duration::from_secs(5), server)
            .await
            .expect("live failure server timed out")
            .expect("live failure server failed");

        assert!(!continuation::is_current_turn_for_owner(&continuation));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        response.status()
    }

    #[tokio::test]
    #[allow(clippy::result_large_err)]
    async fn live_response_created_then_401_rebuilds_before_downstream_publication() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let session_id = "live-presemantic-auth-rebuild";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);

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
        client
            .auth_manager()
            .set_test_auth(auth::token_store::StoredAuth {
                access: "live-a".into(),
                refresh: "refresh-a".into(),
                expires: u64::MAX,
                account_id: Some("live-account".into()),
            });
        let provider = CodexProvider::with_client(client);
        let server_client = provider.client.clone();
        let server = tokio::spawn(async move {
            let (route_a_socket, _) = listener.accept().await.unwrap();
            let mut route_a = tokio_tungstenite::accept_hdr_async(
                route_a_socket,
                |request: &http::Request<()>, response| {
                    assert_eq!(
                        request.headers()[http::header::AUTHORIZATION],
                        "Bearer live-a"
                    );
                    Ok(response)
                },
            )
            .await
            .unwrap();
            let request_a = next_live_websocket_request(&mut route_a).await;
            emit_live_event(
                &mut route_a,
                &serde_json::json!({
                    "type": "response.created",
                    "response": {"id": "private-route-a"}
                }),
            )
            .await;
            server_client
                .auth_manager()
                .set_test_auth(auth::token_store::StoredAuth {
                    access: "live-b".into(),
                    refresh: "refresh-b".into(),
                    expires: u64::MAX,
                    account_id: Some("live-account".into()),
                });
            emit_live_event(
                &mut route_a,
                &serde_json::json!({
                    "type": "response.failed",
                    "status_code": 401,
                    "response": {
                        "error": {"status": 401, "message": "private route A rejection"}
                    }
                }),
            )
            .await;
            drop(route_a);

            let (route_b_socket, _) = listener.accept().await.unwrap();
            let mut route_b = tokio_tungstenite::accept_hdr_async(
                route_b_socket,
                |request: &http::Request<()>, response| {
                    assert_eq!(
                        request.headers()[http::header::AUTHORIZATION],
                        "Bearer live-b"
                    );
                    Ok(response)
                },
            )
            .await
            .unwrap();
            let request_b = next_live_websocket_request(&mut route_b).await;
            assert_ne!(request_a["prompt_cache_key"], request_b["prompt_cache_key"]);
            assert_eq!(request_a["input"], request_b["input"]);
            assert!(request_b.get("previous_response_id").is_none());
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
                    "delta":"public-route-b-only"
                }),
                serde_json::json!({
                    "type":"response.output_item.done",
                    "output_index":0,
                    "item":{"type":"message","id":"msg_b"}
                }),
                serde_json::json!({
                    "type":"response.completed",
                    "response":{"id":"resp_b","status":"completed","usage":{}}
                }),
            ] {
                emit_live_event(&mut route_b, &event).await;
            }
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(125), listener.accept())
                    .await
                    .is_err(),
                "live request sent route C"
            );
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":true,
            "messages":[{"role":"user","content":"route rebuild"}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                request,
                messages_test_context("live-presemantic", Some(session_id)),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("public-route-b-only"));
        assert!(!body.contains("private route A rejection"));
        assert!(!body.contains("authentication_error"));
        server.await.unwrap();
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        continuation::clear_continuation_for_owner(Some(&owner));
    }

    #[tokio::test]
    async fn live_completed_tool_then_401_emits_sanitized_auth_error_without_replay() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let session_id = "live-postsemantic-auth-error";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);

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
        client
            .auth_manager()
            .set_test_auth(auth::token_store::StoredAuth {
                access: "semantic-a".into(),
                refresh: "refresh-a".into(),
                expires: u64::MAX,
                account_id: Some("semantic-account".into()),
            });
        let provider = CodexProvider::with_client(client);
        let server_client = provider.client.clone();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            let _ = next_live_websocket_request(&mut websocket).await;
            for event in [
                serde_json::json!({"type":"response.created","response":{"id":"semantic-a"}}),
                serde_json::json!({
                    "type":"response.output_item.added",
                    "output_index":0,
                    "item":{
                        "type":"function_call",
                        "id":"tool-item",
                        "call_id":"call-1",
                        "name":"Bash",
                        "arguments":""
                    }
                }),
                serde_json::json!({
                    "type":"response.output_item.done",
                    "output_index":0,
                    "item":{
                        "type":"function_call",
                        "id":"tool-item",
                        "call_id":"call-1",
                        "name":"Bash",
                        "arguments":"{\"command\":\"pwd\"}"
                    }
                }),
            ] {
                emit_live_event(&mut websocket, &event).await;
            }
            server_client
                .auth_manager()
                .set_test_auth(auth::token_store::StoredAuth {
                    access: "semantic-b".into(),
                    refresh: "refresh-b".into(),
                    expires: u64::MAX,
                    account_id: Some("semantic-account".into()),
                });
            emit_live_event(
                &mut websocket,
                &serde_json::json!({
                    "type":"response.failed",
                    "status_code":401,
                    "response":{"error":{"status":401,"message":"private semantic rejection"}}
                }),
            )
            .await;
            drop(websocket);
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(150), listener.accept())
                    .await
                    .is_err(),
                "post-semantic 401 replayed the request"
            );
        });
        let request: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model":"gpt-5.4",
            "max_tokens":256,
            "stream":true,
            "messages":[{"role":"user","content":"run a tool"}],
            "tools":[{"name":"Bash","input_schema":{"type":"object"}}]
        }))
        .unwrap();
        let response = provider
            .handle_messages(
                request,
                messages_test_context("live-postsemantic", Some(session_id)),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("tool_use"));
        assert!(body.contains("authentication_error"));
        assert!(body.contains("Authentication failed"));
        assert!(!body.contains("private semantic rejection"));
        assert!(!body.contains("event: message_stop"));
        server.await.unwrap();
        assert!(!continuation::has_continuation_for_owner_for_tests(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        continuation::clear_continuation_for_owner(Some(&owner));
    }

    #[tokio::test]
    async fn dropping_live_stream_during_retry_backoff_aborts_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let session_id = "live-retry-backoff-cleanup";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = live_test_request("one");
        let continuation = continuation_candidate_for_owner(Some(&owner), &request, true);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (event_sent_tx, event_sent_rx) = tokio::sync::oneshot::channel();
        let (socket_closed_tx, socket_closed_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            let _ = next_live_websocket_request(&mut websocket).await;
            emit_live_event(
                &mut websocket,
                &serde_json::json!({
                    "type": "codex.rate_limits",
                    "rate_limits": {"allowed": false, "limit_reached": true}
                }),
            )
            .await;
            event_sent_tx.send(()).unwrap();
            drop(websocket);
            socket_closed_tx.send(()).unwrap();
        });
        let client = authenticated_live_test_client(format!("http://{addr}/responses"));
        let (route, continuation) =
            bind_live_test_route(client.as_ref(), &owner, &continuation).await;
        let task_request = request.clone();
        let task_continuation = continuation.clone();
        let response_task = tokio::spawn(async move {
            let model = task_request.model.clone();
            live_stream_response(
                client,
                route,
                "message".to_string(),
                &model,
                live_test_context(session_id),
                task_request,
                task_continuation,
                None,
                None,
                None,
            )
            .await
        });

        event_sent_rx.await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), socket_closed_rx)
            .await
            .expect("retry handoff did not close the abandoned attempt socket")
            .expect("retry handoff socket-close sender dropped");
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
        assert!(
            !response_task.is_finished(),
            "logical request must still be waiting in retry backoff"
        );
        response_task.abort();
        assert!(response_task.await.unwrap_err().is_cancelled());

        assert!(!continuation::is_current_turn_for_owner(&continuation));
        server.await.unwrap();
        websocket::invalidate_codex_websocket_pool_owner(&owner);
    }

    #[tokio::test]
    async fn dropping_live_response_body_after_first_chunk_aborts_request_state() {
        let _compaction_guard = compaction::lock_compaction_registry_for_async_tests().await;
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let session_id = "live-response-body-drop-cleanup";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = live_test_request("one");
        let continuation = continuation_candidate_for_owner(Some(&owner), &request, true);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (socket_closed_tx, socket_closed_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut websocket = tokio_tungstenite::accept_async(socket).await.unwrap();
            let _ = next_live_websocket_request(&mut websocket).await;
            emit_live_event(
                &mut websocket,
                &serde_json::json!({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "message", "id": "msg-partial"}
                }),
            )
            .await;
            emit_live_event(
                &mut websocket,
                &serde_json::json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "delta": "partial"
                }),
            )
            .await;
            while websocket.next().await.is_some() {}
            socket_closed_tx.send(()).unwrap();
        });
        let client = authenticated_live_test_client_with_transport(
            format!("http://{addr}/responses"),
            config::CodexTransport::Auto,
        );
        let (route, continuation) =
            bind_live_test_route(client.as_ref(), &owner, &continuation).await;
        let compaction_permit = reserve_compaction_start(route.lane()).unwrap();
        let compaction_build =
            begin_compaction_for_route(&compaction_permit, &route, "gpt-5.6-sol").unwrap();
        assert!(store_compaction_for_route(
            &compaction_build,
            native_compaction("drop-body-native")
        ));
        assert!(activate_compaction_for_route(
            &compaction_build,
            &compaction_output()
        ));
        let dropped_replay =
            apply_compaction_replay_for_route(&route, &compaction_replay_request()).unwrap();
        let peer_replay =
            apply_compaction_replay_for_route(&route, &compaction_replay_request()).unwrap();

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            live_stream_response(
                client,
                route.clone(),
                "message".to_string(),
                &request.model,
                live_test_context(session_id),
                request.clone(),
                continuation.clone(),
                Some(dropped_replay.lease),
                None,
                None,
            ),
        )
        .await
        .expect("live response did not publish the first chunk");
        let mut body = response.into_body();
        tokio::time::timeout(std::time::Duration::from_secs(5), body.frame())
            .await
            .expect("first downstream chunk timed out")
            .expect("live response body ended before the first chunk")
            .expect("first downstream chunk failed");
        drop(body);

        tokio::time::timeout(std::time::Duration::from_secs(5), socket_closed_rx)
            .await
            .expect("dropping the downstream body did not close the upstream socket")
            .expect("socket-close acknowledgement sender dropped");
        assert!(!continuation::is_current_turn_for_owner(&continuation));
        assert!(compaction::has_bound_compaction_for_tests(&route));
        assert!(activate_compaction_for_route(&peer_replay.lease, &[]));
        assert!(compaction::has_bound_compaction_for_tests(&route));
        server.await.unwrap();
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        compaction::clear_compactions_for_lane(route.lane().unwrap());
        drop(compaction_permit);
    }

    #[tokio::test]
    async fn queued_terminal_loses_to_downstream_cancellation_before_publication() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let session_id = "queued-terminal-body-drop";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = live_test_request("one");
        let reserved = continuation_candidate_for_owner(Some(&owner), &request, true);
        let client = authenticated_live_test_client("http://127.0.0.1:1/responses".to_string());
        let (route, reserved) = bind_live_test_route(client.as_ref(), &owner, &reserved).await;
        let (upstream_tx, upstream_rx) = tokio::sync::mpsc::channel(1);
        let (upstream_events, socket_id_publisher) =
            websocket::CodexWebSocketEventStream::pending(upstream_rx);
        socket_id_publisher.publish(Some(41));
        let translator = LiveStreamTranslator::with_stable_read_lane(
            "message".to_string(),
            request.model.clone(),
            count_translated_tokens(&request),
            None,
        );
        let response = remaining_live_stream_response(
            upstream_events,
            translator,
            b"first".to_vec(),
            live_test_context(session_id),
            reserved.clone(),
            request.clone(),
            Vec::new(),
            None,
            None,
            client,
            route,
            Arc::new(AuthRejectionBudget::default()),
            None,
        );
        let mut body = response.into_body();
        let first = body
            .frame()
            .await
            .expect("first downstream chunk")
            .expect("first downstream chunk must succeed");
        assert_eq!(first.into_data().unwrap(), Bytes::from_static(b"first"));

        upstream_tx
            .try_send(Ok(serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_queued",
                    "status": "completed",
                    "output": []
                }
            })))
            .expect("provider-facing terminal channel must accept the terminal");
        drop(body);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while continuation::is_current_turn_for_owner(&reserved) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("downstream cancellation did not settle the reserved turn");
        assert!(!continuation::has_continuation_for_owner_for_tests(&owner));
        assert_eq!(
            record_continuation_for_owner(&reserved, &request, Some("resp_queued"), Some(41), &[],),
            ContinuationPublication::Rejected
        );
        let next_request = live_test_request("two");
        let next = continuation_candidate_for_owner(Some(&owner), &next_request, true);
        assert!(next.candidate().previous_response_id.is_none());
        assert_eq!(next.candidate().input_delta_count, next_request.input.len());
        abort_continuation_for_owner(&next);
    }

    #[tokio::test]
    async fn blocked_older_compaction_cannot_publish_after_newer_completion() {
        let _compaction_guard = compaction::lock_compaction_registry_for_async_tests().await;
        let session_id = "blocked-older-compaction";
        let owner = ConversationIdentity::Main(session_id.to_string());
        let lane = RequestScope::from_conversation_identity(
            Some(owner.clone()),
            RequestPurpose::Conversation,
        )
        .provider_lane(LaneDomain::CodexConversation);
        let client = authenticated_live_test_client("http://127.0.0.1:1/responses".to_string());
        let route = client
            .bind_conversation_route(lane, ProtocolLane::ResponsesFull)
            .await
            .unwrap();
        let older_permit = reserve_compaction_start(lane).unwrap();
        let older = begin_compaction_for_route(&older_permit, &route, "gpt-5.6-sol").unwrap();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let older_task = tokio::spawn(async move {
            started_tx.send(()).unwrap();
            release_rx.await.unwrap();
            let stored = store_compaction_for_route(&older, native_compaction("older"));
            let activated = activate_compaction_for_route(&older, &compaction_output());
            drop(older_permit);
            (stored, activated)
        });
        started_rx.await.unwrap();

        let newer_permit = reserve_compaction_start(lane).unwrap();
        let newer = begin_compaction_for_route(&newer_permit, &route, "gpt-5.6-sol").unwrap();
        assert!(store_compaction_for_route(
            &newer,
            native_compaction("newer")
        ));
        assert!(activate_compaction_for_route(&newer, &compaction_output()));
        release_tx.send(()).unwrap();
        assert_eq!(older_task.await.unwrap(), (false, false));

        let replay =
            apply_compaction_replay_for_route(&route, &compaction_replay_request()).unwrap();
        assert!(replay.request.input.iter().any(|item| {
            matches!(item, translate::request::ResponsesInputItem::Compaction { encrypted_content } if encrypted_content == "newer")
        }));
        assert!(activate_compaction_for_route(&replay.lease, &[]));
        compaction::clear_compactions_for_lane(lane.unwrap());
        drop(newer_permit);
    }

    #[tokio::test]
    async fn canceling_blocked_compaction_drops_its_permit_and_build() {
        let _compaction_guard = compaction::lock_compaction_registry_for_async_tests().await;
        let session_id = "cancel-blocked-compaction";
        let owner = ConversationIdentity::Main(session_id.to_string());
        let lane =
            RequestScope::from_conversation_identity(Some(owner), RequestPurpose::Conversation)
                .provider_lane(LaneDomain::CodexConversation);
        let client = authenticated_live_test_client("http://127.0.0.1:1/responses".to_string());
        let route = client
            .bind_conversation_route(lane, ProtocolLane::ResponsesFull)
            .await
            .unwrap();
        let permit = reserve_compaction_start(lane).unwrap();
        let build = begin_compaction_for_route(&permit, &route, "gpt-5.6-sol").unwrap();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (_release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            started_tx.send(()).unwrap();
            let _ = release_rx.await;
            drop((build, permit));
        });
        started_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());

        assert!(!compaction::has_bound_compaction_for_tests(&route));
        assert!(!compaction::has_bound_lane_metadata_for_tests(
            lane.unwrap()
        ));
        let fresh = reserve_compaction_start(lane).unwrap();
        let fresh_build = begin_compaction_for_route(&fresh, &route, "gpt-5.6-sol").unwrap();
        drop((fresh_build, fresh));
    }

    #[tokio::test]
    async fn replay_is_canonicalized_before_continuation_evaluation_without_new_generation() {
        let _compaction_guard = compaction::lock_compaction_registry_for_async_tests().await;
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let session_id = "replay-full-context-generation";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        let lane = RequestScope::from_conversation_identity(
            Some(owner.clone()),
            RequestPurpose::Conversation,
        )
        .provider_lane(LaneDomain::CodexConversation);
        let client = authenticated_live_test_client("http://127.0.0.1:1/responses".to_string());
        let route = client
            .bind_conversation_route(lane, ProtocolLane::ResponsesFull)
            .await
            .unwrap();

        let mut first_request = compaction_replay_request();
        first_request.input.pop();
        let first =
            continuation_candidate_for_owner(Some(&owner), &first_request, true).bind_route(&route);
        record_continuation_for_owner(
            &first,
            &first_request,
            Some("resp-before-replay"),
            Some(41),
            &[],
        );
        let next_request = compaction_replay_request();
        let reserved = reserve_continuation_for_owner(Some(&owner), true);
        let generation = reserved.turn_id();

        let permit = reserve_compaction_start(lane).unwrap();
        let build = begin_compaction_for_route(&permit, &route, "gpt-5.6-sol").unwrap();
        assert!(store_compaction_for_route(
            &build,
            native_compaction("native")
        ));
        assert!(activate_compaction_for_route(&build, &compaction_output()));
        let replay = apply_compaction_replay_for_route(&route, &next_request).unwrap();
        let evaluated = reserved.evaluate(&replay.request).bind_route(&route);

        assert_eq!(evaluated.turn_id(), generation);
        assert!(evaluated.candidate().previous_response_id.is_none());
        assert!(evaluated.candidate().input_delta.is_none());
        assert_eq!(
            evaluated.candidate_cause(),
            Some(CodexRecoveryCause::NotAppendOnly)
        );
        assert!(activate_compaction_for_route(&replay.lease, &[]));
        abort_continuation_for_owner(&evaluated);
        compaction::clear_compactions_for_lane(lane.unwrap());
        drop(permit);
    }

    #[tokio::test]
    async fn route_rebound_with_same_permit_requires_no_newer_generation() {
        let _compaction_guard = compaction::lock_compaction_registry_for_async_tests().await;
        let lane = RequestScope::from_conversation_identity(
            Some(ConversationIdentity::Main(
                "route-rebound-permit".to_string(),
            )),
            RequestPurpose::Conversation,
        )
        .provider_lane(LaneDomain::CodexConversation);
        let client_a = authenticated_live_test_client("http://127.0.0.1:1/responses".to_string());
        let client_b = CodexHttpClient::new_for_test(
            reqwest::Client::builder().no_proxy().build().unwrap(),
            "http://127.0.0.1:1/responses".to_string(),
            1_000,
            1_000,
            0,
        );
        client_b
            .auth_manager()
            .set_test_auth(auth::token_store::StoredAuth {
                access: "rotated".to_string(),
                refresh: String::new(),
                expires: u64::MAX,
                account_id: Some("acct".to_string()),
            });
        let route_a = client_a
            .bind_conversation_route(lane, ProtocolLane::ResponsesFull)
            .await
            .unwrap();
        let route_b = client_b
            .bind_conversation_route(lane, ProtocolLane::ResponsesFull)
            .await
            .unwrap();

        let permit = reserve_compaction_start(lane).unwrap();
        let first = begin_compaction_for_route(&permit, &route_a, "gpt-5.6-sol").unwrap();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let rebound_route = route_b.clone();
        let rebound = tokio::spawn(async move {
            started_tx.send(()).unwrap();
            release_rx.await.unwrap();
            begin_compaction_for_route(&permit, &rebound_route, "gpt-5.6-sol")
                .map(|lease| (lease, permit))
        });
        started_rx.await.unwrap();
        release_tx.send(()).unwrap();
        let (rebound_build, permit) = rebound.await.unwrap().unwrap();
        assert!(compaction::has_bound_compaction_for_tests(&route_b));
        drop(first);
        assert!(compaction::has_bound_compaction_for_tests(&route_b));
        drop((rebound_build, permit));

        let older = reserve_compaction_start(lane).unwrap();
        let older_build = begin_compaction_for_route(&older, &route_a, "gpt-5.6-sol").unwrap();
        let (blocked_tx, blocked_rx) = tokio::sync::oneshot::channel();
        let (resume_tx, resume_rx) = tokio::sync::oneshot::channel();
        let stale_route = route_b.clone();
        let stale_rebound = tokio::spawn(async move {
            blocked_tx.send(()).unwrap();
            resume_rx.await.unwrap();
            begin_compaction_for_route(&older, &stale_route, "gpt-5.6-sol")
        });
        blocked_rx.await.unwrap();
        let newer = reserve_compaction_start(lane).unwrap();
        let newer_build = begin_compaction_for_route(&newer, &route_b, "gpt-5.6-sol").unwrap();
        resume_tx.send(()).unwrap();
        assert!(stale_rebound.await.unwrap().is_none());
        assert!(compaction::has_bound_compaction_for_tests(&route_b));
        drop((older_build, newer_build, newer));
    }

    #[tokio::test]
    async fn stale_request_cleanup_preserves_newer_continuation_turn() {
        let _compaction_guard = compaction::lock_compaction_registry_for_async_tests().await;
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let session_id = "stale-live-request-cleanup";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        let request = live_test_request("one");
        let lane = RequestScope::from_conversation_identity(
            Some(owner.clone()),
            RequestPurpose::Conversation,
        )
        .provider_lane(LaneDomain::CodexConversation);
        let client = authenticated_live_test_client("http://127.0.0.1:1/responses".to_string());
        let route = client
            .bind_conversation_route(lane, ProtocolLane::ResponsesFull)
            .await
            .unwrap();

        let stale_continuation = continuation_candidate_for_owner(Some(&owner), &request, true);
        let stale_permit = reserve_compaction_start(lane).unwrap();
        let stale_lease =
            begin_compaction_for_route(&stale_permit, &route, &request.model).unwrap();
        assert!(store_compaction_for_route(
            &stale_lease,
            vec![translate::request::ResponsesInputItem::Compaction {
                encrypted_content: "stale-native-history".to_string(),
            }],
        ));
        let stale_cleanup =
            LiveRequestStateCleanup::new(stale_continuation, Some(stale_lease), Some(stale_permit));

        let newer_continuation = continuation_candidate_for_owner(Some(&owner), &request, true);
        let newer_permit = reserve_compaction_start(lane).unwrap();
        let newer_lease =
            begin_compaction_for_route(&newer_permit, &route, &request.model).unwrap();
        assert!(store_compaction_for_route(
            &newer_lease,
            vec![translate::request::ResponsesInputItem::Compaction {
                encrypted_content: "newer-native-history".to_string(),
            }],
        ));
        drop(stale_cleanup);

        assert!(continuation::is_current_turn_for_owner(&newer_continuation));
        let summary: Vec<translate::request::ResponsesInputItem> =
            serde_json::from_value(serde_json::json!([{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "newer portable summary with enough detail for safe activation"
                }]
            }]))
            .unwrap();
        assert!(activate_compaction_for_route(&newer_lease, &summary));
        abort_request_state(&newer_continuation, Some(&newer_lease));
        compaction::clear_compactions_for_lane(lane.unwrap());
    }

    #[tokio::test]
    async fn retry_exhaustion_aborts_live_request_state_after_eleven_attempts() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let status = run_live_failure_case(
            "live-retry-exhaustion-cleanup",
            serde_json::json!({
                "type": "codex.rate_limits",
                "rate_limits": {
                    "allowed": false,
                    "limit_reached": true,
                    "primary": {"reset_after_seconds": 0}
                }
            }),
            11,
        )
        .await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn excessive_retry_after_aborts_live_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let status = run_live_failure_case(
            "live-excessive-retry-after-cleanup",
            serde_json::json!({
                "type": "codex.rate_limits",
                "rate_limits": {
                    "allowed": false,
                    "limit_reached": true,
                    "primary": {"reset_after_seconds": 31}
                }
            }),
            1,
        )
        .await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn nonretryable_live_error_aborts_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let status = run_live_failure_case(
            "live-nonretryable-cleanup",
            serde_json::json!({
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {"message": "invalid request"}
                }
            }),
            1,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn cancellation_while_replacement_startup_is_blocked_aborts_request_state() {
        let _registry_guard = continuation::lock_continuation_registry_for_async_tests().await;
        let _pool_guard = websocket::lock_codex_websocket_pool_for_tests().await;
        let session_id = "live-blocked-replacement-cleanup";
        let owner = ConversationIdentity::Main(session_id.to_string());
        continuation::clear_continuation_for_owner(Some(&owner));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
        let request = live_test_request("one");
        let continuation = continuation_candidate_for_owner(Some(&owner), &request, true);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (replacement_accepted_tx, replacement_accepted_rx) = tokio::sync::oneshot::channel();
        let (release_replacement_tx, release_replacement_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (first_socket, _) = listener.accept().await.unwrap();
            let mut first_websocket = tokio_tungstenite::accept_async(first_socket).await.unwrap();
            let _ = next_live_websocket_request(&mut first_websocket).await;
            emit_live_event(
                &mut first_websocket,
                &serde_json::json!({
                    "type": "codex.rate_limits",
                    "rate_limits": {
                        "allowed": false,
                        "limit_reached": true,
                        "primary": {"reset_after_seconds": 0}
                    }
                }),
            )
            .await;
            drop(first_websocket);

            let (_replacement_socket, _) = listener.accept().await.unwrap();
            replacement_accepted_tx.send(()).unwrap();
            let _ = release_replacement_rx.await;
        });
        let client = authenticated_live_test_client(format!("http://{addr}/responses"));
        let (route, continuation) =
            bind_live_test_route(client.as_ref(), &owner, &continuation).await;
        let task_request = request.clone();
        let task_continuation = continuation.clone();
        let response_task = tokio::spawn(async move {
            let model = task_request.model.clone();
            live_stream_response(
                client,
                route,
                "message".to_string(),
                &model,
                live_test_context(session_id),
                task_request,
                task_continuation,
                None,
                None,
                None,
            )
            .await
        });

        tokio::time::timeout(std::time::Duration::from_secs(2), replacement_accepted_rx)
            .await
            .expect("replacement startup did not reach the blocked handshake")
            .expect("replacement startup acknowledgement sender dropped");
        response_task.abort();
        assert!(response_task.await.unwrap_err().is_cancelled());
        let _ = release_replacement_tx.send(());
        server.await.unwrap();

        assert!(!continuation::is_current_turn_for_owner(&continuation));
        websocket::invalidate_codex_websocket_pool_owner(&owner);
    }
}
