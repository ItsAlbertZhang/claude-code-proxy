use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::monitor::{
    CodexAppendOnlyDiagnostics, CodexAppendOnlyOutcome, CodexPoolResolution,
    CodexPreviousIdOutcome, CodexRecoveryCause, CodexSocketValidationFailure, MonitorHandle,
};
use crate::request_identity::ConversationIdentity;

use super::state::{CodexBoundRoute, SocketPoolKey};
use super::translate::request::{ResponsesInputItem, ResponsesRequest};

const TTL_MS: u64 = 30 * 60 * 1000;
const MAX_STATES: usize = 10_000;
const MAX_OWNER_RETAINED_BYTES: u64 = 2_000_000;
const MAX_TOTAL_RETAINED_BYTES: u64 = 20_000_000;

#[derive(Clone)]
struct ContinuationState {
    response_id: String,
    socket_id: u64,
    route_key: Option<SocketPoolKey>,
    prompt_signature: String,
    transcript: Vec<ResponsesInputItem>,
    retained_bytes: u64,
    updated_at: u64,
}

struct OwnerState {
    current_turn: u64,
    cleanup_epoch: u64,
    bound_route_key: Option<SocketPoolKey>,
    continuation: Option<ContinuationState>,
    updated_at: u64,
}

#[derive(Default)]
struct ContinuationRegistry {
    owners: HashMap<ConversationIdentity, OwnerState>,
    total_retained_bytes: u64,
}

static REGISTRY: Mutex<Option<ContinuationRegistry>> = Mutex::new(None);
static NEXT_TURN_ID: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
static TEST_REGISTRY_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[cfg(test)]
pub(crate) fn lock_continuation_registry_for_tests() -> tokio::sync::MutexGuard<'static, ()> {
    TEST_REGISTRY_LOCK.blocking_lock()
}

#[cfg(test)]
pub(crate) async fn lock_continuation_registry_for_async_tests()
-> tokio::sync::MutexGuard<'static, ()> {
    TEST_REGISTRY_LOCK.lock().await
}

#[derive(Clone)]
struct ReservationSnapshot {
    state: Option<ContinuationState>,
    superseded_turn: bool,
    reserved_at: u64,
}

const TURN_OPEN: u8 = 0;
const TURN_PUBLISHED: u8 = 1;
const TURN_CANCELLED: u8 = 2;

#[derive(Default)]
struct TurnOutcomeFence {
    state: AtomicU8,
}

impl TurnOutcomeFence {
    fn claim_publication(&self) -> bool {
        self.state
            .compare_exchange(
                TURN_OPEN,
                TURN_PUBLISHED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    fn cancel(&self) -> bool {
        self.state
            .compare_exchange(
                TURN_OPEN,
                TURN_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    #[cfg(test)]
    fn state(&self) -> u8 {
        self.state.load(Ordering::Acquire)
    }
}

#[derive(Clone)]
pub struct ContinuationCandidate {
    pub turn_id: Option<u64>,
    pub previous_response_id: Option<String>,
    pub input_delta: Option<Vec<ResponsesInputItem>>,
    pub input_delta_count: usize,
    pub disabled_reason: Option<String>,
}

struct PreviousIdMetrics {
    monitor: MonitorHandle,
    request_id: String,
    had_candidate_at_attachment: AtomicBool,
    settled: AtomicBool,
}

impl PreviousIdMetrics {
    fn set_had_candidate(&self, had_candidate: bool) {
        self.had_candidate_at_attachment
            .store(had_candidate, Ordering::Release);
    }

    fn settle(&self, outcome: CodexPreviousIdOutcome) {
        if self
            .settled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.monitor
                .codex_previous_id_settled(&self.request_id, outcome);
        }
    }

    fn record_previous_id_cause(&self, cause: CodexRecoveryCause) {
        self.monitor
            .codex_previous_id_cause(&self.request_id, cause);
    }

    fn record_socket_cause(&self, cause: CodexRecoveryCause) {
        self.monitor.codex_socket_cause(&self.request_id, cause);
    }

    fn record_append_only(&self, diagnostics: CodexAppendOnlyDiagnostics) {
        self.monitor
            .codex_append_only(&self.request_id, diagnostics);
    }

    fn record_pool_resolution(&self, resolution: CodexPoolResolution) {
        self.monitor
            .codex_pool_resolution(&self.request_id, resolution);
    }

    fn record_socket_validation(
        &self,
        failure: Option<CodexSocketValidationFailure>,
        elapsed_ms: u32,
        required_origin: bool,
    ) {
        self.monitor.codex_socket_validation(
            &self.request_id,
            failure,
            elapsed_ms,
            required_origin,
        );
    }
}

#[derive(Clone)]
pub(crate) struct ContinuationReservation {
    candidate: ContinuationCandidate,
    owner: Option<ConversationIdentity>,
    origin_socket_id: Option<u64>,
    route_key: Option<SocketPoolKey>,
    cleanup_epoch: Option<u64>,
    candidate_cause: Option<CodexRecoveryCause>,
    append_only: Option<CodexAppendOnlyDiagnostics>,
    previous_id_metrics: Option<Arc<PreviousIdMetrics>>,
    snapshot: Option<ReservationSnapshot>,
    turn_outcome: Option<Arc<TurnOutcomeFence>>,
    detached: bool,
}

impl ContinuationReservation {
    pub(crate) fn new(
        candidate: ContinuationCandidate,
        owner: Option<ConversationIdentity>,
        origin_socket_id: Option<u64>,
    ) -> Self {
        let turn_outcome = (owner.is_some() && candidate.turn_id.is_some())
            .then(|| Arc::new(TurnOutcomeFence::default()));
        Self {
            candidate,
            owner,
            origin_socket_id,
            route_key: None,
            cleanup_epoch: None,
            candidate_cause: None,
            append_only: None,
            previous_id_metrics: None,
            snapshot: None,
            turn_outcome,
            detached: false,
        }
    }

    pub(crate) fn detached(input_count: usize, reason: &str) -> Self {
        let mut reservation = Self::new(
            ContinuationCandidate {
                turn_id: None,
                previous_response_id: None,
                input_delta: None,
                input_delta_count: input_count,
                disabled_reason: Some(reason.to_string()),
            },
            None,
            None,
        );
        reservation.detached = true;
        reservation
    }

    fn with_candidate_cause(mut self, cause: CodexRecoveryCause) -> Self {
        if let Some(metrics) = self.previous_id_metrics.as_ref() {
            metrics.record_previous_id_cause(cause);
        }
        self.candidate_cause = Some(cause);
        self
    }

    fn with_append_only(mut self, diagnostics: CodexAppendOnlyDiagnostics) -> Self {
        if let Some(metrics) = self.previous_id_metrics.as_ref() {
            metrics.record_append_only(diagnostics);
        }
        self.append_only = Some(diagnostics);
        self
    }

    pub(crate) fn with_previous_id_metrics(
        mut self,
        monitor: Option<MonitorHandle>,
        request_id: &str,
    ) -> Self {
        if self.turn_id().is_some()
            && let Some(monitor) = monitor
        {
            monitor.codex_previous_id_pending(request_id);
            let metrics = Arc::new(PreviousIdMetrics {
                monitor,
                request_id: request_id.to_string(),
                had_candidate_at_attachment: AtomicBool::new(
                    self.candidate.previous_response_id.is_some(),
                ),
                settled: AtomicBool::new(false),
            });
            if let Some(cause) = self.candidate_cause {
                metrics.record_previous_id_cause(cause);
            }
            if let Some(append_only) = self.append_only {
                metrics.record_append_only(append_only);
            }
            self.previous_id_metrics = Some(metrics);
        }
        self
    }

    pub(crate) fn settle_previous_id_completed(&self) {
        let Some(metrics) = self.previous_id_metrics.as_ref() else {
            return;
        };
        let outcome = match (
            metrics.had_candidate_at_attachment.load(Ordering::Acquire),
            self.candidate.previous_response_id.is_some(),
        ) {
            (false, _) => CodexPreviousIdOutcome::NoCandidate,
            (true, true) => CodexPreviousIdOutcome::Hit,
            (true, false) => CodexPreviousIdOutcome::Fallback,
        };
        metrics.settle(outcome);
    }

    pub(crate) fn from_public_candidate(candidate: &ContinuationCandidate) -> Self {
        Self::new(candidate.clone(), None, None)
    }

    pub(crate) fn for_owner_turn(
        owner: Option<&ConversationIdentity>,
        turn_id: Option<u64>,
    ) -> Self {
        Self::new(
            ContinuationCandidate {
                turn_id,
                previous_response_id: None,
                input_delta: None,
                input_delta_count: 0,
                disabled_reason: None,
            },
            owner.cloned(),
            None,
        )
    }

    pub(crate) fn candidate(&self) -> &ContinuationCandidate {
        &self.candidate
    }

    pub(crate) fn owner(&self) -> Option<&ConversationIdentity> {
        self.owner.as_ref()
    }

    pub(crate) fn turn_id(&self) -> Option<u64> {
        self.candidate.turn_id
    }

    pub(crate) fn origin_socket_id(&self) -> Option<u64> {
        self.origin_socket_id
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn candidate_cause(&self) -> Option<CodexRecoveryCause> {
        self.candidate_cause
    }

    pub(crate) fn record_socket_cause(&self, cause: CodexRecoveryCause) {
        if let Some(metrics) = self.previous_id_metrics.as_ref() {
            metrics.record_socket_cause(cause);
        }
    }

    pub(crate) fn record_pool_resolution(&self, resolution: CodexPoolResolution) {
        if let Some(metrics) = self.previous_id_metrics.as_ref() {
            metrics.record_pool_resolution(resolution);
        }
    }

    pub(crate) fn record_socket_validation(
        &self,
        failure: Option<CodexSocketValidationFailure>,
        elapsed_ms: u32,
        required_origin: bool,
    ) {
        if let Some(metrics) = self.previous_id_metrics.as_ref() {
            metrics.record_socket_validation(failure, elapsed_ms, required_origin);
        }
    }

    fn record_previous_id_cause(&self, cause: CodexRecoveryCause) {
        if self.candidate.previous_response_id.is_some()
            && let Some(metrics) = self.previous_id_metrics.as_ref()
        {
            metrics.record_previous_id_cause(cause);
        }
    }

    pub(crate) fn route_key(&self) -> Option<SocketPoolKey> {
        self.route_key
    }

    #[cfg(test)]
    pub(crate) fn cleanup_epoch(&self) -> Option<u64> {
        self.cleanup_epoch
    }

    pub(crate) fn bind_route(&self, route: &CodexBoundRoute) -> Self {
        if self.detached {
            return self.clone();
        }
        let mut bound = self.clone();
        let route_key = route.socket_pool_key();
        let previous_route_matches = bound.candidate.previous_response_id.is_none()
            || (route_key.is_some() && bound.route_key == route_key);
        if !previous_route_matches {
            bound.record_previous_id_cause(CodexRecoveryCause::RouteChanged);
            bound.candidate_cause = Some(CodexRecoveryCause::RouteChanged);
            bound.candidate.previous_response_id = None;
            bound.candidate.input_delta = None;
            bound.candidate.disabled_reason = Some("route_changed".to_string());
            bound.origin_socket_id = None;
        }
        bound.route_key = route_key;
        bound.cleanup_epoch = None;

        let (Some(owner), Some(turn_id), Some(route_key)) =
            (bound.owner.as_ref(), bound.candidate.turn_id, route_key)
        else {
            return bound;
        };
        let mut guard = REGISTRY.lock().unwrap();
        let Some(owner_state) = guard
            .as_mut()
            .and_then(|registry| registry.owners.get_mut(owner))
        else {
            return bound;
        };
        if owner_state.current_turn != turn_id {
            bound.record_previous_id_cause(CodexRecoveryCause::SupersededTurn);
            bound.candidate_cause = Some(CodexRecoveryCause::SupersededTurn);
            bound.candidate.previous_response_id = None;
            bound.candidate.input_delta = None;
            bound.candidate.disabled_reason = Some("superseded_turn".to_string());
            bound.origin_socket_id = None;
            return bound;
        }
        owner_state.cleanup_epoch = owner_state
            .cleanup_epoch
            .checked_add(1)
            .expect("Codex continuation cleanup epoch exhausted");
        owner_state.bound_route_key = Some(route_key);
        owner_state.updated_at = now_ms();
        bound.cleanup_epoch = Some(owner_state.cleanup_epoch);
        bound
    }

    fn with_previous_route_key(mut self, route_key: Option<SocketPoolKey>) -> Self {
        self.route_key = route_key;
        self
    }

    pub(crate) fn evaluate(&self, body: &ResponsesRequest) -> Self {
        let Some(snapshot) = self.snapshot.clone() else {
            let mut evaluated = self.clone();
            evaluated.candidate.input_delta_count = body.input.len();
            return evaluated;
        };
        continuation_candidate_from_state(self, body, snapshot, true)
    }

    pub(crate) fn evaluate_hidden_compaction(&self, body: &ResponsesRequest) -> Self {
        let Some(snapshot) = self.snapshot.clone() else {
            return self.evaluate(body);
        };
        continuation_candidate_from_state(self, body, snapshot, false)
    }

    fn claim_publication(&self) -> bool {
        self.turn_outcome
            .as_ref()
            .is_some_and(|outcome| outcome.claim_publication())
    }

    fn cancel(&self) -> bool {
        self.turn_outcome
            .as_ref()
            .is_some_and(|outcome| outcome.cancel())
    }

    pub(crate) fn into_candidate(self) -> ContinuationCandidate {
        self.candidate
    }

    pub(crate) fn full_context_retry(&self, cause: CodexRecoveryCause) -> Self {
        self.record_previous_id_cause(cause);
        let mut retry = self.clone();
        retry.candidate.previous_response_id = None;
        retry.candidate.input_delta = None;
        retry.candidate.disabled_reason = Some("full_context_retry".to_string());
        retry.origin_socket_id = None;
        retry.snapshot = None;
        retry.candidate_cause = Some(cause);
        retry
    }
}

fn next_monotonic_nonzero(sequence: &AtomicU64, label: &str) -> u64 {
    let previous = sequence
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .unwrap_or_else(|_| panic!("{label} sequence exhausted"));
    previous + 1
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[deprecated(note = "use the owner-aware provider flow for typed conversation ownership")]
pub fn continuation_candidate(
    session_id: Option<&str>,
    body: &ResponsesRequest,
    enabled: bool,
) -> ContinuationCandidate {
    let owner = session_id.map(|session_id| ConversationIdentity::Main(session_id.to_owned()));
    reserve_continuation_inner(owner.as_ref(), enabled, "missing_session")
        .evaluate(body)
        .into_candidate()
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn continuation_candidate_for_owner(
    owner: Option<&ConversationIdentity>,
    body: &ResponsesRequest,
    enabled: bool,
) -> ContinuationReservation {
    reserve_continuation_for_owner(owner, enabled).evaluate(body)
}

pub(crate) fn reserve_continuation_for_owner(
    owner: Option<&ConversationIdentity>,
    enabled: bool,
) -> ContinuationReservation {
    reserve_continuation_inner(owner, enabled, "missing_identity")
}

fn reserve_continuation_inner(
    owner: Option<&ConversationIdentity>,
    enabled: bool,
    missing_owner_reason: &str,
) -> ContinuationReservation {
    if !enabled {
        return ContinuationReservation::new(
            ContinuationCandidate {
                turn_id: None,
                previous_response_id: None,
                input_delta: None,
                input_delta_count: 0,
                disabled_reason: Some("disabled".to_string()),
            },
            owner.cloned(),
            None,
        );
    }

    let Some(owner) = owner else {
        return ContinuationReservation::new(
            ContinuationCandidate {
                turn_id: None,
                previous_response_id: None,
                input_delta: None,
                input_delta_count: 0,
                disabled_reason: Some(missing_owner_reason.to_string()),
            },
            None,
            None,
        );
    };

    let turn_id = next_monotonic_nonzero(&NEXT_TURN_ID, "Codex continuation generation");
    let now = now_ms();
    let (state, superseded_turn) = {
        let mut guard = REGISTRY.lock().unwrap();
        let registry = guard.get_or_insert_with(ContinuationRegistry::default);
        let existing = registry.owners.remove(owner);
        let superseded_turn = existing.is_some();
        let state = existing.and_then(|owner_state| {
            registry.total_retained_bytes = registry
                .total_retained_bytes
                .saturating_sub(owner_retained_size(owner, &owner_state));
            owner_state.continuation
        });
        let owner_state = OwnerState {
            current_turn: turn_id,
            cleanup_epoch: 0,
            bound_route_key: None,
            continuation: None,
            updated_at: now,
        };
        registry.total_retained_bytes = registry
            .total_retained_bytes
            .saturating_add(owner_retained_size(owner, &owner_state));
        registry.owners.insert(owner.clone(), owner_state);
        evict_oldest(registry);
        (state, superseded_turn)
    };

    let mut reservation = ContinuationReservation::new(
        ContinuationCandidate {
            turn_id: Some(turn_id),
            previous_response_id: None,
            input_delta: None,
            input_delta_count: 0,
            disabled_reason: None,
        },
        Some(owner.clone()),
        None,
    );
    reservation.snapshot = Some(ReservationSnapshot {
        state,
        superseded_turn,
        reserved_at: now,
    });
    reservation
}

fn continuation_candidate_from_state(
    reservation: &ContinuationReservation,
    body: &ResponsesRequest,
    snapshot: ReservationSnapshot,
    require_prompt_signature: bool,
) -> ContinuationReservation {
    debug_assert!(reservation.owner().is_some());
    let turn_id = reservation
        .turn_id()
        .expect("continuation snapshot must retain its generation");
    let mut evaluated = reservation.clone();
    evaluated.candidate = ContinuationCandidate {
        turn_id: Some(turn_id),
        previous_response_id: None,
        input_delta: None,
        input_delta_count: body.input.len(),
        disabled_reason: None,
    };
    evaluated.origin_socket_id = None;
    evaluated.route_key = None;
    evaluated.cleanup_epoch = None;
    evaluated.candidate_cause = None;
    evaluated.append_only = None;

    let state = match snapshot.state {
        Some(state) if snapshot.reserved_at.saturating_sub(state.updated_at) <= TTL_MS => state,
        Some(_) | None => {
            evaluated.candidate.disabled_reason = Some(if snapshot.superseded_turn {
                "superseded_turn".to_string()
            } else {
                "missing_state".to_string()
            });
            return evaluated.with_candidate_cause(if snapshot.superseded_turn {
                CodexRecoveryCause::SupersededTurn
            } else {
                CodexRecoveryCause::MissingState
            });
        }
    };

    let previous_route_key = state.route_key;
    let signature = prompt_signature(body);
    if require_prompt_signature && signature != state.prompt_signature {
        evaluated.candidate.disabled_reason = Some("prompt_changed".to_string());
        return evaluated.with_candidate_cause(CodexRecoveryCause::PromptChanged);
    }

    let (suffix, append_only) = match input_suffix_after_prefix(&body.input, &state.transcript) {
        InputPrefixComparison::Appended {
            suffix,
            diagnostics,
        } => (suffix, diagnostics),
        InputPrefixComparison::NoDelta { diagnostics } => {
            evaluated.candidate.input_delta_count = 0;
            evaluated.candidate.disabled_reason = Some("empty_delta".to_string());
            return evaluated
                .with_candidate_cause(CodexRecoveryCause::EmptyDelta)
                .with_append_only(diagnostics);
        }
        InputPrefixComparison::RetainedLonger { diagnostics }
        | InputPrefixComparison::FirstMismatch { diagnostics } => {
            evaluated.candidate.disabled_reason = Some("not_append_only".to_string());
            return evaluated
                .with_candidate_cause(CodexRecoveryCause::NotAppendOnly)
                .with_append_only(diagnostics);
        }
    };

    evaluated.candidate.previous_response_id = Some(state.response_id);
    evaluated.candidate.input_delta_count = suffix.len();
    evaluated.candidate.input_delta = Some(suffix);
    evaluated.origin_socket_id = Some(state.socket_id);
    if let Some(metrics) = evaluated.previous_id_metrics.as_ref() {
        metrics.set_had_candidate(true);
    }
    evaluated
        .with_previous_route_key(previous_route_key)
        .with_append_only(append_only)
}

#[deprecated(note = "recording without typed socket provenance is not reusable")]
pub fn record_continuation(
    session_id: Option<&str>,
    turn_id: Option<u64>,
    request_body: &ResponsesRequest,
    response_id: Option<&str>,
    output_items: &[ResponsesInputItem],
) {
    let owner = session_id.map(|session_id| ConversationIdentity::Main(session_id.to_owned()));
    let reservation = ContinuationReservation::new(
        ContinuationCandidate {
            turn_id,
            previous_response_id: None,
            input_delta: None,
            input_delta_count: request_body.input.len(),
            disabled_reason: Some("legacy_recording_without_socket".to_string()),
        },
        owner,
        None,
    );
    record_continuation_for_owner(&reservation, request_body, response_id, None, output_items);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContinuationPublication {
    Published,
    Rejected,
}

pub(crate) fn record_continuation_for_owner(
    reservation: &ContinuationReservation,
    request_body: &ResponsesRequest,
    response_id: Option<&str>,
    socket_id: Option<u64>,
    output_items: &[ResponsesInputItem],
) -> ContinuationPublication {
    let owner = match (reservation.owner(), reservation.turn_id()) {
        (Some(owner), Some(_)) => owner,
        _ => return ContinuationPublication::Rejected,
    };

    let (response_id, socket_id) = match (response_id, socket_id) {
        (Some(response_id), Some(socket_id)) if socket_id != 0 => {
            (response_id.to_string(), socket_id)
        }
        _ => {
            abort_continuation_for_owner(reservation);
            return ContinuationPublication::Rejected;
        }
    };
    let mut transcript: Vec<ResponsesInputItem> = request_body.input.clone();
    transcript.extend_from_slice(output_items);

    let transcript_json = serde_json::to_string(&transcript).unwrap_or_default();
    let prompt_signature = prompt_signature(request_body);
    let retained_bytes = continuation_retained_size(
        owner,
        &response_id,
        reservation.route_key.as_ref(),
        &prompt_signature,
        &transcript,
        transcript_json.len(),
    );

    if retained_bytes > MAX_OWNER_RETAINED_BYTES {
        abort_continuation_for_owner(reservation);
        return ContinuationPublication::Rejected;
    }

    let state = ContinuationState {
        response_id,
        socket_id,
        route_key: reservation.route_key,
        prompt_signature,
        transcript,
        retained_bytes,
        updated_at: now_ms(),
    };

    let mut guard = REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return ContinuationPublication::Rejected;
    };
    let Some(owner_state) = registry.owners.get_mut(owner) else {
        return ContinuationPublication::Rejected;
    };
    if !reservation_matches_owner_state(reservation, owner_state)
        || !reservation.claim_publication()
    {
        return ContinuationPublication::Rejected;
    }
    let previous_size = owner_retained_size(owner, owner_state);
    owner_state.continuation = Some(state);
    let next_size = owner_retained_size(owner, owner_state);
    registry.total_retained_bytes = registry
        .total_retained_bytes
        .saturating_sub(previous_size)
        .saturating_add(next_size);
    evict_oldest(registry);
    ContinuationPublication::Published
}

#[deprecated(note = "use the owner-aware provider flow for typed conversation ownership")]
pub fn abort_continuation(session_id: Option<&str>, turn_id: Option<u64>) {
    let owner = session_id.map(|session_id| ConversationIdentity::Main(session_id.to_owned()));
    abort_continuation_inner(owner.as_ref(), turn_id);
}

pub(crate) fn abort_continuation_for_owner(reservation: &ContinuationReservation) {
    let Some(owner) = reservation.owner() else {
        return;
    };
    let mut guard = REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return;
    };
    if registry
        .owners
        .get(owner)
        .is_some_and(|state| reservation_matches_owner_state(reservation, state))
        && reservation.cancel()
    {
        remove_owner(registry, owner);
    }
}

fn abort_continuation_inner(owner: Option<&ConversationIdentity>, turn_id: Option<u64>) {
    let (Some(owner), Some(turn_id)) = (owner, turn_id) else {
        return;
    };
    let mut guard = REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return;
    };
    if registry
        .owners
        .get(owner)
        .is_some_and(|state| state.current_turn == turn_id)
    {
        remove_owner(registry, owner);
    }
}

#[deprecated(note = "use the owner-aware provider flow for typed conversation ownership")]
pub fn if_current_turn<T>(
    session_id: Option<&str>,
    turn_id: Option<u64>,
    action: impl FnOnce() -> T,
) -> Option<T> {
    let owner = session_id.map(|session_id| ConversationIdentity::Main(session_id.to_owned()));
    if_current_turn_inner(owner.as_ref(), turn_id, action)
}

pub(crate) fn if_current_turn_for_owner<T>(
    reservation: &ContinuationReservation,
    action: impl FnOnce() -> T,
) -> Option<T> {
    let owner = reservation.owner()?;
    let guard = REGISTRY.lock().unwrap();
    let current = guard
        .as_ref()
        .and_then(|registry| registry.owners.get(owner))
        .is_some_and(|state| reservation_matches_owner_state(reservation, state));
    current.then(action)
}

fn if_current_turn_inner<T>(
    owner: Option<&ConversationIdentity>,
    turn_id: Option<u64>,
    action: impl FnOnce() -> T,
) -> Option<T> {
    let (Some(owner), Some(turn_id)) = (owner, turn_id) else {
        return None;
    };
    let guard = REGISTRY.lock().unwrap();
    let current = guard
        .as_ref()
        .and_then(|registry| registry.owners.get(owner))
        .is_some_and(|state| state.current_turn == turn_id);
    current.then(action)
}

#[deprecated(note = "use the owner-aware provider flow for typed conversation ownership")]
pub fn with_current_turn(
    session_id: Option<&str>,
    turn_id: Option<u64>,
    action: impl FnOnce(),
) -> bool {
    let owner = session_id.map(|session_id| ConversationIdentity::Main(session_id.to_owned()));
    if_current_turn_inner(owner.as_ref(), turn_id, action).is_some()
}

pub(crate) fn with_current_turn_for_owner(
    reservation: &ContinuationReservation,
    action: impl FnOnce(),
) -> bool {
    if_current_turn_for_owner(reservation, action).is_some()
}

#[deprecated(note = "use the owner-aware provider flow for typed conversation ownership")]
pub fn is_current_turn(session_id: Option<&str>, turn_id: Option<u64>) -> bool {
    let owner = session_id.map(|session_id| ConversationIdentity::Main(session_id.to_owned()));
    is_current_turn_inner(owner.as_ref(), turn_id)
}

#[allow(dead_code)]
pub(crate) fn is_current_turn_for_owner(reservation: &ContinuationReservation) -> bool {
    let Some(owner) = reservation.owner() else {
        return false;
    };
    let guard = REGISTRY.lock().unwrap();
    guard
        .as_ref()
        .and_then(|registry| registry.owners.get(owner))
        .is_some_and(|state| reservation_matches_owner_state(reservation, state))
}

fn is_current_turn_inner(owner: Option<&ConversationIdentity>, turn_id: Option<u64>) -> bool {
    let (Some(owner), Some(turn_id)) = (owner, turn_id) else {
        return false;
    };
    let guard = REGISTRY.lock().unwrap();
    guard
        .as_ref()
        .and_then(|registry| registry.owners.get(owner))
        .is_some_and(|state| state.current_turn == turn_id)
}

#[deprecated(note = "use the owner-aware provider flow for typed conversation ownership")]
pub fn clear_continuation(session_id: Option<&str>) {
    let owner = session_id.map(|session_id| ConversationIdentity::Main(session_id.to_owned()));
    clear_continuation_for_owner(owner.as_ref());
}

pub(crate) fn clear_continuation_for_owner(owner: Option<&ConversationIdentity>) {
    let Some(owner) = owner else {
        return;
    };
    let mut guard = REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return;
    };
    remove_owner(registry, owner);
}

#[deprecated(note = "use the owner-aware test helper for typed conversation ownership")]
pub fn has_continuation_for_tests(session_id: &str) -> bool {
    let owner = ConversationIdentity::Main(session_id.to_owned());
    has_continuation_for_owner_for_tests(&owner)
}

pub(crate) fn has_continuation_for_owner_for_tests(owner: &ConversationIdentity) -> bool {
    let guard = REGISTRY.lock().unwrap();
    guard
        .as_ref()
        .and_then(|registry| registry.owners.get(owner))
        .is_some_and(|state| state.continuation.is_some())
}

#[cfg(test)]
pub(crate) fn has_continuation_owner_state_for_tests(owner: &ConversationIdentity) -> bool {
    REGISTRY
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|registry| registry.owners.contains_key(owner))
}

pub fn clear_all_continuations_for_tests() {
    let mut guard = REGISTRY.lock().unwrap();
    *guard = None;
}

fn reservation_matches_owner_state(
    reservation: &ContinuationReservation,
    state: &OwnerState,
) -> bool {
    if reservation.turn_id() != Some(state.current_turn) {
        return false;
    }
    match (reservation.cleanup_epoch, reservation.route_key) {
        (Some(epoch), Some(route_key)) => {
            state.cleanup_epoch == epoch && state.bound_route_key == Some(route_key)
        }
        (None, None) => true,
        _ => false,
    }
}

fn owner_base_size(owner: &ConversationIdentity) -> u64 {
    let identity_bytes = match owner {
        ConversationIdentity::Main(session) => session.len(),
        ConversationIdentity::Agent(session, agent) => session.len().saturating_add(agent.len()),
    };
    std::mem::size_of::<ConversationIdentity>()
        .saturating_add(std::mem::size_of::<OwnerState>())
        .saturating_add(3 * std::mem::size_of::<usize>())
        .saturating_add(identity_bytes) as u64
}

fn continuation_retained_size(
    owner: &ConversationIdentity,
    response_id: &str,
    route_key: Option<&SocketPoolKey>,
    prompt_signature: &str,
    transcript: &[ResponsesInputItem],
    serialized_transcript_bytes: usize,
) -> u64 {
    let route_bytes = route_key.map_or(0, |_| std::mem::size_of::<SocketPoolKey>());
    owner_base_size(owner).saturating_add(
        std::mem::size_of::<ContinuationState>()
            .saturating_add(response_id.len())
            .saturating_add(route_bytes)
            .saturating_add(prompt_signature.len())
            .saturating_add(serialized_transcript_bytes)
            .saturating_add(
                transcript
                    .len()
                    .saturating_mul(std::mem::size_of::<ResponsesInputItem>()),
            ) as u64,
    )
}

fn owner_retained_size(owner: &ConversationIdentity, state: &OwnerState) -> u64 {
    state
        .continuation
        .as_ref()
        .map_or_else(|| owner_base_size(owner), |state| state.retained_bytes)
}

fn remove_owner(registry: &mut ContinuationRegistry, owner: &ConversationIdentity) {
    if let Some(state) = registry.owners.remove(owner) {
        registry.total_retained_bytes = registry
            .total_retained_bytes
            .saturating_sub(owner_retained_size(owner, &state));
    }
}

enum InputPrefixComparison {
    Appended {
        suffix: Vec<ResponsesInputItem>,
        diagnostics: CodexAppendOnlyDiagnostics,
    },
    NoDelta {
        diagnostics: CodexAppendOnlyDiagnostics,
    },
    RetainedLonger {
        diagnostics: CodexAppendOnlyDiagnostics,
    },
    FirstMismatch {
        diagnostics: CodexAppendOnlyDiagnostics,
    },
}

fn bounded_item_count(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

fn input_suffix_after_prefix(
    input: &[ResponsesInputItem],
    prefix: &[ResponsesInputItem],
) -> InputPrefixComparison {
    let incoming_items = bounded_item_count(input.len());
    let retained_items = bounded_item_count(prefix.len());
    if prefix.len() > input.len() {
        return InputPrefixComparison::RetainedLonger {
            diagnostics: CodexAppendOnlyDiagnostics {
                outcome: CodexAppendOnlyOutcome::RetainedLonger,
                incoming_items,
                retained_items,
                delta_items: 0,
                first_mismatch_index: None,
            },
        };
    }
    for i in 0..prefix.len() {
        let a = serde_json::to_value(&input[i]).unwrap_or_default();
        let b = serde_json::to_value(&prefix[i]).unwrap_or_default();
        if a != b {
            return InputPrefixComparison::FirstMismatch {
                diagnostics: CodexAppendOnlyDiagnostics {
                    outcome: CodexAppendOnlyOutcome::FirstMismatch,
                    incoming_items,
                    retained_items,
                    delta_items: bounded_item_count(input.len().saturating_sub(prefix.len())),
                    first_mismatch_index: Some(bounded_item_count(i)),
                },
            };
        }
    }
    let suffix = input[prefix.len()..].to_vec();
    let diagnostics = CodexAppendOnlyDiagnostics {
        outcome: if suffix.is_empty() {
            CodexAppendOnlyOutcome::NoDelta
        } else {
            CodexAppendOnlyOutcome::Appended
        },
        incoming_items,
        retained_items,
        delta_items: bounded_item_count(suffix.len()),
        first_mismatch_index: None,
    };
    if suffix.is_empty() {
        InputPrefixComparison::NoDelta { diagnostics }
    } else {
        InputPrefixComparison::Appended {
            suffix,
            diagnostics,
        }
    }
}

fn prompt_signature(body: &ResponsesRequest) -> String {
    let value = serde_json::to_value(body).unwrap_or_default();
    let obj = match value.as_object() {
        Some(o) => o,
        None => return String::new(),
    };
    let mut entries: Vec<(&String, &serde_json::Value)> = obj
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "input" | "prompt_cache_key"))
        .collect();
    entries.sort_by_key(|(a, _)| *a);
    let mut sig = String::from("{");
    for (i, (key, val)) in entries.iter().enumerate() {
        if i > 0 {
            sig.push(',');
        }
        sig.push_str(&format!("\"{}\":{}", key, stable_json(val)));
    }
    sig.push('}');
    sig
}

fn stable_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(stable_json).collect();
            format!("[{}]", items.join(","))
        }
        serde_json::Value::Object(obj) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = obj.iter().collect();
            entries.sort_by_key(|(a, _)| *a);
            let items: Vec<String> = entries
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap_or_default(),
                        stable_json(v)
                    )
                })
                .collect();
            format!("{{{}}}", items.join(","))
        }
    }
}

fn evict_oldest(registry: &mut ContinuationRegistry) {
    while registry.owners.len() > MAX_STATES
        || registry.total_retained_bytes > MAX_TOTAL_RETAINED_BYTES
    {
        let owner = registry
            .owners
            .iter()
            .min_by_key(|(_, state)| state.updated_at)
            .map(|(owner, _)| owner.clone());
        let Some(owner) = owner else {
            break;
        };
        remove_owner(registry, &owner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::{EndpointKind, MonitorHandle};
    use crate::request_identity::{LaneDomain, RequestPurpose, RequestScope};
    use serde_json::json;

    fn lock_registry() -> tokio::sync::MutexGuard<'static, ()> {
        let guard = lock_continuation_registry_for_tests();
        clear_all_continuations_for_tests();
        guard
    }

    fn main_owner(session_id: &str) -> ConversationIdentity {
        ConversationIdentity::Main(session_id.to_string())
    }

    fn agent_owner(session_id: &str, agent_id: &str) -> ConversationIdentity {
        ConversationIdentity::Agent(session_id.to_string(), agent_id.to_string())
    }

    fn route(owner: &ConversationIdentity, access: &str) -> CodexBoundRoute {
        let lane = RequestScope::from_conversation_identity(
            Some(owner.clone()),
            RequestPurpose::Conversation,
        )
        .provider_lane(LaneDomain::CodexConversation);
        CodexBoundRoute::new(
            super::super::auth::token_store::StoredAuth {
                access: access.to_string(),
                refresh: String::new(),
                expires: u64::MAX,
                account_id: Some("account-a".to_string()),
            },
            "https://example.test/backend-api/codex/responses",
            super::super::state::ProtocolLane::ResponsesFull,
            lane,
        )
        .unwrap()
    }

    fn input(text: &str) -> ResponsesInputItem {
        ResponsesInputItem::Message {
            role: "user".to_string(),
            content: vec![
                super::super::translate::request::ResponsesContentPart::InputText {
                    text: text.to_string(),
                },
            ],
        }
    }

    fn request_with_input(
        input: Vec<ResponsesInputItem>,
        extra: Option<serde_json::Value>,
    ) -> ResponsesRequest {
        let mut fields = serde_json::Map::new();
        fields.insert("model".into(), json!("gpt-5.5"));
        fields.insert("input".into(), json!(input));
        fields.insert("store".into(), json!(false));
        fields.insert("stream".into(), json!(true));
        fields.insert("text".into(), json!({"verbosity": "low"}));
        fields.insert("parallel_tool_calls".into(), json!(true));
        if let Some(extras) = extra
            && let Some(obj) = extras.as_object()
        {
            for (key, value) in obj {
                fields.insert(key.clone(), value.clone());
            }
        }
        serde_json::from_value(serde_json::Value::Object(fields)).unwrap()
    }

    fn start_and_record(
        owner: &ConversationIdentity,
        request: &ResponsesRequest,
        response_id: &str,
    ) {
        let reservation = continuation_candidate_for_owner(Some(owner), request, true);
        record_continuation_for_owner(&reservation, request, Some(response_id), Some(1), &[]);
    }

    #[test]
    fn prompt_signature_bytes_count_toward_owner_quota() {
        let _registry_guard = lock_registry();
        let owner = main_owner("oversized-prompt");
        let request = request_with_input(
            vec![input("tiny")],
            Some(json!({"instructions": "x".repeat(MAX_OWNER_RETAINED_BYTES as usize)})),
        );
        let reservation = continuation_candidate_for_owner(Some(&owner), &request, true);
        record_continuation_for_owner(&reservation, &request, Some("resp_oversized"), Some(1), &[]);
        assert!(!has_continuation_for_owner_for_tests(&owner));
    }

    #[test]
    #[allow(deprecated)]
    fn disabled_and_missing_identity_requests_are_stateless() {
        let _registry_guard = lock_registry();
        let request = request_with_input(vec![input("one")], None);
        let owner = main_owner("session-a");

        let disabled = continuation_candidate_for_owner(Some(&owner), &request, false);
        assert_eq!(disabled.owner(), Some(&owner));
        assert_eq!(disabled.turn_id(), None);
        assert_eq!(disabled.candidate().input_delta_count, request.input.len());
        assert_eq!(
            disabled.candidate().disabled_reason.as_deref(),
            Some("disabled")
        );

        let missing = continuation_candidate_for_owner(None, &request, true);
        assert_eq!(missing.owner(), None);
        assert_eq!(missing.turn_id(), None);
        assert_eq!(missing.candidate().input_delta_count, request.input.len());
        assert_eq!(
            missing.candidate().disabled_reason.as_deref(),
            Some("missing_identity")
        );

        let legacy_missing = continuation_candidate(None, &request, true);
        assert_eq!(
            legacy_missing.disabled_reason.as_deref(),
            Some("missing_session")
        );
    }

    #[test]
    fn append_only_comparison_preserves_shape_without_content() {
        let prefix = vec![input("one"), input("two")];
        let appended = vec![input("one"), input("two"), input("three")];
        match input_suffix_after_prefix(&appended, &prefix) {
            InputPrefixComparison::Appended {
                suffix,
                diagnostics,
            } => {
                assert_eq!(suffix.len(), 1);
                assert_eq!(
                    serde_json::to_value(&suffix[0]).unwrap(),
                    serde_json::to_value(input("three")).unwrap()
                );
                assert_eq!(diagnostics.outcome, CodexAppendOnlyOutcome::Appended);
                assert_eq!(diagnostics.incoming_items, 3);
                assert_eq!(diagnostics.retained_items, 2);
                assert_eq!(diagnostics.delta_items, 1);
                assert_eq!(diagnostics.first_mismatch_index, None);
            }
            _ => panic!("append-only extension should produce a suffix"),
        }

        match input_suffix_after_prefix(&prefix, &prefix) {
            InputPrefixComparison::NoDelta { diagnostics } => {
                assert_eq!(diagnostics.outcome, CodexAppendOnlyOutcome::NoDelta);
                assert_eq!(diagnostics.delta_items, 0);
            }
            _ => panic!("equal histories should report no delta"),
        }

        match input_suffix_after_prefix(&prefix[..1], &prefix) {
            InputPrefixComparison::RetainedLonger { diagnostics } => {
                assert_eq!(diagnostics.outcome, CodexAppendOnlyOutcome::RetainedLonger);
                assert_eq!(diagnostics.incoming_items, 1);
                assert_eq!(diagnostics.retained_items, 2);
                assert_eq!(diagnostics.first_mismatch_index, None);
            }
            _ => panic!("shorter incoming history should report retained longer"),
        }

        let mismatch = vec![input("one"), input("changed"), input("three")];
        match input_suffix_after_prefix(&mismatch, &prefix) {
            InputPrefixComparison::FirstMismatch { diagnostics } => {
                assert_eq!(diagnostics.outcome, CodexAppendOnlyOutcome::FirstMismatch);
                assert_eq!(diagnostics.first_mismatch_index, Some(1));
                assert_eq!(diagnostics.delta_items, 1);
            }
            _ => panic!("rewritten history should report its first mismatch"),
        }
    }

    #[test]
    fn previous_id_metrics_settle_once_across_full_context_retries() {
        let monitor = MonitorHandle::new(10);
        let owner = agent_owner("metrics-session", "metrics-agent");
        let reservation = |turn_id, previous_response_id: Option<&str>| {
            ContinuationReservation::new(
                ContinuationCandidate {
                    turn_id,
                    previous_response_id: previous_response_id.map(str::to_string),
                    input_delta: previous_response_id.map(|_| Vec::new()),
                    input_delta_count: 0,
                    disabled_reason: None,
                },
                Some(owner.clone()),
                Some(11),
            )
            .with_previous_id_metrics(Some(monitor.clone()), "req-metrics")
        };

        let no_candidate = reservation(Some(1), None);
        no_candidate.settle_previous_id_completed();
        no_candidate.settle_previous_id_completed();

        let hit = reservation(Some(2), Some("resp_hit"));
        let hit_retry = hit.full_context_retry(CodexRecoveryCause::ResponseStartTimeout);
        hit.settle_previous_id_completed();
        hit.settle_previous_id_completed();
        hit_retry.settle_previous_id_completed();

        let fallback = reservation(Some(3), Some("resp_fallback"));
        let fallback_retry =
            fallback.full_context_retry(CodexRecoveryCause::PreviousResponseMissing);
        fallback_retry.settle_previous_id_completed();
        fallback.settle_previous_id_completed();

        let ineligible = reservation(None, Some("resp_ineligible"));
        ineligible.settle_previous_id_completed();

        let metrics = monitor.snapshot().codex;
        assert_eq!(metrics.previous_id_no_candidates, 1);
        assert_eq!(metrics.previous_id_hits, 1);
        assert_eq!(metrics.previous_id_fallbacks, 1);
    }

    #[test]
    fn recovery_metrics_preserve_first_previous_cause_and_union_socket_causes() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("req-causes", None, None, EndpointKind::Messages);
        let reservation = ContinuationReservation::new(
            ContinuationCandidate {
                turn_id: Some(1),
                previous_response_id: Some("resp_1".to_string()),
                input_delta: Some(Vec::new()),
                input_delta_count: 0,
                disabled_reason: None,
            },
            Some(main_owner("metrics-causes")),
            Some(11),
        )
        .with_previous_id_metrics(Some(monitor.clone()), "req-causes");

        let _ = reservation.full_context_retry(CodexRecoveryCause::PreviousResponseMissing);
        let _ = reservation.full_context_retry(CodexRecoveryCause::AuthRejection);
        reservation.record_socket_cause(CodexRecoveryCause::TransportFailure);
        reservation.record_socket_cause(CodexRecoveryCause::OriginSocketMissing);
        reservation.record_socket_cause(CodexRecoveryCause::TransportFailure);

        let recovery = monitor.snapshot().active[0]
            .codex_diagnostics()
            .unwrap()
            .recovery;
        assert_eq!(
            recovery.previous_id_cause,
            Some(CodexRecoveryCause::PreviousResponseMissing)
        );
        assert_eq!(
            recovery.socket_causes.iter().collect::<Vec<_>>(),
            vec![
                CodexRecoveryCause::OriginSocketMissing,
                CodexRecoveryCause::TransportFailure,
            ]
        );

        monitor.request_started("req-initial-cause", None, None, EndpointKind::Messages);
        let initial = ContinuationReservation::new(
            ContinuationCandidate {
                turn_id: Some(2),
                previous_response_id: None,
                input_delta: None,
                input_delta_count: 0,
                disabled_reason: Some("missing_state".to_string()),
            },
            Some(main_owner("metrics-initial-cause")),
            None,
        )
        .with_candidate_cause(CodexRecoveryCause::MissingState)
        .with_previous_id_metrics(Some(monitor.clone()), "req-initial-cause");
        let _ = initial.full_context_retry(CodexRecoveryCause::AuthRejection);

        assert_eq!(
            monitor
                .snapshot()
                .active
                .iter()
                .find(|request| request.request_id == "req-initial-cause")
                .unwrap()
                .codex_diagnostics()
                .unwrap()
                .recovery
                .previous_id_cause,
            Some(CodexRecoveryCause::MissingState)
        );
    }

    #[test]
    fn sibling_agents_reserve_and_publish_independently() {
        let _registry_guard = lock_registry();
        let sibling_one = agent_owner("session-a", "agent-one");
        let sibling_two = agent_owner("session-a", "agent-two");
        let first_request = request_with_input(vec![input("one")], None);

        let first = continuation_candidate_for_owner(Some(&sibling_one), &first_request, true);
        let second = continuation_candidate_for_owner(Some(&sibling_two), &first_request, true);
        assert_ne!(first.turn_id(), second.turn_id());
        record_continuation_for_owner(&first, &first_request, Some("resp_one"), Some(11), &[]);
        record_continuation_for_owner(&second, &first_request, Some("resp_two"), Some(22), &[]);
        assert!(has_continuation_for_owner_for_tests(&sibling_one));
        assert!(has_continuation_for_owner_for_tests(&sibling_two));

        let next_request = request_with_input(vec![input("one"), input("two")], None);
        let first_next = continuation_candidate_for_owner(Some(&sibling_one), &next_request, true);
        let second_next = continuation_candidate_for_owner(Some(&sibling_two), &next_request, true);
        assert_eq!(
            first_next.candidate().previous_response_id.as_deref(),
            Some("resp_one")
        );
        assert_eq!(first_next.origin_socket_id(), Some(11));
        assert_eq!(
            second_next.candidate().previous_response_id.as_deref(),
            Some("resp_two")
        );
        assert_eq!(second_next.origin_socket_id(), Some(22));
    }

    #[test]
    fn different_owner_completion_order_cannot_interfere() {
        let _registry_guard = lock_registry();
        let main = main_owner("session-a");
        let agent = agent_owner("session-a", "agent-a");
        let request = request_with_input(vec![input("one")], None);
        let main_reservation = continuation_candidate_for_owner(Some(&main), &request, true);
        let agent_reservation = continuation_candidate_for_owner(Some(&agent), &request, true);

        record_continuation_for_owner(
            &agent_reservation,
            &request,
            Some("resp_agent"),
            Some(1),
            &[],
        );
        record_continuation_for_owner(&main_reservation, &request, Some("resp_main"), Some(1), &[]);
        abort_continuation_for_owner(&ContinuationReservation::new(
            ContinuationCandidate {
                turn_id: agent_reservation.turn_id(),
                previous_response_id: None,
                input_delta: None,
                input_delta_count: 0,
                disabled_reason: None,
            },
            Some(main.clone()),
            None,
        ));

        assert!(has_continuation_for_owner_for_tests(&main));
        assert!(has_continuation_for_owner_for_tests(&agent));
        let next = request_with_input(vec![input("one"), input("two")], None);
        assert_eq!(
            continuation_candidate_for_owner(Some(&agent), &next, true)
                .candidate()
                .previous_response_id
                .as_deref(),
            Some("resp_agent")
        );
    }

    #[test]
    fn missing_response_id_aborts_only_the_current_owner() {
        let _registry_guard = lock_registry();
        let owner = main_owner("session-a");
        let sibling = agent_owner("session-a", "agent-a");
        let request = request_with_input(vec![input("one")], None);
        start_and_record(&owner, &request, "resp_main");
        start_and_record(&sibling, &request, "resp_agent");

        let reservation = continuation_candidate_for_owner(Some(&owner), &request, true);
        record_continuation_for_owner(&reservation, &request, None, Some(1), &[]);

        assert!(!has_continuation_for_owner_for_tests(&owner));
        assert!(has_continuation_for_owner_for_tests(&sibling));
    }

    #[test]
    fn missing_socket_id_does_not_publish_reusable_state() {
        let _registry_guard = lock_registry();
        let owner = main_owner("session-no-socket");
        let request = request_with_input(vec![input("one")], None);
        let reservation = continuation_candidate_for_owner(Some(&owner), &request, true);

        record_continuation_for_owner(
            &reservation,
            &request,
            Some("resp_without_socket"),
            None,
            &[],
        );

        assert!(!has_continuation_for_owner_for_tests(&owner));
        let next = continuation_candidate_for_owner(Some(&owner), &request, true);
        assert_eq!(next.candidate().previous_response_id, None);
        assert_eq!(next.origin_socket_id(), None);
    }

    #[test]
    #[allow(deprecated)]
    fn legacy_recording_without_provenance_publishes_no_reusable_state() {
        let _registry_guard = lock_registry();
        let session_id = "legacy-no-provenance";
        let request = request_with_input(vec![input("one")], None);
        let candidate = continuation_candidate(Some(session_id), &request, true);

        record_continuation(
            Some(session_id),
            candidate.turn_id,
            &request,
            Some("resp_legacy"),
            &[],
        );

        assert!(!has_continuation_for_tests(session_id));
        let next = continuation_candidate(Some(session_id), &request, true);
        assert_eq!(next.previous_response_id, None);
    }

    #[test]
    fn same_owner_stale_turn_cannot_publish_clear_or_run_actions() {
        let _registry_guard = lock_registry();
        let owner = main_owner("session-a");
        let request = request_with_input(vec![input("one")], None);
        start_and_record(&owner, &request, "resp_1");

        let stale = continuation_candidate_for_owner(Some(&owner), &request, true);
        let current = continuation_candidate_for_owner(Some(&owner), &request, true);
        assert_eq!(
            current.candidate().disabled_reason.as_deref(),
            Some("superseded_turn")
        );
        record_continuation_for_owner(&stale, &request, Some("resp_stale"), Some(1), &[]);
        assert!(!has_continuation_for_owner_for_tests(&owner));
        record_continuation_for_owner(&current, &request, Some("resp_current"), Some(1), &[]);
        assert!(has_continuation_for_owner_for_tests(&owner));
        abort_continuation_for_owner(&stale);
        assert!(has_continuation_for_owner_for_tests(&owner));

        let mut ran = false;
        assert_eq!(if_current_turn_for_owner(&stale, || ran = true), None);
        assert!(!ran);
    }

    #[test]
    #[allow(deprecated)]
    fn missing_owner_or_turn_mutations_are_hard_noops() {
        let _registry_guard = lock_registry();
        let owner = main_owner("session-a");
        let request = request_with_input(vec![input("one")], None);
        start_and_record(&owner, &request, "resp_1");

        let missing_owner = ContinuationReservation::new(
            ContinuationCandidate {
                turn_id: Some(1),
                previous_response_id: None,
                input_delta: None,
                input_delta_count: 1,
                disabled_reason: None,
            },
            None,
            None,
        );
        let missing_turn = ContinuationReservation::new(
            ContinuationCandidate {
                turn_id: None,
                previous_response_id: None,
                input_delta: None,
                input_delta_count: 1,
                disabled_reason: None,
            },
            Some(owner.clone()),
            None,
        );
        record_continuation_for_owner(&missing_owner, &request, Some("ignored"), Some(1), &[]);
        record_continuation_for_owner(&missing_turn, &request, Some("ignored"), Some(1), &[]);
        abort_continuation_for_owner(&missing_owner);
        abort_continuation_for_owner(&missing_turn);
        clear_continuation_for_owner(None);
        assert!(has_continuation_for_owner_for_tests(&owner));

        let mut runs = 0;
        assert_eq!(
            if_current_turn_for_owner(&missing_owner, || runs += 1),
            None
        );
        assert_eq!(if_current_turn_for_owner(&missing_turn, || runs += 1), None);
        assert!(!with_current_turn_for_owner(&missing_owner, || runs += 1));
        assert!(!with_current_turn_for_owner(&missing_turn, || runs += 1));
        assert_eq!(if_current_turn(None, Some(1), || runs += 1), None);
        assert!(!with_current_turn(None, Some(1), || runs += 1));
        assert_eq!(runs, 0);
    }

    #[test]
    fn append_only_and_prompt_guards_stay_owner_scoped() {
        let _registry_guard = lock_registry();
        let owner = main_owner("session-a");
        let request = request_with_input(vec![input("one")], None);
        start_and_record(&owner, &request, "resp_1");

        let appended = request_with_input(vec![input("one"), input("two")], None);
        let reservation = continuation_candidate_for_owner(Some(&owner), &appended, true);
        assert_eq!(
            reservation.candidate().previous_response_id.as_deref(),
            Some("resp_1")
        );
        assert_eq!(reservation.candidate().input_delta_count, 1);

        let full_context =
            reservation.full_context_retry(CodexRecoveryCause::PreviousResponseMissing);
        assert_eq!(full_context.owner(), Some(&owner));
        assert_eq!(full_context.turn_id(), reservation.turn_id());
        assert_eq!(full_context.candidate().previous_response_id, None);
        assert!(full_context.candidate().input_delta.is_none());
        assert_eq!(full_context.origin_socket_id(), None);

        record_continuation_for_owner(&reservation, &appended, Some("resp_2"), Some(1), &[]);
        let changed = request_with_input(
            vec![input("one"), input("two"), input("three")],
            Some(json!({"service_tier": "flex"})),
        );
        let reservation = continuation_candidate_for_owner(Some(&owner), &changed, true);
        assert_eq!(
            reservation.candidate().disabled_reason.as_deref(),
            Some("prompt_changed")
        );
        assert!(!has_continuation_for_owner_for_tests(&owner));
    }

    #[test]
    fn route_owned_prompt_cache_key_does_not_break_append_only_detection() {
        let _registry_guard = lock_registry();
        let owner = agent_owner("session-a", "agent-a");
        let first = request_with_input(
            vec![input("one")],
            Some(json!({"prompt_cache_key": "route-owned-key"})),
        );
        start_and_record(&owner, &first, "resp_1");

        let appended = request_with_input(vec![input("one"), input("two")], None);
        let reservation = continuation_candidate_for_owner(Some(&owner), &appended, true);
        assert_eq!(
            reservation.candidate().previous_response_id.as_deref(),
            Some("resp_1")
        );
        assert_eq!(reservation.candidate().input_delta_count, 1);
    }

    #[test]
    fn route_rebuild_retains_generation_and_fences_stale_route_cleanup() {
        let _registry_guard = lock_registry();
        let owner = agent_owner("route-session", "route-agent");
        let request = request_with_input(vec![input("one")], None);
        let route_a = route(&owner, "token-a");
        let route_b = route(&owner, "token-b");

        let first =
            continuation_candidate_for_owner(Some(&owner), &request, true).bind_route(&route_a);
        assert!(first.turn_id().is_some_and(|generation| generation != 0));
        assert!(first.cleanup_epoch().is_some_and(|epoch| epoch != 0));
        record_continuation_for_owner(&first, &request, Some("resp_a"), Some(11), &[]);

        let appended = request_with_input(vec![input("one"), input("two")], None);
        let reserved = continuation_candidate_for_owner(Some(&owner), &appended, true);
        let generation = reserved.turn_id();
        let bound_a = reserved.bind_route(&route_a);
        assert_eq!(bound_a.turn_id(), generation);
        assert_eq!(
            bound_a.candidate().previous_response_id.as_deref(),
            Some("resp_a")
        );
        assert_eq!(bound_a.origin_socket_id(), Some(11));

        let bound_b = bound_a.bind_route(&route_b);
        assert_eq!(bound_b.turn_id(), generation);
        assert_eq!(bound_b.candidate().previous_response_id, None);
        assert_eq!(bound_b.origin_socket_id(), None);
        assert_eq!(
            bound_b.candidate().disabled_reason.as_deref(),
            Some("route_changed")
        );
        assert!(bound_b.cleanup_epoch() > bound_a.cleanup_epoch());
        assert_ne!(bound_a.route_key(), bound_b.route_key());

        record_continuation_for_owner(&bound_a, &appended, Some("stale"), Some(11), &[]);
        abort_continuation_for_owner(&bound_a);
        assert!(is_current_turn_for_owner(&bound_b));
        record_continuation_for_owner(&bound_b, &appended, Some("resp_b"), Some(22), &[]);
        assert!(has_continuation_for_owner_for_tests(&owner));
    }

    #[test]
    fn newer_generation_supersedes_older_before_route_binding() {
        let _registry_guard = lock_registry();
        let owner = main_owner("generation-session");
        let request = request_with_input(vec![input("one")], None);
        let stale = continuation_candidate_for_owner(Some(&owner), &request, true);
        let current = continuation_candidate_for_owner(Some(&owner), &request, true);

        assert!(current.turn_id().unwrap() > stale.turn_id().unwrap());
        let stale = stale.bind_route(&route(&owner, "token-a"));
        assert_eq!(
            stale.candidate().disabled_reason.as_deref(),
            Some("superseded_turn")
        );
        assert!(!is_current_turn_for_owner(&stale));
        assert!(is_current_turn_for_owner(&current));
    }

    #[tokio::test]
    async fn reservation_fences_owner_before_deferred_canonical_evaluation() {
        let _registry_guard = lock_continuation_registry_for_async_tests().await;
        clear_all_continuations_for_tests();
        let owner = main_owner("deferred-evaluation");
        let first_request = request_with_input(vec![input("one")], None);
        start_and_record(&owner, &first_request, "resp_1");

        let reserved = reserve_continuation_for_owner(Some(&owner), true);
        assert!(reserved.candidate().previous_response_id.is_none());
        let (request_tx, request_rx) = tokio::sync::oneshot::channel();
        let evaluator = tokio::spawn(async move {
            let canonical = request_rx.await.unwrap();
            reserved.evaluate(&canonical)
        });

        let newer = reserve_continuation_for_owner(Some(&owner), true);
        let canonical = request_with_input(vec![input("one"), input("two")], None);
        request_tx.send(canonical).unwrap();
        let stale = evaluator.await.unwrap();

        assert_eq!(
            stale.candidate().previous_response_id.as_deref(),
            Some("resp_1")
        );
        assert_eq!(stale.candidate().input_delta_count, 1);
        assert!(!is_current_turn_for_owner(&stale));
        assert!(is_current_turn_for_owner(&newer));
    }

    #[test]
    fn detached_summary_has_no_owner_provenance_or_owner_side_effects() {
        let _registry_guard = lock_registry();
        let owner = main_owner("detached-summary");
        let request = request_with_input(vec![input("one")], None);
        start_and_record(&owner, &request, "resp_1");

        let detached =
            ContinuationReservation::detached(request.input.len(), "claude_plaintext_summary")
                .bind_route(&route(&owner, "token-a"));
        assert_eq!(detached.owner(), None);
        assert_eq!(detached.turn_id(), None);
        assert_eq!(detached.route_key(), None);
        assert_eq!(detached.origin_socket_id(), None);
        assert_eq!(
            record_continuation_for_owner(&detached, &request, Some("resp_summary"), Some(99), &[],),
            ContinuationPublication::Rejected
        );
        abort_continuation_for_owner(&detached);
        assert!(has_continuation_for_owner_for_tests(&owner));
    }

    #[test]
    fn turn_publication_is_one_shot_and_late_abort_cannot_clear_it() {
        let _registry_guard = lock_registry();
        let owner = main_owner("one-shot-publication");
        let request = request_with_input(vec![input("one")], None);
        let reservation = continuation_candidate_for_owner(Some(&owner), &request, true);

        assert_eq!(
            record_continuation_for_owner(&reservation, &request, Some("resp_first"), Some(1), &[],),
            ContinuationPublication::Published
        );
        assert_eq!(
            record_continuation_for_owner(&reservation, &request, Some("resp_late"), Some(2), &[],),
            ContinuationPublication::Rejected
        );
        abort_continuation_for_owner(&reservation);
        assert!(has_continuation_for_owner_for_tests(&owner));
        assert_eq!(
            reservation.turn_outcome.as_ref().unwrap().state(),
            TURN_PUBLISHED
        );
    }

    #[tokio::test]
    async fn cancellation_winning_rejects_late_publication_and_pool_reinsertion() {
        let _registry_guard = lock_continuation_registry_for_async_tests().await;
        clear_all_continuations_for_tests();
        let owner = main_owner("cancel-wins");
        let request = request_with_input(vec![input("one")], None);
        let reservation = continuation_candidate_for_owner(Some(&owner), &request, true);
        let late = reservation.clone();
        let late_request = request.clone();
        let cancelled = Arc::new(tokio::sync::Notify::new());
        let cancellation_done = cancelled.clone();
        let publisher = tokio::spawn(async move {
            cancelled.notified().await;
            record_continuation_for_owner(&late, &late_request, Some("resp_late"), Some(1), &[])
        });

        abort_continuation_for_owner(&reservation);
        cancellation_done.notify_one();
        assert_eq!(publisher.await.unwrap(), ContinuationPublication::Rejected);
        assert!(
            if_current_turn_for_owner(&reservation, || ()).is_none(),
            "a cancelled turn must not reinsert its socket"
        );
        assert_eq!(
            reservation.turn_outcome.as_ref().unwrap().state(),
            TURN_CANCELLED
        );
        assert!(!has_continuation_owner_state_for_tests(&owner));
    }

    #[test]
    fn pool_reinsertion_check_does_not_beat_cancellation() {
        let _registry_guard = lock_registry();
        let owner = main_owner("pool-before-cancel");
        let request = request_with_input(vec![input("one")], None);
        let reservation = continuation_candidate_for_owner(Some(&owner), &request, true);

        assert!(
            if_current_turn_for_owner(&reservation, || ()).is_some(),
            "the physical completion may reinsert while the turn is current"
        );
        abort_continuation_for_owner(&reservation);
        assert_eq!(
            record_continuation_for_owner(
                &reservation,
                &request,
                Some("resp_completed"),
                Some(41),
                &[],
            ),
            ContinuationPublication::Rejected
        );
        assert!(!has_continuation_owner_state_for_tests(&owner));
        assert_eq!(
            reservation.turn_outcome.as_ref().unwrap().state(),
            TURN_CANCELLED
        );
    }

    #[test]
    fn lane_switch_changes_continuation_prompt_signature() {
        let input = vec![ResponsesInputItem::Message {
            role: "user".to_string(),
            content: vec![
                super::super::translate::request::ResponsesContentPart::InputText {
                    text: "one".to_string(),
                },
            ],
        }];
        let lite = request_with_input(
            input.clone(),
            Some(json!({
                "parallel_tool_calls": false,
                "client_metadata": {
                    "ws_request_header_x_openai_internal_codex_responses_lite": "true"
                }
            })),
        );
        let full = request_with_input(input.clone(), None);
        let state = ContinuationState {
            response_id: "resp_1".to_string(),
            socket_id: 1,
            route_key: None,
            prompt_signature: prompt_signature(&lite),
            transcript: input,
            retained_bytes: 0,
            updated_at: now_ms(),
        };

        let owner = main_owner("session-a");
        let reservation = ContinuationReservation::for_owner_turn(Some(&owner), Some(1));
        let candidate = continuation_candidate_from_state(
            &reservation,
            &full,
            ReservationSnapshot {
                state: Some(state),
                superseded_turn: false,
                reserved_at: now_ms(),
            },
            true,
        );

        let candidate = candidate.candidate();
        assert_eq!(candidate.disabled_reason.as_deref(), Some("prompt_changed"));
        assert!(candidate.previous_response_id.is_none());
        assert!(candidate.input_delta.is_none());
    }
}
