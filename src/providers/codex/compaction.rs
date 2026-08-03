use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::anthropic::sse::parse_sse_events;
use crate::provider::RequestContext;
use crate::providers::codex::client::{BufferedRetryState, CodexError, CodexHttpClient};
use crate::request_identity::OpaqueLane;

use super::state::{CodexBoundRoute, CodexConversationKey};

use super::translate::request::{
    ResponsesContentPart, ResponsesInputItem, ResponsesRequest, is_compact_message_text,
    request_uses_responses_lite,
};

const RETAINED_MESSAGE_TOKEN_BUDGET: u64 = 20_000;
const STATE_TTL_MS: u64 = 30 * 60 * 1_000;
const MAX_STATES: usize = 1_000;
const MAX_STATE_BYTES: usize = 4 * 1024 * 1024;
const MAX_TOTAL_STATE_BYTES: usize = 20_000_000;
const MIN_PORTABLE_SUMMARY_BYTES: usize = 32;

#[derive(Debug)]
pub enum CompactionError {
    Upstream(CodexError),
    InvalidResponse(String),
}

impl std::fmt::Display for CompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upstream(error) => write!(f, "{error}"),
            Self::InvalidResponse(message) => f.write_str(message),
        }
    }
}

enum CompactionPhase {
    Unconfirmed,
    Anchored { portable_summary: String },
}

struct CompactionState {
    model: String,
    use_responses_lite: Option<bool>,
    native_history: Vec<ResponsesInputItem>,
    phase: CompactionPhase,
    updated_at: u64,
}

#[derive(Default)]
struct CompactionRegistry {
    states: HashMap<String, CompactionState>,
    total_bytes: usize,
}

static REGISTRY: Mutex<Option<CompactionRegistry>> = Mutex::new(None);

#[derive(Debug)]
enum BoundPhase {
    PendingRemote,
    PendingAnchor {
        native_history: Vec<ResponsesInputItem>,
    },
    Anchored {
        native_history: Vec<ResponsesInputItem>,
        portable_summary: String,
        active_replays: HashSet<u64>,
    },
}

#[derive(Debug)]
struct BoundCompactionState {
    lane: OpaqueLane,
    cleanup_epoch: u64,
    generation: u64,
    revision: u64,
    model: String,
    phase: BoundPhase,
    updated_at: u64,
    activity: u64,
}

#[derive(Debug)]
struct LaneState {
    cleanup_epoch: u64,
    latest_generation: u64,
    pending_generations: HashSet<u64>,
    owned_route_keys: HashSet<CodexConversationKey>,
}

#[derive(Default)]
struct BoundRegistry {
    bound_states: HashMap<CodexConversationKey, BoundCompactionState>,
    lanes: HashMap<OpaqueLane, LaneState>,
    next_generation: u64,
    next_revision: u64,
    next_replay_id: u64,
    next_activity: u64,
}

static BOUND_REGISTRY: Mutex<Option<BoundRegistry>> = Mutex::new(None);

#[cfg(test)]
static TEST_REGISTRY_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[cfg(test)]
pub(crate) fn lock_compaction_registry_for_tests() -> tokio::sync::MutexGuard<'static, ()> {
    TEST_REGISTRY_LOCK.blocking_lock()
}

#[cfg(test)]
pub(crate) async fn lock_compaction_registry_for_async_tests()
-> tokio::sync::MutexGuard<'static, ()> {
    TEST_REGISTRY_LOCK.lock().await
}

#[derive(Debug)]
pub(crate) struct CompactionStartPermit {
    lane: OpaqueLane,
    cleanup_epoch: u64,
    generation: u64,
}

impl Drop for CompactionStartPermit {
    fn drop(&mut self) {
        remove_pending_generation(self.lane, self.cleanup_epoch, self.generation);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundLeaseKind {
    Build,
    Replay { replay_id: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BoundLeaseStamp {
    route_key: CodexConversationKey,
    lane: OpaqueLane,
    cleanup_epoch: u64,
    generation: u64,
    revision: u64,
}

#[derive(Debug)]
struct BoundLeaseCleanup {
    stamp: BoundLeaseStamp,
    kind: BoundLeaseKind,
    armed: AtomicBool,
}

impl Drop for BoundLeaseCleanup {
    fn drop(&mut self) {
        if self.armed.swap(false, Ordering::AcqRel) {
            abort_bound_lease(self.stamp, self.kind);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CompactionLease {
    cleanup: Arc<BoundLeaseCleanup>,
}

impl CompactionLease {
    fn new(stamp: BoundLeaseStamp, kind: BoundLeaseKind) -> Self {
        Self {
            cleanup: Arc::new(BoundLeaseCleanup {
                stamp,
                kind,
                armed: AtomicBool::new(true),
            }),
        }
    }

    fn stamp(&self) -> BoundLeaseStamp {
        self.cleanup.stamp
    }

    fn kind(&self) -> BoundLeaseKind {
        self.cleanup.kind
    }

    fn disarm(&self) -> bool {
        self.cleanup.armed.swap(false, Ordering::AcqRel)
    }
}

pub(crate) struct CompactionReplay {
    pub(crate) request: ResponsesRequest,
    pub(crate) lease: CompactionLease,
}

pub(crate) fn reserve_compaction_start(lane: Option<OpaqueLane>) -> Option<CompactionStartPermit> {
    let lane = lane?;
    let mut guard = BOUND_REGISTRY.lock().unwrap();
    let registry = guard.get_or_insert_with(BoundRegistry::default);
    let generation =
        next_checked_nonzero(&mut registry.next_generation, "Codex compaction generation");
    let lane_state = registry.lanes.entry(lane).or_insert_with(|| LaneState {
        cleanup_epoch: generation,
        latest_generation: generation,
        pending_generations: HashSet::new(),
        owned_route_keys: HashSet::new(),
    });
    lane_state.latest_generation = generation;
    lane_state.pending_generations.insert(generation);
    Some(CompactionStartPermit {
        lane,
        cleanup_epoch: lane_state.cleanup_epoch,
        generation,
    })
}

pub(crate) fn begin_compaction_for_route(
    permit: &CompactionStartPermit,
    route: &CodexBoundRoute,
    model: &str,
) -> Option<CompactionLease> {
    if route.lane() != Some(permit.lane) {
        return None;
    }
    let route_key = route.conversation_key()?;
    let now = now_ms();
    let mut guard = BOUND_REGISTRY.lock().unwrap();
    let registry = guard.as_mut()?;
    let lane_state = registry.lanes.get(&permit.lane)?;
    if lane_state.cleanup_epoch != permit.cleanup_epoch
        || lane_state.latest_generation != permit.generation
        || !lane_state.pending_generations.contains(&permit.generation)
    {
        return None;
    }

    remove_owned_lane_states(registry, permit.lane);
    let revision = next_checked_nonzero(&mut registry.next_revision, "Codex compaction revision");
    let activity = next_checked_nonzero(&mut registry.next_activity, "Codex compaction activity");
    registry.bound_states.insert(
        route_key,
        BoundCompactionState {
            lane: permit.lane,
            cleanup_epoch: permit.cleanup_epoch,
            generation: permit.generation,
            revision,
            model: model.to_string(),
            phase: BoundPhase::PendingRemote,
            updated_at: now,
            activity,
        },
    );
    registry
        .lanes
        .get_mut(&permit.lane)
        .expect("validated compaction lane")
        .owned_route_keys
        .insert(route_key);

    Some(CompactionLease::new(
        BoundLeaseStamp {
            route_key,
            lane: permit.lane,
            cleanup_epoch: permit.cleanup_epoch,
            generation: permit.generation,
            revision,
        },
        BoundLeaseKind::Build,
    ))
}

pub(crate) fn store_compaction_for_route(
    lease: &CompactionLease,
    native_history: Vec<ResponsesInputItem>,
) -> bool {
    if lease.kind() != BoundLeaseKind::Build {
        return false;
    }
    let stamp = lease.stamp();
    let now = now_ms();
    let mut guard = BOUND_REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return false;
    };
    let Some(state) = registry.bound_states.get(&stamp.route_key) else {
        return false;
    };
    if !bound_state_matches(state, stamp) || !matches!(state.phase, BoundPhase::PendingRemote) {
        return false;
    }

    let candidate_size = bound_state_size_parts(
        &stamp.route_key,
        stamp.lane,
        &state.model,
        &native_history,
        None,
        0,
    );
    if candidate_size > MAX_STATE_BYTES
        || !make_bound_retained_room(registry, stamp.route_key, candidate_size, now)
    {
        remove_bound_state_if_matches(registry, stamp, BoundExpectedPhase::Build);
        return false;
    }

    let activity = next_checked_nonzero(&mut registry.next_activity, "Codex compaction activity");
    let Some(state) = registry.bound_states.get_mut(&stamp.route_key) else {
        return false;
    };
    if !bound_state_matches(state, stamp) || !matches!(state.phase, BoundPhase::PendingRemote) {
        return false;
    }
    state.phase = BoundPhase::PendingAnchor { native_history };
    state.updated_at = now;
    state.activity = activity;
    true
}

pub(crate) fn apply_compaction_replay_for_route(
    route: &CodexBoundRoute,
    request: &ResponsesRequest,
) -> Option<CompactionReplay> {
    route.validate_responses_request(request).ok()?;
    let route_key = route.conversation_key()?;
    let lane = route.lane()?;
    let now = now_ms();
    let mut guard = BOUND_REGISTRY.lock().unwrap();
    let registry = guard.as_mut()?;
    let expired = registry.bound_states.get(&route_key).is_some_and(|state| {
        bound_state_is_inactive(state) && now.saturating_sub(state.updated_at) > STATE_TTL_MS
    });
    if expired {
        remove_bound_state(registry, route_key);
        return None;
    }

    let (cleanup_epoch, generation, revision, native_history, portable_summary, active_replays) = {
        let state = registry.bound_states.get(&route_key)?;
        if state.lane != lane || state.model != request.model {
            return None;
        }
        let BoundPhase::Anchored {
            native_history,
            portable_summary,
            active_replays,
        } = &state.phase
        else {
            return None;
        };
        (
            state.cleanup_epoch,
            state.generation,
            state.revision,
            native_history.clone(),
            portable_summary.clone(),
            active_replays.len(),
        )
    };

    let (envelope, conversation) = split_input_envelope(&request.input);
    let summary_item = conversation.first()?;
    let text = message_text(summary_item)?;
    if text.match_indices(&portable_summary).count() != 1 || conversation.len() == 1 {
        return None;
    }
    let candidate_state_size = bound_state_size_parts(
        &route_key,
        lane,
        &request.model,
        &native_history,
        Some(&portable_summary),
        active_replays.saturating_add(1),
    );
    if candidate_state_size > MAX_STATE_BYTES {
        return None;
    }
    let mut replay = request.clone();
    replay.input = envelope
        .iter()
        .cloned()
        .chain(native_history)
        .chain(conversation[1..].iter().cloned())
        .collect();
    if serialized_size(&replay.input) > MAX_STATE_BYTES
        || !make_bound_retained_room(registry, route_key, candidate_state_size, now)
    {
        return None;
    }

    let replay_id = next_checked_nonzero(&mut registry.next_replay_id, "Codex compaction replay");
    let activity = next_checked_nonzero(&mut registry.next_activity, "Codex compaction activity");
    let state = registry.bound_states.get_mut(&route_key)?;
    if state.cleanup_epoch != cleanup_epoch
        || state.generation != generation
        || state.revision != revision
    {
        return None;
    }
    let BoundPhase::Anchored { active_replays, .. } = &mut state.phase else {
        return None;
    };
    active_replays.insert(replay_id);
    state.updated_at = now;
    state.activity = activity;

    Some(CompactionReplay {
        request: replay,
        lease: CompactionLease::new(
            BoundLeaseStamp {
                route_key,
                lane,
                cleanup_epoch,
                generation,
                revision,
            },
            BoundLeaseKind::Replay { replay_id },
        ),
    })
}

pub(crate) fn activate_compaction_for_route(
    lease: &CompactionLease,
    output: &[ResponsesInputItem],
) -> bool {
    if !lease.disarm() {
        return false;
    }
    let stamp = lease.stamp();
    let mut guard = BOUND_REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return false;
    };
    match lease.kind() {
        BoundLeaseKind::Build => activate_bound_build(registry, stamp, output),
        BoundLeaseKind::Replay { replay_id } => activate_bound_replay(registry, stamp, replay_id),
    }
}

pub(crate) fn abort_compaction_for_route(lease: &CompactionLease) {
    if lease.disarm() {
        abort_bound_lease(lease.stamp(), lease.kind());
    }
}

pub(crate) fn clear_compactions_for_lane(lane: OpaqueLane) {
    let mut guard = BOUND_REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return;
    };
    remove_owned_lane_states(registry, lane);
    registry.lanes.remove(&lane);
}

#[cfg(test)]
pub(crate) fn has_bound_compaction_for_tests(route: &CodexBoundRoute) -> bool {
    let Some(route_key) = route.conversation_key() else {
        return false;
    };
    BOUND_REGISTRY
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|registry| registry.bound_states.contains_key(&route_key))
}

#[cfg(test)]
pub(crate) fn has_bound_lane_metadata_for_tests(lane: OpaqueLane) -> bool {
    BOUND_REGISTRY
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|registry| registry.lanes.contains_key(&lane))
}

fn activate_bound_build(
    registry: &mut BoundRegistry,
    stamp: BoundLeaseStamp,
    output: &[ResponsesInputItem],
) -> bool {
    let Some(state) = registry.bound_states.get(&stamp.route_key) else {
        return false;
    };
    if !bound_state_matches(state, stamp)
        || !matches!(state.phase, BoundPhase::PendingAnchor { .. })
    {
        return false;
    }
    let Some(portable_summary) = portable_summary_text(output) else {
        remove_bound_state_if_matches(registry, stamp, BoundExpectedPhase::Build);
        return false;
    };
    let BoundPhase::PendingAnchor { native_history } = &state.phase else {
        unreachable!("validated pending anchor phase");
    };
    let candidate_size = bound_state_size_parts(
        &stamp.route_key,
        stamp.lane,
        &state.model,
        native_history,
        Some(&portable_summary),
        0,
    );
    let now = now_ms();
    if candidate_size > MAX_STATE_BYTES
        || !make_bound_retained_room(registry, stamp.route_key, candidate_size, now)
    {
        remove_bound_state_if_matches(registry, stamp, BoundExpectedPhase::Build);
        return false;
    }

    let revision = next_checked_nonzero(&mut registry.next_revision, "Codex compaction revision");
    let activity = next_checked_nonzero(&mut registry.next_activity, "Codex compaction activity");
    let Some(state) = registry.bound_states.get_mut(&stamp.route_key) else {
        return false;
    };
    if !bound_state_matches(state, stamp) {
        return false;
    }
    let BoundPhase::PendingAnchor { native_history } =
        std::mem::replace(&mut state.phase, BoundPhase::PendingRemote)
    else {
        return false;
    };
    state.phase = BoundPhase::Anchored {
        native_history,
        portable_summary,
        active_replays: HashSet::new(),
    };
    state.revision = revision;
    state.updated_at = now;
    state.activity = activity;
    true
}

fn activate_bound_replay(
    registry: &mut BoundRegistry,
    stamp: BoundLeaseStamp,
    replay_id: u64,
) -> bool {
    let revision = next_checked_nonzero(&mut registry.next_revision, "Codex compaction revision");
    let activity = next_checked_nonzero(&mut registry.next_activity, "Codex compaction activity");
    let Some(state) = registry.bound_states.get_mut(&stamp.route_key) else {
        return false;
    };
    if !bound_state_matches(state, stamp) {
        return false;
    }
    let BoundPhase::Anchored { active_replays, .. } = &mut state.phase else {
        return false;
    };
    if !active_replays.remove(&replay_id) {
        return false;
    }
    active_replays.clear();
    state.revision = revision;
    state.updated_at = now_ms();
    state.activity = activity;
    true
}

fn abort_bound_lease(stamp: BoundLeaseStamp, kind: BoundLeaseKind) {
    let mut guard = BOUND_REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return;
    };
    match kind {
        BoundLeaseKind::Build => {
            remove_bound_state_if_matches(registry, stamp, BoundExpectedPhase::Build);
        }
        BoundLeaseKind::Replay { replay_id } => {
            let remove = registry
                .bound_states
                .get_mut(&stamp.route_key)
                .filter(|state| bound_state_matches(state, stamp))
                .and_then(|state| match &mut state.phase {
                    BoundPhase::Anchored { active_replays, .. } => {
                        if active_replays.remove(&replay_id) {
                            Some(active_replays.is_empty())
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .unwrap_or(false);
            if remove {
                remove_bound_state(registry, stamp.route_key);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum BoundExpectedPhase {
    Build,
}

fn remove_bound_state_if_matches(
    registry: &mut BoundRegistry,
    stamp: BoundLeaseStamp,
    expected: BoundExpectedPhase,
) -> bool {
    let matches = registry
        .bound_states
        .get(&stamp.route_key)
        .is_some_and(|state| {
            bound_state_matches(state, stamp)
                && match expected {
                    BoundExpectedPhase::Build => matches!(
                        state.phase,
                        BoundPhase::PendingRemote | BoundPhase::PendingAnchor { .. }
                    ),
                }
        });
    if matches {
        remove_bound_state(registry, stamp.route_key);
    }
    matches
}

fn bound_state_matches(state: &BoundCompactionState, stamp: BoundLeaseStamp) -> bool {
    state.lane == stamp.lane
        && state.cleanup_epoch == stamp.cleanup_epoch
        && state.generation == stamp.generation
        && state.revision == stamp.revision
}

fn remove_pending_generation(lane: OpaqueLane, cleanup_epoch: u64, generation: u64) {
    let mut guard = BOUND_REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return;
    };
    if let Some(state) = registry.lanes.get_mut(&lane)
        && state.cleanup_epoch == cleanup_epoch
    {
        state.pending_generations.remove(&generation);
    }
    cleanup_lane_metadata(registry, lane);
}

fn remove_owned_lane_states(registry: &mut BoundRegistry, lane: OpaqueLane) {
    let route_keys = registry
        .lanes
        .get(&lane)
        .map(|state| state.owned_route_keys.iter().copied().collect::<Vec<_>>())
        .unwrap_or_else(|| {
            registry
                .bound_states
                .iter()
                .filter_map(|(route_key, state)| (state.lane == lane).then_some(*route_key))
                .collect()
        });
    for route_key in route_keys {
        remove_bound_state(registry, route_key);
    }
}

fn remove_bound_state(registry: &mut BoundRegistry, route_key: CodexConversationKey) {
    let Some(state) = registry.bound_states.remove(&route_key) else {
        return;
    };
    if let Some(lane) = registry.lanes.get_mut(&state.lane) {
        lane.owned_route_keys.remove(&route_key);
    }
    cleanup_lane_metadata(registry, state.lane);
}

fn cleanup_lane_metadata(registry: &mut BoundRegistry, lane: OpaqueLane) {
    if registry.lanes.get(&lane).is_some_and(|state| {
        state.pending_generations.is_empty() && state.owned_route_keys.is_empty()
    }) {
        registry.lanes.remove(&lane);
    }
}

fn next_checked_nonzero(sequence: &mut u64, label: &str) -> u64 {
    *sequence = sequence
        .checked_add(1)
        .unwrap_or_else(|| panic!("{label} exhausted"));
    *sequence
}

#[cfg(test)]
fn bound_state_size(route_key: &CodexConversationKey, state: &BoundCompactionState) -> usize {
    match &state.phase {
        BoundPhase::PendingRemote => 0,
        BoundPhase::PendingAnchor { native_history } => {
            bound_state_size_parts(route_key, state.lane, &state.model, native_history, None, 0)
        }
        BoundPhase::Anchored {
            native_history,
            portable_summary,
            active_replays,
        } => bound_state_size_parts(
            route_key,
            state.lane,
            &state.model,
            native_history,
            Some(portable_summary),
            active_replays.len(),
        ),
    }
}

fn bound_state_size_parts(
    _route_key: &CodexConversationKey,
    _lane: OpaqueLane,
    model: &str,
    native_history: &[ResponsesInputItem],
    portable_summary: Option<&str>,
    active_replays: usize,
) -> usize {
    std::mem::size_of::<CodexConversationKey>()
        .saturating_add(std::mem::size_of::<OpaqueLane>())
        .saturating_add(model.len())
        .saturating_add(serialized_size(native_history))
        .saturating_add(portable_summary.map_or(0, str::len))
        .saturating_add(active_replays.saturating_mul(std::mem::size_of::<u64>()))
}

fn bound_retained_state_size(
    route_key: &CodexConversationKey,
    state: &BoundCompactionState,
) -> usize {
    match &state.phase {
        BoundPhase::PendingRemote => 0,
        BoundPhase::PendingAnchor { native_history } => {
            bound_state_size_parts(route_key, state.lane, &state.model, native_history, None, 0)
        }
        BoundPhase::Anchored {
            native_history,
            portable_summary,
            active_replays,
        } => bound_state_size_parts(
            route_key,
            state.lane,
            &state.model,
            native_history,
            Some(portable_summary),
            active_replays.len(),
        ),
    }
}

fn bound_state_is_retained(state: &BoundCompactionState) -> bool {
    !matches!(state.phase, BoundPhase::PendingRemote)
}

fn bound_state_is_inactive(state: &BoundCompactionState) -> bool {
    matches!(
        &state.phase,
        BoundPhase::Anchored { active_replays, .. } if active_replays.is_empty()
    )
}

fn bound_stored_count(registry: &BoundRegistry) -> usize {
    registry
        .bound_states
        .values()
        .filter(|state| bound_state_is_retained(state))
        .count()
}

fn bound_total_bytes(registry: &BoundRegistry) -> usize {
    registry
        .bound_states
        .iter()
        .map(|(route_key, state)| bound_retained_state_size(route_key, state))
        .sum()
}

fn make_bound_retained_room(
    registry: &mut BoundRegistry,
    route_key: CodexConversationKey,
    candidate_size: usize,
    now: u64,
) -> bool {
    let current_state = registry.bound_states.get(&route_key);
    let current_size =
        current_state.map_or(0, |state| bound_retained_state_size(&route_key, state));
    let current_count = usize::from(current_state.is_some_and(bound_state_is_retained));
    let mut count = bound_stored_count(registry)
        .saturating_sub(current_count)
        .saturating_add(1);
    let mut total = bound_total_bytes(registry)
        .saturating_sub(current_size)
        .saturating_add(candidate_size);
    let mut evictable = registry
        .bound_states
        .iter()
        .filter(|(key, state)| **key != route_key && bound_state_is_inactive(state))
        .map(|(key, state)| {
            (
                *key,
                state.activity,
                bound_retained_state_size(key, state),
                now.saturating_sub(state.updated_at) > STATE_TTL_MS,
            )
        })
        .collect::<Vec<_>>();
    evictable.sort_unstable_by_key(|(_, activity, _, _)| *activity);

    let mut remove = Vec::new();
    for (key, _, size, _) in evictable.iter().filter(|(_, _, _, expired)| *expired) {
        count = count.saturating_sub(1);
        total = total.saturating_sub(*size);
        remove.push(*key);
    }
    if count > MAX_STATES || total > MAX_TOTAL_STATE_BYTES {
        for (key, _, size, _) in evictable.iter().filter(|(_, _, _, expired)| !*expired) {
            count = count.saturating_sub(1);
            total = total.saturating_sub(*size);
            remove.push(*key);
            if count <= MAX_STATES && total <= MAX_TOTAL_STATE_BYTES {
                break;
            }
        }
    }
    if count > MAX_STATES || total > MAX_TOTAL_STATE_BYTES {
        return false;
    }
    for key in remove {
        remove_bound_state(registry, key);
    }
    true
}

pub async fn request_compaction(
    client: &CodexHttpClient,
    request: &ResponsesRequest,
    ctx: &RequestContext,
) -> Result<Vec<ResponsesInputItem>, CompactionError> {
    let (compaction_request, conversation) = prepare_compaction_request(request);
    let response = client
        .post_codex_for_owner(&compaction_request, ctx, None)
        .await
        .map_err(CompactionError::Upstream)?;
    let compaction = parse_compaction_response(&response.body)?;
    Ok(build_compacted_history(&conversation, compaction))
}

pub(crate) async fn request_compaction_bound(
    client: &CodexHttpClient,
    route: &CodexBoundRoute,
    request: &ResponsesRequest,
    ctx: &RequestContext,
    retry_state: &mut BufferedRetryState,
) -> Result<Vec<ResponsesInputItem>, CompactionError> {
    let (compaction_request, conversation) = prepare_compaction_request(request);
    let response = client
        .post_codex_bound_with_retry_state(route, &compaction_request, ctx, None, retry_state)
        .await
        .map_err(CompactionError::Upstream)?;
    let compaction = parse_compaction_response(&response.body)?;
    Ok(build_compacted_history(&conversation, compaction))
}

fn prepare_compaction_request(
    request: &ResponsesRequest,
) -> (ResponsesRequest, Vec<ResponsesInputItem>) {
    let (envelope, conversation) = split_input_envelope(&request.input);
    let conversation = without_compaction_instruction(conversation);
    let mut compaction_request = request.clone();
    compaction_request.instructions = None;
    compaction_request.input = envelope
        .iter()
        .filter(|item| matches!(item, ResponsesInputItem::AdditionalTools { .. }))
        .cloned()
        .chain(conversation.iter().cloned())
        .chain(std::iter::once(ResponsesInputItem::CompactionTrigger))
        .collect();
    compaction_request.include = Some(vec!["reasoning.encrypted_content".to_string()]);
    (compaction_request, conversation)
}

pub fn store_compaction(
    session_id: &str,
    model: &str,
    native_history: Vec<ResponsesInputItem>,
) -> bool {
    store_compaction_state(session_id, model, None, native_history)
}

pub(crate) fn store_compaction_for_request(
    session_id: &str,
    request: &ResponsesRequest,
    native_history: Vec<ResponsesInputItem>,
) -> bool {
    store_compaction_state(
        session_id,
        &request.model,
        Some(request_uses_responses_lite(request)),
        native_history,
    )
}

fn store_compaction_state(
    session_id: &str,
    model: &str,
    use_responses_lite: Option<bool>,
    native_history: Vec<ResponsesInputItem>,
) -> bool {
    let state = CompactionState {
        model: model.to_string(),
        use_responses_lite,
        native_history,
        phase: CompactionPhase::Unconfirmed,
        updated_at: now_ms(),
    };
    if state_size(session_id, &state) > MAX_STATE_BYTES {
        clear_compaction(session_id);
        return false;
    }

    let now = state.updated_at;
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.get_or_insert_with(CompactionRegistry::default);
    evict_states(registry, now);
    registry.states.insert(session_id.to_string(), state);
    evict_states(registry, now);
    registry.states.contains_key(session_id)
}

pub fn activate_compaction(
    session_id: Option<&str>,
    model: &str,
    output: &[ResponsesInputItem],
) -> bool {
    activate_compaction_state(session_id, model, None, output)
}

pub(crate) fn activate_compaction_for_request(
    session_id: Option<&str>,
    request: &ResponsesRequest,
    output: &[ResponsesInputItem],
) -> bool {
    activate_compaction_state(
        session_id,
        &request.model,
        Some(request_uses_responses_lite(request)),
        output,
    )
}

fn activate_compaction_state(
    session_id: Option<&str>,
    model: &str,
    use_responses_lite: Option<bool>,
    output: &[ResponsesInputItem],
) -> bool {
    let Some(session_id) = session_id else {
        return false;
    };
    let Some(portable_summary) = portable_summary_text(output) else {
        clear_compaction(session_id);
        return false;
    };

    let now = now_ms();
    let mut guard = REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return false;
    };
    evict_states(registry, now);
    let Some(state) = registry.states.get_mut(session_id) else {
        return false;
    };
    if state.model != model
        || matches!(
            (state.use_responses_lite, use_responses_lite),
            (Some(stored_lane), Some(request_lane)) if stored_lane != request_lane
        )
        || !matches!(state.phase, CompactionPhase::Unconfirmed)
    {
        registry.states.remove(session_id);
        update_total_bytes(registry);
        return false;
    }
    state.phase = CompactionPhase::Anchored { portable_summary };
    state.updated_at = now;
    if state_size(session_id, state) > MAX_STATE_BYTES {
        registry.states.remove(session_id);
        update_total_bytes(registry);
        return false;
    }
    evict_states(registry, now);
    registry.states.contains_key(session_id)
}

pub fn apply_compaction_replay(
    session_id: Option<&str>,
    request: &ResponsesRequest,
) -> Option<ResponsesRequest> {
    let session_id = session_id?;
    let now = now_ms();
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.as_mut()?;
    evict_states(registry, now);
    let state = registry.states.get_mut(session_id)?;
    if state.model != request.model
        || state
            .use_responses_lite
            .is_some_and(|stored_lane| stored_lane != request_uses_responses_lite(request))
    {
        registry.states.remove(session_id);
        update_total_bytes(registry);
        return None;
    }
    let CompactionPhase::Anchored { portable_summary } = &state.phase else {
        return None;
    };

    let (envelope, conversation) = split_input_envelope(&request.input);
    let summary_item = conversation.first()?;
    let Some(text) = message_text(summary_item) else {
        registry.states.remove(session_id);
        update_total_bytes(registry);
        return None;
    };
    if text.match_indices(portable_summary).count() != 1 {
        registry.states.remove(session_id);
        update_total_bytes(registry);
        return None;
    }
    if conversation.len() == 1 {
        return None;
    }

    let mut replay = request.clone();
    replay.input = envelope
        .iter()
        .cloned()
        .chain(state.native_history.iter().cloned())
        .chain(conversation[1..].iter().cloned())
        .collect();
    if serialized_size(&replay.input) > MAX_STATE_BYTES {
        registry.states.remove(session_id);
        update_total_bytes(registry);
        return None;
    }
    state.updated_at = now;
    Some(replay)
}

pub fn abort_compaction_attempt(
    session_id: Option<&str>,
    compact_boundary: bool,
    request: &ResponsesRequest,
) {
    if (compact_boundary || request_contains_compaction(request))
        && let Some(session_id) = session_id
    {
        clear_compaction(session_id);
    }
}

pub fn request_contains_compaction(request: &ResponsesRequest) -> bool {
    request
        .input
        .iter()
        .any(|item| matches!(item, ResponsesInputItem::Compaction { .. }))
}

pub fn clear_compaction(session_id: &str) {
    let mut guard = REGISTRY.lock().unwrap();
    if let Some(registry) = guard.as_mut() {
        registry.states.remove(session_id);
        update_total_bytes(registry);
    }
}

fn split_input_envelope(
    input: &[ResponsesInputItem],
) -> (&[ResponsesInputItem], &[ResponsesInputItem]) {
    let prefix_len = input
        .iter()
        .take_while(|item| is_envelope_item(item))
        .count();
    input.split_at(prefix_len)
}

fn without_compaction_instruction(input: &[ResponsesInputItem]) -> Vec<ResponsesInputItem> {
    let mut input = input.to_vec();
    let remove_empty_message = if let Some(ResponsesInputItem::Message { role, content }) =
        input.last_mut()
        && role == "user"
    {
        content.retain(|part| {
            !matches!(
                part,
                ResponsesContentPart::InputText { text }
                    if is_compact_message_text(text)
            )
        });
        content.is_empty()
    } else {
        false
    };
    if remove_empty_message {
        input.pop();
    }
    input
}

fn is_envelope_item(item: &ResponsesInputItem) -> bool {
    match item {
        ResponsesInputItem::AdditionalTools { .. } => true,
        ResponsesInputItem::Message { role, .. } => role == "developer",
        _ => false,
    }
}

fn portable_summary_text(output: &[ResponsesInputItem]) -> Option<String> {
    let text = output
        .iter()
        .filter_map(|item| match item {
            ResponsesInputItem::Message { role, content } if role == "assistant" => Some(content),
            _ => None,
        })
        .flat_map(|content| content.iter())
        .filter_map(|part| match part {
            ResponsesContentPart::InputText { text }
            | ResponsesContentPart::OutputText { text } => Some(text.as_str()),
            ResponsesContentPart::InputImage { .. } => None,
        })
        .collect::<String>();
    let trimmed = text.trim();
    let summary = trimmed
        .split_once("<summary>")
        .and_then(|(_, rest)| rest.split_once("</summary>"))
        .map(|(summary, _)| summary.trim())
        .filter(|summary| !summary.is_empty())
        .unwrap_or(trimmed);
    (summary.len() >= MIN_PORTABLE_SUMMARY_BYTES).then(|| summary.to_string())
}

fn message_text(item: &ResponsesInputItem) -> Option<String> {
    let ResponsesInputItem::Message { content, .. } = item else {
        return None;
    };
    Some(
        content
            .iter()
            .filter_map(|part| match part {
                ResponsesContentPart::InputText { text }
                | ResponsesContentPart::OutputText { text } => Some(text.as_str()),
                ResponsesContentPart::InputImage { .. } => None,
            })
            .collect(),
    )
}

fn parse_compaction_response(body: &[u8]) -> Result<ResponsesInputItem, CompactionError> {
    let mut completed = false;
    let mut compacted = Vec::new();

    for event in parse_sse_events(body) {
        if event.data == "[DONE]" {
            continue;
        }
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(&event.data) else {
            continue;
        };
        match payload.get("type").and_then(serde_json::Value::as_str) {
            Some("error" | "response.error" | "response.failed") => {
                let message = payload
                    .pointer("/response/error/message")
                    .or_else(|| payload.pointer("/error/message"))
                    .or_else(|| payload.get("message"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("remote compaction failed");
                return Err(CompactionError::InvalidResponse(message.to_string()));
            }
            Some("response.output_item.done") => {
                let Some(item) = payload.get("item") else {
                    continue;
                };
                if item.get("type").and_then(serde_json::Value::as_str) == Some("compaction")
                    && let Some(encrypted_content) = item
                        .get("encrypted_content")
                        .and_then(serde_json::Value::as_str)
                {
                    compacted.push(ResponsesInputItem::Compaction {
                        encrypted_content: encrypted_content.to_string(),
                    });
                }
            }
            Some("response.completed") => completed = true,
            _ => {}
        }
    }

    if !completed {
        return Err(CompactionError::InvalidResponse(
            "remote compaction stream ended before response.completed".to_string(),
        ));
    }
    if compacted.len() != 1 {
        return Err(CompactionError::InvalidResponse(format!(
            "remote compaction expected exactly one compaction item, got {}",
            compacted.len()
        )));
    }
    Ok(compacted.pop().expect("validated one compaction item"))
}

fn build_compacted_history(
    input: &[ResponsesInputItem],
    compaction: ResponsesInputItem,
) -> Vec<ResponsesInputItem> {
    let retained = input
        .iter()
        .filter(|item| {
            matches!(
                item,
                ResponsesInputItem::Message { role, .. }
                    if matches!(role.as_str(), "user" | "developer" | "system")
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut retained = truncate_retained_messages(retained, RETAINED_MESSAGE_TOKEN_BUDGET);
    retained.push(compaction);
    retained
}

fn truncate_retained_messages(
    items: Vec<ResponsesInputItem>,
    max_tokens: u64,
) -> Vec<ResponsesInputItem> {
    let mut remaining = max_tokens;
    let mut retained = Vec::new();
    for item in items.into_iter().rev() {
        if remaining == 0 {
            break;
        }
        let tokens = message_tokens(&item).max(1);
        if tokens <= remaining {
            retained.push(item);
            remaining -= tokens;
        } else if let Some(item) = truncate_message(item, remaining) {
            retained.push(item);
            remaining = 0;
        }
    }
    retained.reverse();
    retained
}

fn message_tokens(item: &ResponsesInputItem) -> u64 {
    let ResponsesInputItem::Message { content, .. } = item else {
        return 0;
    };
    content
        .iter()
        .map(|part| match part {
            ResponsesContentPart::InputText { text }
            | ResponsesContentPart::OutputText { text } => text.len().div_ceil(4) as u64,
            ResponsesContentPart::InputImage { .. } => 2_000,
        })
        .sum()
}

fn truncate_message(item: ResponsesInputItem, max_tokens: u64) -> Option<ResponsesInputItem> {
    let ResponsesInputItem::Message { role, content } = item else {
        return Some(item);
    };
    let mut remaining_chars = max_tokens.saturating_mul(4) as usize;
    let mut truncated = Vec::new();
    for part in content {
        match part {
            ResponsesContentPart::InputImage { .. } => truncated.push(part),
            ResponsesContentPart::InputText { text } => {
                let text = truncate_text(text, &mut remaining_chars);
                if !text.is_empty() {
                    truncated.push(ResponsesContentPart::InputText { text });
                }
            }
            ResponsesContentPart::OutputText { text } => {
                let text = truncate_text(text, &mut remaining_chars);
                if !text.is_empty() {
                    truncated.push(ResponsesContentPart::OutputText { text });
                }
            }
        }
    }
    (!truncated.is_empty()).then_some(ResponsesInputItem::Message {
        role,
        content: truncated,
    })
}

fn truncate_text(mut text: String, remaining_chars: &mut usize) -> String {
    if *remaining_chars == 0 {
        return String::new();
    }
    if text.len() > *remaining_chars {
        let mut boundary = *remaining_chars;
        while !text.is_char_boundary(boundary) {
            boundary -= 1;
        }
        text.truncate(boundary);
    }
    *remaining_chars -= text.len();
    text
}

fn serialized_size(items: &[ResponsesInputItem]) -> usize {
    serde_json::to_vec(items).map_or(usize::MAX, |value| value.len())
}

fn state_size(session_id: &str, state: &CompactionState) -> usize {
    let summary_len = match &state.phase {
        CompactionPhase::Unconfirmed => 0,
        CompactionPhase::Anchored { portable_summary } => portable_summary.len(),
    };
    session_id.len()
        + state.model.len()
        + std::mem::size_of::<Option<bool>>()
        + summary_len
        + serialized_size(&state.native_history)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn update_total_bytes(registry: &mut CompactionRegistry) {
    registry.total_bytes = registry
        .states
        .iter()
        .map(|(session_id, state)| state_size(session_id, state))
        .sum();
}

fn evict_states(registry: &mut CompactionRegistry, now: u64) {
    registry
        .states
        .retain(|_, state| now.saturating_sub(state.updated_at) <= STATE_TTL_MS);
    update_total_bytes(registry);
    while registry.states.len() > MAX_STATES || registry.total_bytes > MAX_TOTAL_STATE_BYTES {
        let oldest = registry
            .states
            .iter()
            .min_by_key(|(_, state)| state.updated_at)
            .map(|(session_id, _)| session_id.clone());
        let Some(oldest) = oldest else {
            break;
        };
        registry.states.remove(&oldest);
        update_total_bytes(registry);
    }
}

pub fn clear_all_compactions_for_tests() {
    *REGISTRY.lock().unwrap() = None;
    *BOUND_REGISTRY.lock().unwrap() = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::codex::auth::token_store::StoredAuth;
    use crate::providers::codex::state::ProtocolLane;
    use crate::request_identity::{ConversationIdentity, LaneDomain, RequestPurpose, RequestScope};
    use serde_json::json;

    const SUMMARY: &str =
        "portable summary with enough detail to identify this compacted conversation";

    fn request(input: serde_json::Value) -> ResponsesRequest {
        request_for_lane(input, true)
    }

    fn request_for_lane(input: serde_json::Value, use_responses_lite: bool) -> ResponsesRequest {
        let mut request: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "input": input,
            "store": false,
            "stream": true,
            "parallel_tool_calls": !use_responses_lite,
            "text": {"verbosity":"low"}
        }))
        .unwrap();
        if use_responses_lite {
            request.client_metadata = Some(HashMap::from([(
                super::super::translate::request::RESPONSES_LITE_METADATA_KEY.to_string(),
                "true".to_string(),
            )]));
        }
        request
    }

    fn output(text: &str) -> Vec<ResponsesInputItem> {
        serde_json::from_value(json!([{
            "type":"message",
            "role":"assistant",
            "content":[{"type":"output_text","text":text}]
        }]))
        .unwrap()
    }

    fn conversation_lane(identity: ConversationIdentity) -> OpaqueLane {
        RequestScope::from_conversation_identity(Some(identity), RequestPurpose::Conversation)
            .provider_lane(LaneDomain::CodexConversation)
            .unwrap()
    }

    fn main_lane(session: &str) -> OpaqueLane {
        conversation_lane(ConversationIdentity::Main(session.to_string()))
    }

    fn agent_lane(session: &str, agent: &str) -> OpaqueLane {
        conversation_lane(ConversationIdentity::Agent(
            session.to_string(),
            agent.to_string(),
        ))
    }

    fn bound_route(lane: OpaqueLane, access: &str) -> CodexBoundRoute {
        CodexBoundRoute::new(
            StoredAuth {
                access: access.to_string(),
                refresh: String::new(),
                expires: u64::MAX,
                account_id: Some("account-a".to_string()),
            },
            "https://example.test/backend-api/codex/responses",
            ProtocolLane::ResponsesLite,
            Some(lane),
        )
        .unwrap()
    }

    fn native_history(tag: &str) -> Vec<ResponsesInputItem> {
        vec![ResponsesInputItem::Compaction {
            encrypted_content: tag.to_string(),
        }]
    }

    fn replay_request() -> ResponsesRequest {
        request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]))
    }

    fn stage_bound(route: &CodexBoundRoute, tag: &str) -> (CompactionStartPermit, CompactionLease) {
        let permit = reserve_compaction_start(route.lane()).unwrap();
        let lease = begin_compaction_for_route(&permit, route, "gpt-5.6-sol").unwrap();
        assert!(store_compaction_for_route(&lease, native_history(tag)));
        (permit, lease)
    }

    fn anchor_bound(lease: &CompactionLease) {
        assert!(activate_compaction_for_route(lease, &output(SUMMARY)));
    }

    fn has_bound_state(route: &CodexBoundRoute) -> bool {
        let guard = BOUND_REGISTRY.lock().unwrap();
        guard.as_ref().is_some_and(|registry| {
            route
                .conversation_key()
                .is_some_and(|key| registry.bound_states.contains_key(&key))
        })
    }

    fn pending_anchor_contains(route: &CodexBoundRoute, expected: &str) -> bool {
        let guard = BOUND_REGISTRY.lock().unwrap();
        let Some(state) = guard.as_ref().and_then(|registry| {
            route
                .conversation_key()
                .and_then(|key| registry.bound_states.get(&key))
        }) else {
            return false;
        };
        matches!(
            &state.phase,
            BoundPhase::PendingAnchor { native_history }
                if matches!(native_history.as_slice(), [ResponsesInputItem::Compaction { encrypted_content }] if encrypted_content == expected)
        )
    }

    fn bound_retained_usage_for_tests() -> (usize, usize) {
        let guard = BOUND_REGISTRY.lock().unwrap();
        let registry = guard.as_ref().unwrap();
        let states = registry
            .bound_states
            .values()
            .filter(|state| !matches!(state.phase, BoundPhase::PendingRemote))
            .count();
        let bytes = registry
            .bound_states
            .iter()
            .map(|(route_key, state)| bound_state_size(route_key, state))
            .sum();
        (states, bytes)
    }

    fn native_history_with_state_size(
        route: &CodexBoundRoute,
        state_size: usize,
        portable_summary: Option<&str>,
    ) -> Vec<ResponsesInputItem> {
        let route_key = route.conversation_key().unwrap();
        let lane = route.lane().unwrap();
        let empty = native_history("");
        let base_size =
            bound_state_size_parts(&route_key, lane, "gpt-5.6-sol", &empty, portable_summary, 0);
        assert!(state_size >= base_size);
        let history = native_history(&"x".repeat(state_size - base_size));
        assert_eq!(
            bound_state_size_parts(
                &route_key,
                lane,
                "gpt-5.6-sol",
                &history,
                portable_summary,
                0,
            ),
            state_size
        );
        history
    }

    #[test]
    fn parses_exactly_one_completed_compaction_item() {
        let body = b"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"compaction\",\"encrypted_content\":\"opaque\"}}\n\ndata: {\"type\":\"response.completed\",\"response\":{}}\n\n";
        assert!(matches!(
            parse_compaction_response(body).unwrap(),
            ResponsesInputItem::Compaction { encrypted_content } if encrypted_content == "opaque"
        ));
    }

    #[test]
    fn rejects_incomplete_or_ambiguous_compaction_streams() {
        let incomplete = b"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"compaction\",\"encrypted_content\":\"opaque\"}}\n\n";
        assert!(
            parse_compaction_response(incomplete)
                .unwrap_err()
                .to_string()
                .contains("before response.completed")
        );
        let missing = b"data: {\"type\":\"response.completed\",\"response\":{}}\n\n";
        assert!(
            parse_compaction_response(missing)
                .unwrap_err()
                .to_string()
                .contains("exactly one")
        );
    }

    #[test]
    fn replay_requires_activation_and_wrapped_summary_anchor() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        store_compaction(
            "session",
            "gpt-5.6-sol",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            }],
        );
        let next = request(json!([
            {"type":"additional_tools","role":"developer","tools":[]},
            {"type":"message","role":"developer","content":[{"type":"input_text","text":"instructions"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":format!("<summary>{SUMMARY}</summary>")}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        assert!(apply_compaction_replay(Some("session"), &next).is_none());
        assert!(activate_compaction(
            Some("session"),
            "gpt-5.6-sol",
            &output(&format!(
                "<analysis>summary preparation</analysis>\n<summary>\n{SUMMARY}\n</summary>"
            ))
        ));

        let replay = apply_compaction_replay(Some("session"), &next).unwrap();
        assert!(matches!(
            replay.input[0],
            ResponsesInputItem::AdditionalTools { .. }
        ));
        assert!(
            matches!(replay.input[1], ResponsesInputItem::Message { ref role, .. } if role == "developer")
        );
        assert!(matches!(
            replay.input[2],
            ResponsesInputItem::Compaction { .. }
        ));
        assert_eq!(replay.client_metadata, next.client_metadata);
    }

    #[test]
    fn replay_clears_on_missing_or_duplicate_anchor() {
        let _guard = lock_compaction_registry_for_tests();
        for text in [
            "different conversation without the expected summary".to_string(),
            format!("{SUMMARY} and {SUMMARY}"),
        ] {
            clear_all_compactions_for_tests();
            store_compaction(
                "session",
                "gpt-5.6-sol",
                vec![ResponsesInputItem::Compaction {
                    encrypted_content: "opaque".to_string(),
                }],
            );
            activate_compaction(Some("session"), "gpt-5.6-sol", &output(SUMMARY));
            let changed = request(json!([
                {"type":"message","role":"user","content":[{"type":"input_text","text":text}]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
            ]));
            assert!(apply_compaction_replay(Some("session"), &changed).is_none());
            assert!(apply_compaction_replay(Some("session"), &changed).is_none());
        }
    }

    #[test]
    fn compacted_history_excludes_lite_envelope() {
        let input: Vec<ResponsesInputItem> = serde_json::from_value(json!([
            {"type":"additional_tools","role":"developer","tools":[]},
            {"type":"message","role":"developer","content":[{"type":"input_text","text":"summarize"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"remember me"}]}
        ])).unwrap();
        let (_, conversation) = split_input_envelope(&input);
        let history = build_compacted_history(
            conversation,
            ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            },
        );
        assert_eq!(history.len(), 2);
        assert!(
            matches!(history[0], ResponsesInputItem::Message { ref role, .. } if role == "user")
        );
    }

    #[test]
    fn failed_replay_clears_anchored_state() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        store_compaction(
            "failed-replay",
            "gpt-5.6-sol",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            }],
        );
        activate_compaction(Some("failed-replay"), "gpt-5.6-sol", &output(SUMMARY));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay = apply_compaction_replay(Some("failed-replay"), &next).unwrap();

        abort_compaction_attempt(Some("failed-replay"), false, &replay);

        assert!(apply_compaction_replay(Some("failed-replay"), &next).is_none());
    }

    #[test]
    fn replay_clears_on_model_change() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        store_compaction(
            "session",
            "gpt-5.6-sol",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            }],
        );
        activate_compaction(Some("session"), "gpt-5.6-sol", &output(SUMMARY));
        let mut changed = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        changed.model = "gpt-5.4".to_string();
        assert!(apply_compaction_replay(Some("session"), &changed).is_none());
    }

    #[test]
    fn replay_clears_on_lane_change() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let stored = request(json!([]));
        assert!(store_compaction_for_request(
            "session",
            &stored,
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            }],
        ));
        assert!(activate_compaction_for_request(
            Some("session"),
            &stored,
            &output(SUMMARY),
        ));
        let input = json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]);

        let full = request_for_lane(input.clone(), false);
        assert!(apply_compaction_replay(Some("session"), &full).is_none());

        let lite = request_for_lane(input, true);
        assert!(apply_compaction_replay(Some("session"), &lite).is_none());
    }

    #[test]
    fn activation_clears_on_lane_change() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let lite = request(json!([]));
        assert!(store_compaction_for_request(
            "session",
            &lite,
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            }],
        ));
        let full = request_for_lane(json!([]), false);

        assert!(!activate_compaction_for_request(
            Some("session"),
            &full,
            &output(SUMMARY),
        ));
        assert!(!activate_compaction_for_request(
            Some("session"),
            &lite,
            &output(SUMMARY),
        ));
    }

    #[test]
    fn active_replay_ids_count_toward_bound_state_size_limit() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let route = bound_route(main_lane("replay-size"), "token-a");
        let permit = reserve_compaction_start(route.lane()).unwrap();
        let build = begin_compaction_for_route(&permit, &route, "gpt-5.6-sol").unwrap();
        let history = native_history_with_state_size(
            &route,
            MAX_STATE_BYTES - std::mem::size_of::<u64>() + 1,
            Some(SUMMARY),
        );
        assert!(store_compaction_for_route(&build, history));
        anchor_bound(&build);

        assert!(apply_compaction_replay_for_route(&route, &replay_request()).is_none());
        assert!(has_bound_state(&route));
        drop(permit);
    }

    #[test]
    fn pending_remote_metadata_does_not_evict_anchored_state() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let anchored_route = bound_route(main_lane("metadata-anchored"), "token-anchored");
        let (anchored_permit, anchored_build) = stage_bound(&anchored_route, "anchored");
        anchor_bound(&anchored_build);
        {
            let route_key = anchored_route.conversation_key().unwrap();
            let mut guard = BOUND_REGISTRY.lock().unwrap();
            guard
                .as_mut()
                .unwrap()
                .bound_states
                .get_mut(&route_key)
                .unwrap()
                .updated_at = now_ms().saturating_sub(STATE_TTL_MS + 1);
        }

        let pending_route = bound_route(main_lane("metadata-pending"), "token-pending");
        let pending_permit = reserve_compaction_start(pending_route.lane()).unwrap();
        let pending_build =
            begin_compaction_for_route(&pending_permit, &pending_route, "gpt-5.6-sol").unwrap();

        assert!(has_bound_state(&anchored_route));
        assert!(has_bound_state(&pending_route));
        clear_all_compactions_for_tests();
        drop((pending_build, anchored_permit, pending_permit));
    }

    #[test]
    fn pending_anchor_histories_obey_global_byte_cap() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let active_route = bound_route(main_lane("pending-bytes-active"), "token-active");
        let (active_permit, active_build) = stage_bound(&active_route, "active");
        anchor_bound(&active_build);
        let active_replay =
            apply_compaction_replay_for_route(&active_route, &replay_request()).unwrap();
        let mut permits = vec![active_permit];
        let mut builds = Vec::new();
        let mut rejected = false;

        for index in 0..6 {
            let route = bound_route(
                main_lane(&format!("pending-bytes-{index}")),
                &format!("token-{index}"),
            );
            let permit = reserve_compaction_start(route.lane()).unwrap();
            let build = begin_compaction_for_route(&permit, &route, "gpt-5.6-sol").unwrap();
            let history = native_history_with_state_size(&route, MAX_STATE_BYTES - 1_024, None);
            if store_compaction_for_route(&build, history) {
                let (count, bytes) = bound_retained_usage_for_tests();
                assert!(count <= MAX_STATES);
                assert!(bytes <= MAX_TOTAL_STATE_BYTES);
            } else {
                rejected = true;
                assert!(!has_bound_state(&route));
            }
            permits.push(permit);
            builds.push(build);
        }

        assert!(rejected);
        assert!(has_bound_state(&active_route));
        clear_all_compactions_for_tests();
        drop((active_replay, builds, permits));
    }

    #[test]
    fn pending_anchor_count_is_capped_without_evicting_active_replay() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let active_route = bound_route(main_lane("pending-count-active"), "token-active");
        let (active_permit, active_build) = stage_bound(&active_route, "active");
        anchor_bound(&active_build);
        let active_replay =
            apply_compaction_replay_for_route(&active_route, &replay_request()).unwrap();
        let mut permits = vec![active_permit];
        let mut builds = Vec::new();

        for index in 0..MAX_STATES - 1 {
            let route = bound_route(
                main_lane(&format!("pending-count-{index}")),
                &format!("token-{index}"),
            );
            let permit = reserve_compaction_start(route.lane()).unwrap();
            let build = begin_compaction_for_route(&permit, &route, "gpt-5.6-sol").unwrap();
            assert!(store_compaction_for_route(
                &build,
                native_history("pending")
            ));
            permits.push(permit);
            builds.push(build);
        }
        assert_eq!(bound_retained_usage_for_tests().0, MAX_STATES);

        let rejected_route = bound_route(main_lane("pending-count-rejected"), "token-rejected");
        let rejected_permit = reserve_compaction_start(rejected_route.lane()).unwrap();
        let rejected_build =
            begin_compaction_for_route(&rejected_permit, &rejected_route, "gpt-5.6-sol").unwrap();
        assert!(has_bound_state(&active_route));
        assert!(!store_compaction_for_route(
            &rejected_build,
            native_history("rejected")
        ));
        assert!(!has_bound_state(&rejected_route));
        assert_eq!(bound_retained_usage_for_tests().0, MAX_STATES);
        assert!(has_bound_state(&active_route));

        clear_all_compactions_for_tests();
        drop((
            active_replay,
            rejected_build,
            builds,
            rejected_permit,
            permits,
        ));
    }

    #[test]
    fn active_replay_ids_obey_global_byte_cap_with_protected_builds() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let active_route = bound_route(main_lane("replay-bytes-active"), "token-active");
        let (active_permit, active_build) = stage_bound(&active_route, "active");
        anchor_bound(&active_build);
        let active_replay =
            apply_compaction_replay_for_route(&active_route, &replay_request()).unwrap();
        let (_, active_bytes) = bound_retained_usage_for_tests();
        let replay_id_bytes = std::mem::size_of::<u64>();
        let mut remaining = MAX_TOTAL_STATE_BYTES - active_bytes - (replay_id_bytes - 1);
        let parts = remaining.div_ceil(MAX_STATE_BYTES);
        let mut permits = vec![active_permit];
        let mut builds = Vec::new();

        for index in 0..parts {
            let parts_left = parts - index;
            let state_size = remaining.div_ceil(parts_left);
            let route = bound_route(
                main_lane(&format!("replay-bytes-build-{index}")),
                &format!("token-{index}"),
            );
            let permit = reserve_compaction_start(route.lane()).unwrap();
            let build = begin_compaction_for_route(&permit, &route, "gpt-5.6-sol").unwrap();
            let history = native_history_with_state_size(&route, state_size, None);
            assert!(store_compaction_for_route(&build, history));
            remaining -= state_size;
            permits.push(permit);
            builds.push(build);
        }
        assert_eq!(remaining, 0);
        assert_eq!(
            bound_retained_usage_for_tests().1,
            MAX_TOTAL_STATE_BYTES - (replay_id_bytes - 1)
        );

        assert!(apply_compaction_replay_for_route(&active_route, &replay_request()).is_none());
        assert!(has_bound_state(&active_route));
        assert_eq!(
            bound_retained_usage_for_tests().1,
            MAX_TOTAL_STATE_BYTES - (replay_id_bytes - 1)
        );

        clear_all_compactions_for_tests();
        drop((active_replay, builds, permits));
    }

    #[test]
    fn stale_store_cannot_replace_newer_bound_build() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let route = bound_route(main_lane("stale-store"), "token-a");
        let older_permit = reserve_compaction_start(route.lane()).unwrap();
        let older = begin_compaction_for_route(&older_permit, &route, "gpt-5.6-sol").unwrap();
        let (newer_permit, newer) = stage_bound(&route, "newer");

        assert!(!store_compaction_for_route(&older, native_history("older")));
        assert!(pending_anchor_contains(&route, "newer"));
        drop((older, newer, older_permit, newer_permit));
    }

    #[test]
    fn stale_activation_cannot_anchor_newer_history() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let route = bound_route(main_lane("stale-activation"), "token-a");
        let (older_permit, older) = stage_bound(&route, "older");
        let (newer_permit, newer) = stage_bound(&route, "newer");

        assert!(!activate_compaction_for_route(&older, &output(SUMMARY)));
        assert!(pending_anchor_contains(&route, "newer"));
        anchor_bound(&newer);
        assert!(has_bound_state(&route));
        drop((older_permit, newer_permit));
    }

    #[test]
    fn stale_abort_and_drop_cannot_remove_newer_revision() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let route = bound_route(main_lane("stale-abort"), "token-a");
        let (older_permit, older) = stage_bound(&route, "older");
        let stale_clone = older.clone();
        let (newer_permit, newer) = stage_bound(&route, "newer");

        abort_compaction_for_route(&older);
        drop(stale_clone);
        assert!(pending_anchor_contains(&route, "newer"));
        drop((newer, older_permit, newer_permit));
    }

    #[test]
    fn current_build_drop_removes_pending_remote_or_anchor() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let remote_route = bound_route(main_lane("drop-remote"), "token-a");
        let remote_permit = reserve_compaction_start(remote_route.lane()).unwrap();
        let remote =
            begin_compaction_for_route(&remote_permit, &remote_route, "gpt-5.6-sol").unwrap();
        drop(remote);
        assert!(!has_bound_state(&remote_route));

        let anchor_route = bound_route(main_lane("drop-anchor"), "token-a");
        let (anchor_permit, anchor) = stage_bound(&anchor_route, "pending-anchor");
        drop(anchor);
        assert!(!has_bound_state(&anchor_route));
        drop((remote_permit, anchor_permit));
    }

    #[test]
    fn permit_drop_removes_canceled_pending_start() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let lane = main_lane("permit-drop");
        let permit = reserve_compaction_start(Some(lane)).unwrap();
        let generation = permit.generation;
        drop(permit);

        let guard = BOUND_REGISTRY.lock().unwrap();
        let registry = guard.as_ref().unwrap();
        assert!(
            registry
                .lanes
                .get(&lane)
                .is_none_or(|state| !state.pending_generations.contains(&generation))
        );
    }

    #[test]
    fn lease_cleanup_waits_for_last_clone() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let route = bound_route(main_lane("lease-clone"), "token-a");
        let permit = reserve_compaction_start(route.lane()).unwrap();
        let lease = begin_compaction_for_route(&permit, &route, "gpt-5.6-sol").unwrap();
        let clone = lease.clone();

        drop(lease);
        assert!(has_bound_state(&route));
        drop(clone);
        assert!(!has_bound_state(&route));
        drop(permit);
    }

    #[test]
    fn same_generation_can_rebind_to_new_route() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let lane = main_lane("same-generation-rebind");
        let route_a = bound_route(lane, "token-a");
        let route_b = bound_route(lane, "token-b");
        let permit = reserve_compaction_start(Some(lane)).unwrap();
        let stale = begin_compaction_for_route(&permit, &route_a, "gpt-5.6-sol").unwrap();
        let current = begin_compaction_for_route(&permit, &route_b, "gpt-5.6-sol").unwrap();

        assert!(!has_bound_state(&route_a));
        assert!(has_bound_state(&route_b));
        drop(stale);
        assert!(has_bound_state(&route_b));
        drop((current, permit));
    }

    #[test]
    fn older_permit_cannot_rebind_over_newer_route_state() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let lane = main_lane("older-rebind");
        let route_a = bound_route(lane, "token-a");
        let route_b = bound_route(lane, "token-b");
        let older = reserve_compaction_start(Some(lane)).unwrap();
        let newer = reserve_compaction_start(Some(lane)).unwrap();
        let current = begin_compaction_for_route(&newer, &route_b, "gpt-5.6-sol").unwrap();

        assert!(begin_compaction_for_route(&older, &route_a, "gpt-5.6-sol").is_none());
        assert!(has_bound_state(&route_b));
        drop((current, older, newer));
    }

    #[test]
    fn aborted_newer_start_still_rejects_older_permit() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let lane = main_lane("aborted-newer");
        let route = bound_route(lane, "token-a");
        let older = reserve_compaction_start(Some(lane)).unwrap();
        let newer = reserve_compaction_start(Some(lane)).unwrap();
        let newer_build = begin_compaction_for_route(&newer, &route, "gpt-5.6-sol").unwrap();
        abort_compaction_for_route(&newer_build);
        drop(newer);

        assert!(begin_compaction_for_route(&older, &route, "gpt-5.6-sol").is_none());
        drop(older);
    }

    #[test]
    fn route_mismatch_is_non_destructive() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let lane = main_lane("route-mismatch");
        let route_a = bound_route(lane, "token-a");
        let route_b = bound_route(lane, "token-b");
        let (permit, build) = stage_bound(&route_a, "native");
        anchor_bound(&build);

        assert!(apply_compaction_replay_for_route(&route_b, &replay_request()).is_none());
        assert!(has_bound_state(&route_a));
        let replay = apply_compaction_replay_for_route(&route_a, &replay_request()).unwrap();
        assert!(activate_compaction_for_route(&replay.lease, &[]));
        drop(permit);
    }

    #[test]
    fn main_and_sibling_agent_lanes_never_supersede_each_other() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let main_route = bound_route(main_lane("shared-session"), "token-a");
        let agent_route = bound_route(agent_lane("shared-session", "agent-a"), "token-a");
        let (main_permit, main_build) = stage_bound(&main_route, "main");
        let (agent_permit, agent_build) = stage_bound(&agent_route, "agent");
        anchor_bound(&main_build);
        anchor_bound(&agent_build);

        let main_newer_permit = reserve_compaction_start(main_route.lane()).unwrap();
        let main_newer =
            begin_compaction_for_route(&main_newer_permit, &main_route, "gpt-5.6-sol").unwrap();
        assert!(has_bound_state(&agent_route));
        let agent_replay =
            apply_compaction_replay_for_route(&agent_route, &replay_request()).unwrap();
        assert!(activate_compaction_for_route(&agent_replay.lease, &[]));
        drop((main_newer, main_permit, agent_permit, main_newer_permit));
    }

    #[test]
    fn lane_clear_revokes_only_that_lane_and_its_pending_permits() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let lane_a = main_lane("clear-a");
        let lane_b = main_lane("clear-b");
        let route_a = bound_route(lane_a, "token-a");
        let route_b = bound_route(lane_b, "token-a");
        let stale = reserve_compaction_start(Some(lane_a)).unwrap();
        let (permit_b, build_b) = stage_bound(&route_b, "lane-b");
        anchor_bound(&build_b);

        clear_compactions_for_lane(lane_a);
        assert!(begin_compaction_for_route(&stale, &route_a, "gpt-5.6-sol").is_none());
        assert!(has_bound_state(&route_b));
        let fresh = reserve_compaction_start(Some(lane_a)).unwrap();
        let fresh_build = begin_compaction_for_route(&fresh, &route_a, "gpt-5.6-sol").unwrap();
        assert!(has_bound_state(&route_a));
        drop((stale, fresh, fresh_build, permit_b));
    }

    #[test]
    fn replay_model_and_anchor_mismatch_are_non_destructive() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let route = bound_route(main_lane("replay-mismatch"), "token-a");
        let (permit, build) = stage_bound(&route, "native");
        anchor_bound(&build);

        let mut wrong_model = replay_request();
        wrong_model.model = "gpt-5.4".to_string();
        assert!(apply_compaction_replay_for_route(&route, &wrong_model).is_none());
        let wrong_anchor = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":"a different summary anchor"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        assert!(apply_compaction_replay_for_route(&route, &wrong_anchor).is_none());
        let full_protocol =
            request_for_lane(serde_json::to_value(replay_request().input).unwrap(), false);
        assert!(apply_compaction_replay_for_route(&route, &full_protocol).is_none());
        assert!(has_bound_state(&route));

        let replay = apply_compaction_replay_for_route(&route, &replay_request()).unwrap();
        assert!(activate_compaction_for_route(&replay.lease, &[]));
        drop(permit);
    }

    #[test]
    fn replay_failure_then_peer_success_preserves_state() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let route = bound_route(main_lane("failure-success"), "token-a");
        let (permit, build) = stage_bound(&route, "native");
        anchor_bound(&build);
        let failed = apply_compaction_replay_for_route(&route, &replay_request()).unwrap();
        let succeeded = apply_compaction_replay_for_route(&route, &replay_request()).unwrap();

        abort_compaction_for_route(&failed.lease);
        assert!(activate_compaction_for_route(&succeeded.lease, &[]));
        assert!(has_bound_state(&route));
        drop(permit);
    }

    #[test]
    fn replay_success_then_stale_peer_abort_preserves_state() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let route = bound_route(main_lane("success-abort"), "token-a");
        let (permit, build) = stage_bound(&route, "native");
        anchor_bound(&build);
        let succeeded = apply_compaction_replay_for_route(&route, &replay_request()).unwrap();
        let stale = apply_compaction_replay_for_route(&route, &replay_request()).unwrap();

        assert!(activate_compaction_for_route(&succeeded.lease, &[]));
        abort_compaction_for_route(&stale.lease);
        assert!(has_bound_state(&route));
        drop(permit);
    }

    #[test]
    fn all_failed_concurrent_replays_remove_the_revision() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let route = bound_route(main_lane("all-failed"), "token-a");
        let (permit, build) = stage_bound(&route, "native");
        anchor_bound(&build);
        let first = apply_compaction_replay_for_route(&route, &replay_request()).unwrap();
        let second = apply_compaction_replay_for_route(&route, &replay_request()).unwrap();

        abort_compaction_for_route(&first.lease);
        drop(second);
        assert!(!has_bound_state(&route));
        drop(permit);
    }

    #[test]
    fn stale_replay_drop_cannot_remove_newer_build() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let route = bound_route(main_lane("stale-replay"), "token-a");
        let (older_permit, build) = stage_bound(&route, "native");
        anchor_bound(&build);
        let replay = apply_compaction_replay_for_route(&route, &replay_request()).unwrap();
        let newer_permit = reserve_compaction_start(route.lane()).unwrap();
        let newer = begin_compaction_for_route(&newer_permit, &route, "gpt-5.6-sol").unwrap();

        drop(replay);
        assert!(has_bound_state(&route));
        drop((newer, older_permit, newer_permit));
    }

    #[test]
    fn eviction_skips_build_and_active_replay_leases() {
        let _guard = lock_compaction_registry_for_tests();
        clear_all_compactions_for_tests();
        let active_route = bound_route(main_lane("eviction-active"), "token-active");
        let (active_permit, active_build) = stage_bound(&active_route, "active");
        anchor_bound(&active_build);
        let active_replay =
            apply_compaction_replay_for_route(&active_route, &replay_request()).unwrap();

        let build_route = bound_route(main_lane("eviction-build"), "token-build");
        let (build_permit, protected_build) = stage_bound(&build_route, "protected-build");
        let mut oldest_inactive_route = None;
        for index in 0..MAX_STATES - 2 {
            let route = bound_route(
                main_lane(&format!("eviction-inactive-{index}")),
                &format!("token-{index}"),
            );
            let (permit, build) = stage_bound(&route, "inactive");
            anchor_bound(&build);
            if oldest_inactive_route.is_none() {
                oldest_inactive_route = Some(route.clone());
            }
            drop(permit);
        }

        let oldest_inactive_route = oldest_inactive_route.unwrap();
        let candidate_route = bound_route(main_lane("eviction-candidate"), "token-candidate");
        let (candidate_permit, candidate) = stage_bound(&candidate_route, "candidate");
        assert!(!has_bound_state(&oldest_inactive_route));
        assert!(has_bound_state(&active_route));
        assert!(pending_anchor_contains(&build_route, "protected-build"));
        assert!(pending_anchor_contains(&candidate_route, "candidate"));

        anchor_bound(&candidate);
        assert!(has_bound_state(&active_route));
        assert!(pending_anchor_contains(&build_route, "protected-build"));
        assert!(has_bound_state(&candidate_route));

        clear_all_compactions_for_tests();
        drop((
            active_replay,
            protected_build,
            candidate,
            active_permit,
            build_permit,
            candidate_permit,
        ));
    }

    #[test]
    fn retained_history_obeys_token_budget_at_utf8_boundary() {
        let text = "é".repeat((RETAINED_MESSAGE_TOKEN_BUDGET as usize + 10) * 4);
        let input: Vec<ResponsesInputItem> = serde_json::from_value(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":text}]}
        ]))
        .unwrap();
        let history = build_compacted_history(
            &input,
            ResponsesInputItem::Compaction {
                encrypted_content: "opaque".to_string(),
            },
        );
        let ResponsesInputItem::Message { content, .. } = &history[0] else {
            panic!("expected retained message");
        };
        let ResponsesContentPart::InputText { text } = &content[0] else {
            panic!("expected retained text");
        };
        assert!(text.len() <= RETAINED_MESSAGE_TOKEN_BUDGET as usize * 4);
        assert!(std::str::from_utf8(text.as_bytes()).is_ok());
    }
}
