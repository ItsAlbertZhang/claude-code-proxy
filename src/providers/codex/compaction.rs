use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::anthropic::sse::parse_sse_events;
use crate::provider::RequestContext;
use crate::providers::codex::client::{CodexConversationRoute, CodexError, CodexHttpClient};

use super::translate::request::{
    ResponsesContentPart, ResponsesInputItem, ResponsesRequest, is_compact_message_text,
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
    PendingRemote,
    PendingAnchor {
        native_history: Vec<ResponsesInputItem>,
    },
    Anchored {
        native_history: Vec<ResponsesInputItem>,
        portable_summary: String,
        active_replays: HashSet<String>,
    },
}

struct CompactionState {
    lane_token: String,
    operation_id: String,
    start_order: u64,
    revision: u64,
    model: String,
    phase: CompactionPhase,
    updated_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompactionLeaseKind {
    Build,
    Replay,
}

#[derive(Debug)]
struct CompactionLeaseCleanup {
    session_id: String,
    operation_id: String,
    revision: u64,
    kind: CompactionLeaseKind,
    armed: AtomicBool,
}

impl Drop for CompactionLeaseCleanup {
    fn drop(&mut self) {
        if !self.armed.swap(false, Ordering::AcqRel) {
            return;
        }
        let mut guard = REGISTRY.lock().unwrap();
        if let Some(registry) = guard.as_mut() {
            cleanup_lease_attempt(
                registry,
                &self.session_id,
                &self.operation_id,
                self.revision,
                self.kind,
            );
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactionLease {
    session_id: String,
    operation_id: String,
    revision: u64,
    kind: CompactionLeaseKind,
    cleanup: Arc<CompactionLeaseCleanup>,
}

impl PartialEq for CompactionLease {
    fn eq(&self, other: &Self) -> bool {
        self.session_id == other.session_id
            && self.operation_id == other.operation_id
            && self.revision == other.revision
            && self.kind == other.kind
    }
}

impl Eq for CompactionLease {}

impl CompactionLease {
    fn new(session_id: &str, operation_id: &str, revision: u64, kind: CompactionLeaseKind) -> Self {
        Self {
            session_id: session_id.to_string(),
            operation_id: operation_id.to_string(),
            revision,
            kind,
            cleanup: Arc::new(CompactionLeaseCleanup {
                session_id: session_id.to_string(),
                operation_id: operation_id.to_string(),
                revision,
                kind,
                armed: AtomicBool::new(true),
            }),
        }
    }

    fn disarm_cleanup(&self) {
        self.cleanup.armed.store(false, Ordering::Release);
    }
}

pub struct CompactionReplay {
    pub request: ResponsesRequest,
    pub lease: CompactionLease,
}

#[derive(Default)]
struct CompactionRegistry {
    states: HashMap<String, CompactionState>,
    pending_starts: HashMap<(String, String), u64>,
    lane_start_orders: HashMap<String, u64>,
    next_revision: u64,
    total_bytes: usize,
}

#[derive(Debug)]
pub struct CompactionStartPermit {
    lane_token: String,
    operation_id: String,
    start_order: u64,
    active: bool,
}

impl Drop for CompactionStartPermit {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut guard = REGISTRY.lock().unwrap();
        if let Some(registry) = guard.as_mut() {
            let key = (self.lane_token.clone(), self.operation_id.clone());
            if registry.pending_starts.get(&key) == Some(&self.start_order) {
                registry.pending_starts.remove(&key);
            }
            cleanup_lane_start_order(registry, &self.lane_token);
        }
    }
}

static REGISTRY: Mutex<Option<CompactionRegistry>> = Mutex::new(None);

pub async fn request_compaction(
    client: &CodexHttpClient,
    route: &CodexConversationRoute,
    request: &ResponsesRequest,
    ctx: &RequestContext,
) -> Result<Vec<ResponsesInputItem>, CompactionError> {
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

    let response = client
        .post_codex_bound(route, &compaction_request, ctx, None)
        .await
        .map_err(CompactionError::Upstream)?;
    let compaction = parse_compaction_response(&response.body)?;
    Ok(build_compacted_history(&conversation, compaction))
}

pub fn reserve_compaction_start(
    lane_token: Option<&str>,
    operation_id: &str,
) -> Option<CompactionStartPermit> {
    let lane_token = lane_token?;
    let key = (lane_token.to_string(), operation_id.to_string());
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.get_or_insert_with(CompactionRegistry::default);
    if registry.pending_starts.contains_key(&key) {
        return None;
    }
    registry.next_revision = registry.next_revision.wrapping_add(1).max(1);
    let start_order = registry.next_revision;
    registry.pending_starts.insert(key, start_order);
    registry
        .lane_start_orders
        .insert(lane_token.to_string(), start_order);
    Some(CompactionStartPermit {
        lane_token: lane_token.to_string(),
        operation_id: operation_id.to_string(),
        start_order,
        active: true,
    })
}

pub fn begin_compaction(
    session_id: Option<&str>,
    model: &str,
    operation_id: &str,
) -> Option<CompactionLease> {
    begin_compaction_for_lane(session_id, session_id, model, operation_id)
}

pub fn begin_compaction_for_lane(
    session_id: Option<&str>,
    lane_token: Option<&str>,
    model: &str,
    operation_id: &str,
) -> Option<CompactionLease> {
    let session_id = session_id?;
    let lane_token = lane_token?;
    let now = now_ms();
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.get_or_insert_with(CompactionRegistry::default);
    registry.next_revision = registry.next_revision.wrapping_add(1).max(1);
    let start_order = registry.next_revision;
    begin_compaction_locked(
        registry,
        session_id,
        lane_token,
        model,
        operation_id,
        start_order,
        now,
    )
}

pub fn begin_compaction_with_permit(
    session_id: Option<&str>,
    model: &str,
    mut permit: CompactionStartPermit,
) -> Option<CompactionLease> {
    let session_id = session_id?;
    let now = now_ms();
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.get_or_insert_with(CompactionRegistry::default);
    let key = (permit.lane_token.clone(), permit.operation_id.clone());
    if registry.pending_starts.get(&key) != Some(&permit.start_order) {
        permit.active = false;
        return None;
    }
    registry.pending_starts.remove(&key);
    permit.active = false;
    let lease = begin_compaction_locked(
        registry,
        session_id,
        &permit.lane_token,
        model,
        &permit.operation_id,
        permit.start_order,
        now,
    );
    cleanup_lane_start_order(registry, &permit.lane_token);
    lease
}

fn begin_compaction_locked(
    registry: &mut CompactionRegistry,
    session_id: &str,
    lane_token: &str,
    model: &str,
    operation_id: &str,
    start_order: u64,
    now: u64,
) -> Option<CompactionLease> {
    if registry
        .lane_start_orders
        .get(lane_token)
        .is_some_and(|latest| *latest > start_order)
    {
        cleanup_lane_start_order(registry, lane_token);
        return None;
    }
    registry
        .lane_start_orders
        .insert(lane_token.to_string(), start_order);
    evict_states(registry, now);
    if registry
        .states
        .get(session_id)
        .is_some_and(|state| state.start_order >= start_order)
    {
        cleanup_lane_start_order(registry, lane_token);
        return None;
    }
    registry.next_revision = registry.next_revision.wrapping_add(1).max(1);
    let lease = CompactionLease::new(
        session_id,
        operation_id,
        registry.next_revision,
        CompactionLeaseKind::Build,
    );
    let replaced = registry.states.insert(
        session_id.to_string(),
        CompactionState {
            lane_token: lane_token.to_string(),
            operation_id: operation_id.to_string(),
            start_order,
            revision: lease.revision,
            model: model.to_string(),
            phase: CompactionPhase::PendingRemote,
            updated_at: now,
        },
    );
    registry
        .lane_start_orders
        .insert(lane_token.to_string(), start_order);
    if let Some(replaced) = replaced
        && replaced.lane_token != lane_token
    {
        cleanup_lane_start_order(registry, &replaced.lane_token);
    }
    evict_states(registry, now);
    if registry.states.get(session_id).is_some_and(|state| {
        state.revision == lease.revision && state.operation_id == lease.operation_id
    }) {
        Some(lease)
    } else {
        lease.disarm_cleanup();
        None
    }
}

pub fn store_compaction(
    lease: &CompactionLease,
    model: &str,
    native_history: Vec<ResponsesInputItem>,
) -> bool {
    let now = now_ms();
    let mut guard = REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return false;
    };
    evict_states(registry, now);
    let Some(state) = registry.states.get_mut(&lease.session_id) else {
        return false;
    };
    if !build_lease_matches(state, lease)
        || state.model != model
        || !matches!(state.phase, CompactionPhase::PendingRemote)
    {
        return false;
    }
    state.phase = CompactionPhase::PendingAnchor { native_history };
    state.updated_at = now;
    if state_size(&lease.session_id, state) > MAX_STATE_BYTES {
        remove_if_current_build(registry, lease);
        return false;
    }
    evict_states(registry, now);
    registry
        .states
        .get(&lease.session_id)
        .is_some_and(|state| build_lease_matches(state, lease))
}

pub fn activate_compaction(
    lease: Option<&CompactionLease>,
    model: &str,
    output: &[ResponsesInputItem],
) -> bool {
    let Some(lease) = lease else {
        return false;
    };
    lease.disarm_cleanup();
    let now = now_ms();

    match lease.kind {
        CompactionLeaseKind::Build => {
            let Some(portable_summary) = portable_summary_text(output) else {
                abort_compaction_attempt(Some(lease));
                return false;
            };
            let mut guard = REGISTRY.lock().unwrap();
            let Some(registry) = guard.as_mut() else {
                return false;
            };
            evict_states(registry, now);
            registry.next_revision = registry.next_revision.wrapping_add(1).max(1);
            let anchored_revision = registry.next_revision;
            let Some(state) = registry.states.get_mut(&lease.session_id) else {
                return false;
            };
            if !build_lease_matches(state, lease)
                || state.model != model
                || !matches!(state.phase, CompactionPhase::PendingAnchor { .. })
            {
                return false;
            }
            let CompactionPhase::PendingAnchor { native_history } =
                std::mem::replace(&mut state.phase, CompactionPhase::PendingRemote)
            else {
                unreachable!("phase checked above")
            };
            state.phase = CompactionPhase::Anchored {
                native_history,
                portable_summary,
                active_replays: HashSet::new(),
            };
            state.revision = anchored_revision;
            state.updated_at = now;
            if state_size(&lease.session_id, state) > MAX_STATE_BYTES {
                remove_compaction_state(registry, &lease.session_id);
                return false;
            }
            evict_states(registry, now);
            registry
                .states
                .get(&lease.session_id)
                .is_some_and(|state| state.revision == anchored_revision)
        }
        CompactionLeaseKind::Replay => {
            let mut guard = REGISTRY.lock().unwrap();
            let Some(registry) = guard.as_mut() else {
                return false;
            };
            evict_states(registry, now);
            registry.next_revision = registry.next_revision.wrapping_add(1).max(1);
            let settled_revision = registry.next_revision;
            let Some(state) = registry.states.get_mut(&lease.session_id) else {
                return false;
            };
            if state.revision != lease.revision || state.model != model {
                return false;
            }
            let CompactionPhase::Anchored { active_replays, .. } = &mut state.phase else {
                return false;
            };
            if !active_replays.remove(&lease.operation_id) {
                return false;
            }
            active_replays.clear();
            state.revision = settled_revision;
            state.updated_at = now;
            true
        }
    }
}

pub fn apply_compaction_replay(
    session_id: Option<&str>,
    request: &ResponsesRequest,
    operation_id: &str,
) -> Option<CompactionReplay> {
    let session_id = session_id?;
    if operation_id.is_empty() {
        return None;
    }
    let now = now_ms();
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.as_mut()?;
    evict_states(registry, now);
    let state = registry.states.get_mut(session_id)?;
    if state.model != request.model {
        return None;
    }
    let CompactionPhase::Anchored {
        native_history,
        portable_summary,
        active_replays,
    } = &mut state.phase
    else {
        return None;
    };

    let (envelope, conversation) = split_input_envelope(&request.input);
    let summary_item = conversation.first()?;
    let text = message_text(summary_item)?;
    if text.match_indices(portable_summary.as_str()).count() != 1 || conversation.len() == 1 {
        return None;
    }

    let mut replay = request.clone();
    replay.input = envelope
        .iter()
        .cloned()
        .chain(native_history.iter().cloned())
        .chain(conversation[1..].iter().cloned())
        .collect();
    if serialized_size(&replay.input) > MAX_STATE_BYTES
        || !active_replays.insert(operation_id.to_string())
    {
        return None;
    }
    if state_size(session_id, state) > MAX_STATE_BYTES {
        if let CompactionPhase::Anchored { active_replays, .. } = &mut state.phase {
            active_replays.remove(operation_id);
        }
        return None;
    }
    state.updated_at = now;
    Some(CompactionReplay {
        request: replay,
        lease: CompactionLease::new(
            session_id,
            operation_id,
            state.revision,
            CompactionLeaseKind::Replay,
        ),
    })
}

pub fn abort_compaction_attempt(lease: Option<&CompactionLease>) {
    let Some(lease) = lease else {
        return;
    };
    lease.disarm_cleanup();
    let mut guard = REGISTRY.lock().unwrap();
    let Some(registry) = guard.as_mut() else {
        return;
    };
    cleanup_lease_attempt(
        registry,
        &lease.session_id,
        &lease.operation_id,
        lease.revision,
        lease.kind,
    );
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
        remove_compaction_state(registry, session_id);
    }
}

pub fn clear_compactions_for_lane(lane_token: &str) {
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.get_or_insert_with(CompactionRegistry::default);
    registry
        .states
        .retain(|_, state| state.lane_token != lane_token);
    registry
        .pending_starts
        .retain(|(pending_lane, _), _| pending_lane != lane_token);
    registry.lane_start_orders.remove(lane_token);
    update_total_bytes(registry);
}

fn cleanup_lease_attempt(
    registry: &mut CompactionRegistry,
    session_id: &str,
    operation_id: &str,
    revision: u64,
    kind: CompactionLeaseKind,
) {
    match kind {
        CompactionLeaseKind::Build => {
            remove_if_current_build_identity(registry, session_id, operation_id, revision);
        }
        CompactionLeaseKind::Replay => {
            let remove_state = {
                let Some(state) = registry.states.get_mut(session_id) else {
                    return;
                };
                if state.revision != revision {
                    return;
                }
                let CompactionPhase::Anchored { active_replays, .. } = &mut state.phase else {
                    return;
                };
                if !active_replays.remove(operation_id) {
                    return;
                }
                if active_replays.is_empty() {
                    true
                } else {
                    state.updated_at = now_ms();
                    false
                }
            };
            if remove_state {
                remove_compaction_state(registry, session_id);
            } else {
                update_total_bytes(registry);
            }
        }
    }
}

fn build_identity_matches(state: &CompactionState, operation_id: &str, revision: u64) -> bool {
    state.operation_id == operation_id && state.revision == revision
}

fn build_lease_matches(state: &CompactionState, lease: &CompactionLease) -> bool {
    lease.kind == CompactionLeaseKind::Build
        && build_identity_matches(state, &lease.operation_id, lease.revision)
}

fn remove_if_current_build(registry: &mut CompactionRegistry, lease: &CompactionLease) -> bool {
    remove_if_current_build_identity(
        registry,
        &lease.session_id,
        &lease.operation_id,
        lease.revision,
    )
}

fn remove_if_current_build_identity(
    registry: &mut CompactionRegistry,
    session_id: &str,
    operation_id: &str,
    revision: u64,
) -> bool {
    let current = registry
        .states
        .get(session_id)
        .is_some_and(|state| build_identity_matches(state, operation_id, revision));
    current && remove_compaction_state(registry, session_id)
}

fn remove_compaction_state(registry: &mut CompactionRegistry, session_id: &str) -> bool {
    let Some(state) = registry.states.remove(session_id) else {
        return false;
    };
    update_total_bytes(registry);
    cleanup_lane_start_order(registry, &state.lane_token);
    true
}

fn cleanup_lane_start_order(registry: &mut CompactionRegistry, lane_token: &str) {
    let lane_is_active = registry
        .states
        .values()
        .any(|state| state.lane_token == lane_token)
        || registry
            .pending_starts
            .keys()
            .any(|(pending_lane, _)| pending_lane == lane_token);
    if !lane_is_active {
        registry.lane_start_orders.remove(lane_token);
    }
}

fn cleanup_lane_start_orders(registry: &mut CompactionRegistry) {
    let active_lanes = registry
        .states
        .values()
        .map(|state| state.lane_token.clone())
        .chain(
            registry
                .pending_starts
                .keys()
                .map(|(lane_token, _)| lane_token.clone()),
        )
        .collect::<HashSet<_>>();
    registry
        .lane_start_orders
        .retain(|lane_token, _| active_lanes.contains(lane_token));
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
    let (summary_len, native_history_len) = match &state.phase {
        CompactionPhase::PendingRemote => (0, 0),
        CompactionPhase::PendingAnchor { native_history } => (0, serialized_size(native_history)),
        CompactionPhase::Anchored {
            native_history,
            portable_summary,
            active_replays,
        } => (
            portable_summary.len() + active_replays.iter().map(String::len).sum::<usize>(),
            serialized_size(native_history),
        ),
    };
    session_id.len()
        + state.lane_token.len()
        + state.operation_id.len()
        + state.model.len()
        + summary_len
        + native_history_len
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
    cleanup_lane_start_orders(registry);
}

pub fn clear_all_compactions_for_tests() {
    *REGISTRY.lock().unwrap() = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SUMMARY: &str =
        "portable summary with enough detail to identify this compacted conversation";
    const STALE_SUMMARY: &str =
        "stale portable summary from an older overlapping compaction attempt";
    static TEST_REGISTRY_LOCK: Mutex<()> = Mutex::new(());

    fn request(input: serde_json::Value) -> ResponsesRequest {
        serde_json::from_value(json!({
            "model": "gpt-5.6-sol",
            "input": input,
            "store": false,
            "stream": true,
            "parallel_tool_calls": false,
            "client_metadata": {"lite":"true"},
            "text": {"verbosity":"low"}
        }))
        .unwrap()
    }

    fn output(text: &str) -> Vec<ResponsesInputItem> {
        serde_json::from_value(json!([{
            "type":"message",
            "role":"assistant",
            "content":[{"type":"output_text","text":text}]
        }]))
        .unwrap()
    }

    fn stage(session_id: &str, operation_id: &str, encrypted_content: &str) -> CompactionLease {
        let lease = begin_compaction(Some(session_id), "gpt-5.6-sol", operation_id).unwrap();
        assert!(store_compaction(
            &lease,
            "gpt-5.6-sol",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: encrypted_content.to_string(),
            }],
        ));
        lease
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
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let lease = stage("session", "operation-1", "opaque");
        let next = request(json!([
            {"type":"additional_tools","role":"developer","tools":[]},
            {"type":"message","role":"developer","content":[{"type":"input_text","text":"instructions"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":format!("<summary>{SUMMARY}</summary>")}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        assert!(apply_compaction_replay(Some("session"), &next, "replay").is_none());
        assert!(activate_compaction(
            Some(&lease),
            "gpt-5.6-sol",
            &output(&format!(
                "<analysis>summary preparation</analysis>\n<summary>\n{SUMMARY}\n</summary>"
            ))
        ));

        let replay = apply_compaction_replay(Some("session"), &next, "replay").unwrap();
        assert!(matches!(
            replay.request.input[0],
            ResponsesInputItem::AdditionalTools { .. }
        ));
        assert!(
            matches!(replay.request.input[1], ResponsesInputItem::Message { ref role, .. } if role == "developer")
        );
        assert!(matches!(
            replay.request.input[2],
            ResponsesInputItem::Compaction { .. }
        ));
        assert_eq!(replay.request.client_metadata, next.client_metadata);
    }

    #[test]
    fn stale_activation_cannot_anchor_newer_native_history() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = stage("session", "older", "older-native-history");
        let newer = stage("session", "newer", "newer-native-history");

        assert!(!activate_compaction(
            Some(&older),
            "gpt-5.6-sol",
            &output(STALE_SUMMARY),
        ));
        assert!(activate_compaction(
            Some(&newer),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
    }

    #[test]
    fn stale_store_cannot_replace_newer_compaction() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = begin_compaction(Some("session"), "gpt-5.6-sol", "older").unwrap();
        let newer = stage("session", "newer", "newer-native-history");

        assert!(!store_compaction(
            &older,
            "gpt-5.6-sol",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "late-older-native-history".to_string(),
            }],
        ));
        assert!(activate_compaction(
            Some(&newer),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay = apply_compaction_replay(Some("session"), &next, "replay").unwrap();
        assert!(replay.request.input.iter().any(|item| matches!(
            item,
            ResponsesInputItem::Compaction { encrypted_content }
                if encrypted_content == "newer-native-history"
        )));
    }

    #[test]
    fn pending_remote_compaction_survives_model_mismatched_replay_check() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let lease = begin_compaction(Some("session"), "gpt-5.6-sol", "build").unwrap();
        let mut mismatched = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        mismatched.model = "gpt-5.6-terra".to_string();

        assert!(apply_compaction_replay(Some("session"), &mismatched, "replay").is_none());
        assert!(store_compaction(
            &lease,
            "gpt-5.6-sol",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "native-history".to_string(),
            }],
        ));
    }

    #[test]
    fn stale_abort_cannot_clear_newer_compaction() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = begin_compaction(Some("session"), "gpt-5.6-sol", "older").unwrap();
        let newer = stage("session", "newer", "newer-native-history");

        abort_compaction_attempt(Some(&older));

        assert!(activate_compaction(
            Some(&newer),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
    }

    #[test]
    fn stale_replay_abort_cannot_clear_newer_compaction() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = stage("session", "older-build", "older-native-history");
        assert!(activate_compaction(
            Some(&older),
            "gpt-5.6-sol",
            &output(STALE_SUMMARY),
        ));
        let older_next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":STALE_SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue old"}]}
        ]));
        let older_replay =
            apply_compaction_replay(Some("session"), &older_next, "older-replay").unwrap();

        let newer = stage("session", "newer-build", "newer-native-history");
        assert!(activate_compaction(
            Some(&newer),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));

        abort_compaction_attempt(Some(&older_replay.lease));

        let newer_next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue new"}]}
        ]));
        let replay = apply_compaction_replay(Some("session"), &newer_next, "newer-replay").unwrap();
        assert!(replay.request.input.iter().any(|item| matches!(
            item,
            ResponsesInputItem::Compaction { encrypted_content }
                if encrypted_content == "newer-native-history"
        )));
    }

    #[test]
    fn invalid_stale_summary_cannot_clear_newer_compaction() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = begin_compaction(Some("session"), "gpt-5.6-sol", "older").unwrap();
        let newer = stage("session", "newer", "newer-native-history");

        assert!(!activate_compaction(
            Some(&older),
            "gpt-5.6-sol",
            &output("too short"),
        ));
        assert!(activate_compaction(
            Some(&newer),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
    }

    #[test]
    fn replay_anchor_mismatch_is_non_destructive() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        for text in [
            "different conversation without the expected summary".to_string(),
            format!("{SUMMARY} and {SUMMARY}"),
        ] {
            clear_all_compactions_for_tests();
            let lease = stage("session", "operation-anchor", "opaque");
            activate_compaction(Some(&lease), "gpt-5.6-sol", &output(SUMMARY));
            let changed = request(json!([
                {"type":"message","role":"user","content":[{"type":"input_text","text":text}]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
            ]));
            assert!(apply_compaction_replay(Some("session"), &changed, "mismatch").is_none());
            let matching = request(json!([
                {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
            ]));
            let replay = apply_compaction_replay(Some("session"), &matching, "probe").unwrap();
            assert!(request_contains_compaction(&replay.request));
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
    fn dropped_build_lease_cleans_only_its_exact_current_state() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let stale = begin_compaction(Some("dropped-build"), "gpt-5.6-sol", "stale").unwrap();
        let current = begin_compaction(Some("dropped-build"), "gpt-5.6-sol", "current").unwrap();

        drop(stale);
        {
            let guard = REGISTRY.lock().unwrap();
            let state = guard.as_ref().unwrap().states.get("dropped-build").unwrap();
            assert_eq!(state.operation_id, "current");
            assert_eq!(state.revision, current.revision);
        }

        drop(current);
        let guard = REGISTRY.lock().unwrap();
        let registry = guard.as_ref().unwrap();
        assert!(!registry.states.contains_key("dropped-build"));
        assert!(!registry.lane_start_orders.contains_key("dropped-build"));
    }

    #[test]
    fn dropped_replay_lease_cleans_its_active_replay() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let build = stage("dropped-replay", "build", "opaque");
        assert!(activate_compaction(
            Some(&build),
            "gpt-5.6-sol",
            &output(SUMMARY)
        ));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay =
            apply_compaction_replay(Some("dropped-replay"), &next, "cancelled-replay").unwrap();

        drop(replay);

        let guard = REGISTRY.lock().unwrap();
        let registry = guard.as_ref().unwrap();
        assert!(!registry.states.contains_key("dropped-replay"));
        assert!(!registry.lane_start_orders.contains_key("dropped-replay"));
    }

    #[test]
    fn replay_cleanup_waits_for_last_lease_clone() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let build = stage("cloned-replay", "build", "opaque");
        assert!(activate_compaction(
            Some(&build),
            "gpt-5.6-sol",
            &output(SUMMARY)
        ));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay =
            apply_compaction_replay(Some("cloned-replay"), &next, "cancelled-replay").unwrap();
        let final_owner = replay.lease.clone();

        drop(replay);
        {
            let guard = REGISTRY.lock().unwrap();
            let state = guard.as_ref().unwrap().states.get("cloned-replay").unwrap();
            let CompactionPhase::Anchored { active_replays, .. } = &state.phase else {
                panic!("expected anchored compaction");
            };
            assert!(active_replays.contains("cancelled-replay"));
        }

        drop(final_owner);
        let guard = REGISTRY.lock().unwrap();
        assert!(!guard.as_ref().unwrap().states.contains_key("cloned-replay"));
    }

    #[test]
    fn failed_replay_clears_anchored_state() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let lease = stage("failed-replay", "operation-failed-replay", "opaque");
        activate_compaction(Some(&lease), "gpt-5.6-sol", &output(SUMMARY));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay =
            apply_compaction_replay(Some("failed-replay"), &next, "failed-attempt").unwrap();

        abort_compaction_attempt(Some(&replay.lease));

        assert!(apply_compaction_replay(Some("failed-replay"), &next, "failed-attempt").is_none());
    }

    #[test]
    fn concurrent_replay_failure_cannot_delete_peer_success() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let lease = stage("concurrent", "build", "opaque");
        assert!(activate_compaction(
            Some(&lease),
            "gpt-5.6-sol",
            &output(SUMMARY)
        ));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay_a = apply_compaction_replay(Some("concurrent"), &next, "replay-a").unwrap();
        let replay_b = apply_compaction_replay(Some("concurrent"), &next, "replay-b").unwrap();
        assert_ne!(replay_a.lease.operation_id, replay_b.lease.operation_id);
        assert_eq!(replay_a.lease.revision, replay_b.lease.revision);

        assert!(activate_compaction(
            Some(&replay_a.lease),
            "gpt-5.6-sol",
            &output("ok")
        ));
        abort_compaction_attempt(Some(&replay_b.lease));

        let probe = apply_compaction_replay(Some("concurrent"), &next, "probe").unwrap();
        assert!(request_contains_compaction(&probe.request));
    }

    #[test]
    fn replay_failure_before_peer_success_cannot_delete_state() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let lease = stage("failure-first", "build", "opaque");
        assert!(activate_compaction(
            Some(&lease),
            "gpt-5.6-sol",
            &output(SUMMARY)
        ));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay_a = apply_compaction_replay(Some("failure-first"), &next, "replay-a").unwrap();
        let replay_b = apply_compaction_replay(Some("failure-first"), &next, "replay-b").unwrap();

        abort_compaction_attempt(Some(&replay_b.lease));
        assert!(activate_compaction(
            Some(&replay_a.lease),
            "gpt-5.6-sol",
            &output("ok")
        ));

        let probe = apply_compaction_replay(Some("failure-first"), &next, "probe").unwrap();
        assert!(request_contains_compaction(&probe.request));
    }

    #[test]
    fn all_failed_concurrent_replays_delete_current_revision() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let lease = stage("all-failed", "build", "opaque");
        assert!(activate_compaction(
            Some(&lease),
            "gpt-5.6-sol",
            &output(SUMMARY)
        ));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay_a = apply_compaction_replay(Some("all-failed"), &next, "replay-a").unwrap();
        let replay_b = apply_compaction_replay(Some("all-failed"), &next, "replay-b").unwrap();

        abort_compaction_attempt(Some(&replay_a.lease));
        abort_compaction_attempt(Some(&replay_b.lease));

        assert!(apply_compaction_replay(Some("all-failed"), &next, "probe").is_none());
    }

    #[test]
    fn stale_summary_mismatch_cannot_delete_newer_revision() {
        const NEW_SUMMARY: &str =
            "newer portable summary with enough detail to identify the current conversation";
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let old = stage("newer-revision", "old-build", "opaque-old");
        assert!(activate_compaction(
            Some(&old),
            "gpt-5.6-sol",
            &output(SUMMARY)
        ));
        let new = stage("newer-revision", "new-build", "opaque-new");
        assert!(activate_compaction(
            Some(&new),
            "gpt-5.6-sol",
            &output(NEW_SUMMARY)
        ));

        let old_request = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        assert!(apply_compaction_replay(Some("newer-revision"), &old_request, "stale").is_none());
        let new_request = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":NEW_SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay =
            apply_compaction_replay(Some("newer-revision"), &new_request, "probe").unwrap();
        let serialized = serde_json::to_string(&replay.request.input).unwrap();
        assert!(serialized.contains("opaque-new"));
        assert!(!serialized.contains("opaque-old"));
    }

    #[test]
    fn replay_model_mismatch_is_non_destructive() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let lease = stage("session", "operation-model-change", "opaque");
        activate_compaction(Some(&lease), "gpt-5.6-sol", &output(SUMMARY));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let mut changed = next.clone();
        changed.model = "gpt-5.4".to_string();
        assert!(apply_compaction_replay(Some("session"), &changed, "mismatch").is_none());
        let replay = apply_compaction_replay(Some("session"), &next, "probe").unwrap();
        assert!(request_contains_compaction(&replay.request));
    }

    #[test]
    fn stale_operations_cannot_publish_activate_or_abort_newer_state() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let stale = begin_compaction(Some("lane"), "gpt-5.6-sol", "operation-stale").unwrap();
        let current = begin_compaction(Some("lane"), "gpt-5.6-sol", "operation-current").unwrap();

        assert!(!store_compaction(
            &stale,
            "gpt-5.6-sol",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "stale".to_string(),
            }],
        ));
        assert!(store_compaction(
            &current,
            "gpt-5.6-sol",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "current".to_string(),
            }],
        ));
        assert!(!activate_compaction(
            Some(&stale),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
        assert!(activate_compaction(
            Some(&current),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
        abort_compaction_attempt(Some(&stale));

        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay = apply_compaction_replay(Some("lane"), &next, "lane-replay").unwrap();
        assert!(replay.request.input.iter().any(|item| matches!(
            item,
            ResponsesInputItem::Compaction { encrypted_content } if encrypted_content == "current"
        )));
    }

    #[test]
    fn delayed_older_reservation_cannot_replace_newer_anchor() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = reserve_compaction_start(Some("lane"), "operation-older").unwrap();
        let newer = reserve_compaction_start(Some("lane"), "operation-newer").unwrap();
        let newer_lease =
            begin_compaction_with_permit(Some("bound"), "gpt-5.6-sol", newer).unwrap();
        assert!(store_compaction(
            &newer_lease,
            "gpt-5.6-sol",
            vec![ResponsesInputItem::Compaction {
                encrypted_content: "newer".to_string(),
            }],
        ));
        assert!(activate_compaction(
            Some(&newer_lease),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));

        assert!(begin_compaction_with_permit(Some("bound"), "gpt-5.6-sol", older).is_none());
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        let replay = apply_compaction_replay(Some("bound"), &next, "replay").unwrap();
        assert!(replay.request.input.iter().any(|item| matches!(
            item,
            ResponsesInputItem::Compaction { encrypted_content } if encrypted_content == "newer"
        )));
    }

    #[test]
    fn aborted_newer_start_still_rejects_older_pending_permit() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let older = reserve_compaction_start(Some("ordered-lane"), "operation-older").unwrap();
        let newer = reserve_compaction_start(Some("ordered-lane"), "operation-newer").unwrap();
        let newer_order = newer.start_order;
        let newer_lease =
            begin_compaction_with_permit(Some("bound"), "gpt-5.6-sol", newer).unwrap();

        abort_compaction_attempt(Some(&newer_lease));
        {
            let guard = REGISTRY.lock().unwrap();
            let registry = guard.as_ref().unwrap();
            assert_eq!(
                registry.lane_start_orders.get("ordered-lane"),
                Some(&newer_order)
            );
            assert!(!registry.states.contains_key("bound"));
        }

        assert!(begin_compaction_with_permit(Some("bound"), "gpt-5.6-sol", older).is_none());
        let guard = REGISTRY.lock().unwrap();
        let registry = guard.as_ref().unwrap();
        assert!(registry.pending_starts.is_empty());
        assert!(!registry.lane_start_orders.contains_key("ordered-lane"));
    }

    #[test]
    fn independent_lanes_never_cross_compaction_artifacts() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let lane_a = stage("lane-a", "operation-a", "opaque-a");
        let lane_b = stage("lane-b", "operation-b", "opaque-b");
        assert!(activate_compaction(
            Some(&lane_b),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
        assert!(activate_compaction(
            Some(&lane_a),
            "gpt-5.6-sol",
            &output(SUMMARY),
        ));
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));

        for (lane, expected, rejected) in [
            ("lane-a", "opaque-a", "opaque-b"),
            ("lane-b", "opaque-b", "opaque-a"),
        ] {
            let replay = apply_compaction_replay(Some(lane), &next, "lane-replay").unwrap();
            let serialized = serde_json::to_string(&replay.request.input).unwrap();
            assert!(serialized.contains(expected));
            assert!(!serialized.contains(rejected));
        }
    }

    #[test]
    fn lane_cleanup_removes_all_route_bindings_only_for_that_lane() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let lane_a_route_1 = begin_compaction_for_lane(
            Some("bound-a-1"),
            Some("lane-a"),
            "gpt-5.6-sol",
            "operation-a-1",
        )
        .unwrap();
        let lane_a_route_2 = begin_compaction_for_lane(
            Some("bound-a-2"),
            Some("lane-a"),
            "gpt-5.6-sol",
            "operation-a-2",
        )
        .unwrap();
        let lane_b = begin_compaction_for_lane(
            Some("bound-b"),
            Some("lane-b"),
            "gpt-5.6-sol",
            "operation-b",
        )
        .unwrap();
        for (lease, encrypted) in [
            (&lane_a_route_1, "opaque-a-1"),
            (&lane_a_route_2, "opaque-a-2"),
            (&lane_b, "opaque-b"),
        ] {
            assert!(store_compaction(
                lease,
                "gpt-5.6-sol",
                vec![ResponsesInputItem::Compaction {
                    encrypted_content: encrypted.to_string(),
                }],
            ));
            assert!(activate_compaction(
                Some(lease),
                "gpt-5.6-sol",
                &output(SUMMARY),
            ));
        }

        clear_compactions_for_lane("lane-a");
        let next = request(json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":SUMMARY}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
        ]));
        assert!(apply_compaction_replay(Some("bound-a-1"), &next, "probe-a-1").is_none());
        assert!(apply_compaction_replay(Some("bound-a-2"), &next, "probe-a-2").is_none());
        let replay_b = apply_compaction_replay(Some("bound-b"), &next, "probe-b").unwrap();
        assert!(
            serde_json::to_string(&replay_b.request.input)
                .unwrap()
                .contains("opaque-b")
        );
    }

    #[test]
    fn lane_cleanup_revokes_compaction_started_before_cleanup() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let stale = reserve_compaction_start(Some("lane-a"), "operation-stale").unwrap();
        let unaffected = reserve_compaction_start(Some("lane-b"), "operation-b").unwrap();

        clear_compactions_for_lane("lane-a");

        assert!(begin_compaction_with_permit(Some("bound-a"), "gpt-5.6-sol", stale).is_none());
        assert!(begin_compaction_with_permit(Some("bound-b"), "gpt-5.6-sol", unaffected).is_some());
        let fresh = reserve_compaction_start(Some("lane-a"), "operation-fresh").unwrap();
        assert!(begin_compaction_with_permit(Some("bound-a"), "gpt-5.6-sol", fresh).is_some());
    }

    #[test]
    fn failed_bound_session_releases_start_reservation() {
        let _guard = TEST_REGISTRY_LOCK.lock().unwrap();
        clear_all_compactions_for_tests();
        let permit = reserve_compaction_start(Some("lane-a"), "operation-a").unwrap();

        assert!(begin_compaction_with_permit(None, "gpt-5.6-sol", permit).is_none());
        assert!(reserve_compaction_start(Some("lane-a"), "operation-a").is_some());
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
