use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

use crate::{logging::create_logger, request_identity::ConversationIdentity};

mod mock;

pub use mock::{MockMonitor, mock_state};

const DEFAULT_RECENT_LIMIT: usize = 200;
const MAX_CODEX_OWNER_STATES: usize = 10_000;
const CODEX_OWNER_STATE_TTL: Duration = Duration::from_secs(30 * 60);
pub const SESSION_TOKEN_BUCKET_SECS: u64 = 10;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CodexMetricsSnapshot {
    pub previous_id_no_candidates: u64,
    pub previous_id_hits: u64,
    pub previous_id_fallbacks: u64,
    pub route_baselines: u64,
    pub route_reuses: u64,
    pub route_switches: u64,
    pub lane_switches: u64,
    pub socket_baselines: u64,
    pub socket_reuses: u64,
    pub socket_switches: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodexPreviousIdOutcome {
    NoCandidate,
    Hit,
    Fallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodexLane {
    Lite,
    Full,
}

impl CodexLane {
    fn from_responses_lite(responses_lite: bool) -> Self {
        if responses_lite {
            Self::Lite
        } else {
            Self::Full
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Lite => "lite",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum CodexRequestPreviousId {
    #[default]
    NotApplicable,
    Pending,
    NoCandidate,
    Hit,
    Fallback,
    Unsettled,
}

impl CodexRequestPreviousId {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::Pending => "pending",
            Self::NoCandidate => "no_candidate",
            Self::Hit => "hit",
            Self::Fallback => "fallback",
            Self::Unsettled => "unsettled",
        }
    }
}

impl From<CodexPreviousIdOutcome> for CodexRequestPreviousId {
    fn from(outcome: CodexPreviousIdOutcome) -> Self {
        match outcome {
            CodexPreviousIdOutcome::NoCandidate => Self::NoCandidate,
            CodexPreviousIdOutcome::Hit => Self::Hit,
            CodexPreviousIdOutcome::Fallback => Self::Fallback,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodexDispatchOutcome {
    Baseline,
    Reuse,
    Switch,
}

impl CodexDispatchOutcome {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Reuse => "reuse",
            Self::Switch => "switch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CodexRecoveryCause {
    MissingState,
    SupersededTurn,
    PromptChanged,
    NotAppendOnly,
    EmptyDelta,
    RouteChanged,
    CompactionReplay,
    AuthRejection,
    PreviousResponseMissing,
    OriginSocketMissing,
    SocketValidationFailed,
    ResponseStartTimeout,
    MissingTerminal,
    EmptyCompletion,
    RetryableUpstreamEvent,
    TransportFailure,
    Cancelled,
}

impl CodexRecoveryCause {
    const ALL: [Self; 17] = [
        Self::MissingState,
        Self::SupersededTurn,
        Self::PromptChanged,
        Self::NotAppendOnly,
        Self::EmptyDelta,
        Self::RouteChanged,
        Self::CompactionReplay,
        Self::AuthRejection,
        Self::PreviousResponseMissing,
        Self::OriginSocketMissing,
        Self::SocketValidationFailed,
        Self::ResponseStartTimeout,
        Self::MissingTerminal,
        Self::EmptyCompletion,
        Self::RetryableUpstreamEvent,
        Self::TransportFailure,
        Self::Cancelled,
    ];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::MissingState => "missing_state",
            Self::SupersededTurn => "superseded_turn",
            Self::PromptChanged => "prompt_changed",
            Self::NotAppendOnly => "not_append_only",
            Self::EmptyDelta => "empty_delta",
            Self::RouteChanged => "route_changed",
            Self::CompactionReplay => "compaction_replay",
            Self::AuthRejection => "auth_rejection",
            Self::PreviousResponseMissing => "previous_response_missing",
            Self::OriginSocketMissing => "origin_socket_missing",
            Self::SocketValidationFailed => "socket_validation_failed",
            Self::ResponseStartTimeout => "response_start_timeout",
            Self::MissingTerminal => "missing_terminal",
            Self::EmptyCompletion => "empty_completion",
            Self::RetryableUpstreamEvent => "retryable_upstream_event",
            Self::TransportFailure => "transport_failure",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodexAppendOnlyOutcome {
    Appended,
    NoDelta,
    RetainedLonger,
    FirstMismatch,
}

impl CodexAppendOnlyOutcome {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Appended => "appended",
            Self::NoDelta => "no_delta",
            Self::RetainedLonger => "retained_longer",
            Self::FirstMismatch => "first_mismatch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CodexAppendOnlyDiagnostics {
    pub(crate) outcome: CodexAppendOnlyOutcome,
    pub(crate) incoming_items: u32,
    pub(crate) retained_items: u32,
    pub(crate) delta_items: u32,
    pub(crate) first_mismatch_index: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodexPoolResolution {
    PoolingDisabled,
    NoEntry,
    ReusableEntry,
    ExactOriginMatched,
    ExactOriginUnavailable,
    ExactOriginMismatch,
}

impl CodexPoolResolution {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::PoolingDisabled => "pooling_disabled",
            Self::NoEntry => "no_entry",
            Self::ReusableEntry => "reusable_entry",
            Self::ExactOriginMatched => "exact_origin_matched",
            Self::ExactOriginUnavailable => "exact_origin_unavailable",
            Self::ExactOriginMismatch => "exact_origin_mismatch",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CodexPoolDiagnostics {
    pub(crate) latest: Option<CodexPoolResolution>,
    pub(crate) lookups: u16,
    pub(crate) hits: u16,
    pub(crate) misses: u16,
    pub(crate) exact_matches: u16,
    pub(crate) exact_unavailable: u16,
    pub(crate) exact_mismatches: u16,
}

impl CodexPoolDiagnostics {
    fn record(&mut self, resolution: CodexPoolResolution) {
        self.latest = Some(resolution);
        self.lookups = self.lookups.saturating_add(1);
        match resolution {
            CodexPoolResolution::PoolingDisabled => {}
            CodexPoolResolution::NoEntry => self.misses = self.misses.saturating_add(1),
            CodexPoolResolution::ReusableEntry => self.hits = self.hits.saturating_add(1),
            CodexPoolResolution::ExactOriginMatched => {
                self.hits = self.hits.saturating_add(1);
                self.exact_matches = self.exact_matches.saturating_add(1);
            }
            CodexPoolResolution::ExactOriginUnavailable => {
                self.misses = self.misses.saturating_add(1);
                self.exact_unavailable = self.exact_unavailable.saturating_add(1);
            }
            CodexPoolResolution::ExactOriginMismatch => {
                self.misses = self.misses.saturating_add(1);
                self.exact_mismatches = self.exact_mismatches.saturating_add(1);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodexSocketValidationFailure {
    PingSend,
    PongReplySend,
    Timeout,
    UnexpectedPong,
    UnexpectedText,
    UnexpectedBinary,
    UnexpectedClose,
    UnexpectedRawFrame,
    TransportRead,
    ConnectionClosed,
}

impl CodexSocketValidationFailure {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::PingSend => "ping_send",
            Self::PongReplySend => "pong_reply_send",
            Self::Timeout => "timeout",
            Self::UnexpectedPong => "unexpected_pong",
            Self::UnexpectedText => "unexpected_text",
            Self::UnexpectedBinary => "unexpected_binary",
            Self::UnexpectedClose => "unexpected_close",
            Self::UnexpectedRawFrame => "unexpected_raw_frame",
            Self::TransportRead => "transport_read",
            Self::ConnectionClosed => "connection_closed",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CodexSocketValidationDiagnostics {
    pub(crate) attempts: u16,
    pub(crate) successes: u16,
    pub(crate) failures: u16,
    pub(crate) latest_failure: Option<CodexSocketValidationFailure>,
    pub(crate) latest_elapsed_ms: u32,
    pub(crate) latest_required_origin: bool,
}

impl CodexSocketValidationDiagnostics {
    fn record(
        &mut self,
        failure: Option<CodexSocketValidationFailure>,
        elapsed_ms: u32,
        required_origin: bool,
    ) {
        self.attempts = self.attempts.saturating_add(1);
        self.latest_failure = failure;
        self.latest_elapsed_ms = elapsed_ms;
        self.latest_required_origin = required_origin;
        if failure.is_some() {
            self.failures = self.failures.saturating_add(1);
        } else {
            self.successes = self.successes.saturating_add(1);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CodexCauseSet(u32);

impl CodexCauseSet {
    pub(crate) fn insert(&mut self, cause: CodexRecoveryCause) {
        self.0 |= 1_u32 << cause as u8;
    }

    pub(crate) fn iter(self) -> impl Iterator<Item = CodexRecoveryCause> {
        CodexRecoveryCause::ALL
            .into_iter()
            .filter(move |cause| self.0 & (1_u32 << *cause as u8) != 0)
    }

    pub(crate) fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CodexRecoveryDiagnostics {
    pub(crate) previous_id_cause: Option<CodexRecoveryCause>,
    pub(crate) socket_causes: CodexCauseSet,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CodexDispatchSummary {
    pub(crate) first: Option<CodexDispatchOutcome>,
    pub(crate) latest: Option<CodexDispatchOutcome>,
    pub(crate) dispatches: u16,
    pub(crate) baselines: u16,
    pub(crate) reuses: u16,
    pub(crate) switches: u16,
}

impl CodexDispatchSummary {
    fn record(&mut self, outcome: CodexDispatchOutcome) {
        self.first.get_or_insert(outcome);
        self.latest = Some(outcome);
        self.dispatches = self.dispatches.saturating_add(1);
        let counter = match outcome {
            CodexDispatchOutcome::Baseline => &mut self.baselines,
            CodexDispatchOutcome::Reuse => &mut self.reuses,
            CodexDispatchOutcome::Switch => &mut self.switches,
        };
        *counter = counter.saturating_add(1);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RequestAcceleration {
    pub(crate) requested_speed: Option<&'static str>,
    pub(crate) codex_service_tier: Option<&'static str>,
}

impl RequestAcceleration {
    pub(crate) fn is_empty(self) -> bool {
        self.requested_speed.is_none() && self.codex_service_tier.is_none()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CodexRequestDiagnostics {
    pub(crate) acceleration: RequestAcceleration,
    pub(crate) lane: Option<CodexLane>,
    pub(crate) previous_id: CodexRequestPreviousId,
    pub(crate) recovery: CodexRecoveryDiagnostics,
    pub(crate) append_only: Option<CodexAppendOnlyDiagnostics>,
    pub(crate) pool: CodexPoolDiagnostics,
    pub(crate) validation: CodexSocketValidationDiagnostics,
    pub(crate) route: CodexDispatchSummary,
    pub(crate) socket: CodexDispatchSummary,
    pub(crate) event_sequence: u16,
}

impl CodexRequestDiagnostics {
    fn finish(&mut self) {
        if self.previous_id == CodexRequestPreviousId::Pending {
            self.previous_id = CodexRequestPreviousId::Unsettled;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointKind {
    Messages,
    CountTokens,
    Responses,
    ChatCompletions,
    Images,
    Transcriptions,
}

impl EndpointKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Messages => "messages",
            Self::CountTokens => "count_tokens",
            Self::Responses => "responses",
            Self::ChatCompletions => "chat_completions",
            Self::Images => "images",
            Self::Transcriptions => "transcriptions",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestStatus {
    Started,
    ProviderSelected,
    Compacting,
    Upstream,
    Streaming,
    Completed,
    Failed,
}

impl RequestStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::ProviderSelected => "selected",
            Self::Compacting => "compacting",
            Self::Upstream => "upstream",
            Self::Streaming => "streaming",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub enum MonitorEvent {
    RequestStarted {
        request_id: String,
        session_id: Option<String>,
        session_seq: Option<u64>,
        endpoint: EndpointKind,
    },
    ProjectResolved {
        request_id: String,
        project: String,
    },
    SessionSequenceResolved {
        request_id: String,
        session_seq: u64,
    },
    ProviderSelected {
        request_id: String,
        provider: String,
        model: String,
        effort: Option<String>,
    },
    ModelResolved {
        request_id: String,
        model: String,
    },
    CompactionStarted {
        request_id: String,
    },
    UpstreamStarted {
        request_id: String,
    },
    GenerationStarted {
        request_id: String,
    },
    TrafficCapturePath {
        request_id: String,
        path: PathBuf,
    },
    StreamProgress {
        request_id: String,
        bytes: u64,
        chunks: u64,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
    UsageUpdated {
        request_id: String,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
    RequestCompleted {
        request_id: String,
        http_status: u16,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
    RequestFailed {
        request_id: String,
        http_status: Option<u16>,
        error: String,
    },
    RequestAbandoned {
        request_id: String,
        error: String,
    },
}

#[derive(Debug, Clone)]
pub struct ActiveRequest {
    pub request_id: String,
    pub session_id: Option<String>,
    pub session_seq: Option<u64>,
    pub project: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub endpoint: EndpointKind,
    pub started_at: SystemTime,
    started_instant: Instant,
    pub generation_started_at: Option<SystemTime>,
    generation_started_instant: Option<Instant>,
    generation_initial_output_tokens: u64,
    pub generation_finished_at: Option<SystemTime>,
    pub generation_duration: Option<Duration>,
    pub status: RequestStatus,
    pub streamed_bytes: u64,
    pub stream_chunks: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub error: Option<String>,
    pub traffic_capture_path: Option<PathBuf>,
    codex: Option<CodexRequestDiagnostics>,
}

impl ActiveRequest {
    pub fn elapsed(&self) -> Duration {
        self.started_instant.elapsed()
    }

    pub(crate) fn codex_diagnostics(&self) -> Option<&CodexRequestDiagnostics> {
        self.codex.as_ref()
    }

    pub fn rate(&self) -> Throughput {
        throughput(
            self.output_tokens
                .and_then(|tokens| tokens.checked_sub(self.generation_initial_output_tokens)),
            self.streamed_bytes,
            self.stream_chunks,
            self.generation_duration.unwrap_or(Duration::ZERO),
        )
    }
}

#[derive(Debug, Clone)]
pub struct CompletedRequest {
    pub request_id: String,
    pub session_id: Option<String>,
    pub session_seq: Option<u64>,
    pub project: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub endpoint: EndpointKind,
    pub started_at: SystemTime,
    pub finished_at: SystemTime,
    pub generation_started_at: Option<SystemTime>,
    generation_started_instant: Option<Instant>,
    generation_initial_output_tokens: u64,
    pub generation_finished_at: Option<SystemTime>,
    pub generation_duration: Option<Duration>,
    pub status: RequestStatus,
    pub http_status: Option<u16>,
    pub latency: Duration,
    pub streamed_bytes: u64,
    pub stream_chunks: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub error: Option<String>,
    pub traffic_capture_path: Option<PathBuf>,
    codex: Option<CodexRequestDiagnostics>,
}

impl CompletedRequest {
    pub(crate) fn codex_diagnostics(&self) -> Option<&CodexRequestDiagnostics> {
        self.codex.as_ref()
    }

    pub fn rate(&self) -> Throughput {
        throughput(
            self.output_tokens
                .and_then(|tokens| tokens.checked_sub(self.generation_initial_output_tokens)),
            self.streamed_bytes,
            self.stream_chunks,
            self.generation_duration.unwrap_or(Duration::ZERO),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Throughput {
    TokensPerSecond(f64),
    BytesPerSecond(f64),
    EventsPerSecond(f64),
    None,
}

impl Throughput {
    pub fn label(&self) -> String {
        match self {
            Self::TokensPerSecond(value) => format!("{value:.1} tok/s"),
            Self::BytesPerSecond(value) if *value >= 1024.0 => {
                format!("{:.1} KB/s", value / 1024.0)
            }
            Self::BytesPerSecond(value) => format!("{value:.0} B/s"),
            Self::EventsPerSecond(value) => format!("{value:.1} ev/s"),
            Self::None => "-".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MonitorState {
    pub started_at: SystemTime,
    pub sessions: Vec<SessionSummary>,
    pub active: Vec<ActiveRequest>,
    pub recent: Vec<CompletedRequest>,
    pub codex: CodexMetricsSnapshot,
}

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub session_id: Option<String>,
    pub project: Option<String>,
    pub active_count: usize,
    pub request_count: usize,
    pub failure_count: usize,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub last_seen: SystemTime,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub output_token_samples: Vec<(SystemTime, u64)>,
    rate_output_tokens: u64,
    pub generation_duration: Duration,
    pub last_status: String,
}

impl SessionSummary {
    pub fn rate(&self) -> Throughput {
        throughput(
            Some(self.rate_output_tokens).filter(|tokens| *tokens > 0),
            0,
            0,
            self.generation_duration,
        )
    }

    pub fn label(&self) -> String {
        self.session_id
            .clone()
            .unwrap_or_else(|| "no-session".to_string())
    }
}

#[derive(Debug)]
struct MonitorStore {
    started_at: SystemTime,
    active: HashMap<String, ActiveRequest>,
    recent: VecDeque<CompletedRequest>,
    session_usage: HashMap<Option<String>, SessionUsage>,
    session_output_buckets: HashMap<Option<String>, Vec<(u64, u64)>>,
    codex: CodexMetricsStore,
    recent_limit: usize,
}

#[derive(Debug, Clone, Copy)]
struct LastCodexOwnerDispatch {
    route_identity: [u8; 32],
    responses_lite: bool,
    socket_id: u64,
    updated_at: Instant,
}

#[derive(Debug, Default)]
struct CodexMetricsStore {
    totals: CodexMetricsSnapshot,
    owner_dispatches: HashMap<ConversationIdentity, LastCodexOwnerDispatch>,
}

#[derive(Debug, Default)]
struct SessionUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct MonitorHandle {
    store: Arc<Mutex<MonitorStore>>,
    diagnostic_logging: bool,
}

#[derive(Debug, Clone)]
struct CodexUpdateSnapshot {
    diagnostics: CodexRequestDiagnostics,
    completed: Option<CompletedRequest>,
    changed: bool,
}

fn codex_dispatch_json(summary: CodexDispatchSummary) -> serde_json::Value {
    serde_json::json!({
        "first": summary.first.map(CodexDispatchOutcome::label),
        "latest": summary.latest.map(CodexDispatchOutcome::label),
        "dispatches": summary.dispatches,
        "baselines": summary.baselines,
        "reuses": summary.reuses,
        "switches": summary.switches,
    })
}

fn codex_diagnostics_json(diagnostics: CodexRequestDiagnostics) -> serde_json::Value {
    let append_only = diagnostics.append_only.map(|append| {
        serde_json::json!({
            "outcome": append.outcome.label(),
            "incomingItems": append.incoming_items,
            "retainedItems": append.retained_items,
            "deltaItems": append.delta_items,
            "firstMismatchIndex": append.first_mismatch_index,
        })
    });
    serde_json::json!({
        "eventSequence": diagnostics.event_sequence,
        "requestedSpeed": diagnostics.acceleration.requested_speed,
        "serviceTier": diagnostics.acceleration.codex_service_tier,
        "lane": diagnostics.lane.map(CodexLane::label),
        "previousId": diagnostics.previous_id.label(),
        "previousIdCause": diagnostics.recovery.previous_id_cause.map(CodexRecoveryCause::label),
        "socketCauses": diagnostics.recovery.socket_causes
            .iter()
            .map(CodexRecoveryCause::label)
            .collect::<Vec<_>>(),
        "appendOnly": append_only,
        "pool": {
            "latest": diagnostics.pool.latest.map(CodexPoolResolution::label),
            "lookups": diagnostics.pool.lookups,
            "hits": diagnostics.pool.hits,
            "misses": diagnostics.pool.misses,
            "exactMatches": diagnostics.pool.exact_matches,
            "exactUnavailable": diagnostics.pool.exact_unavailable,
            "exactMismatches": diagnostics.pool.exact_mismatches,
        },
        "validation": {
            "attempts": diagnostics.validation.attempts,
            "successes": diagnostics.validation.successes,
            "failures": diagnostics.validation.failures,
            "latestFailure": diagnostics.validation.latest_failure.map(CodexSocketValidationFailure::label),
            "latestElapsedMs": diagnostics.validation.latest_elapsed_ms,
            "latestRequiredOrigin": diagnostics.validation.latest_required_origin,
        },
        "route": codex_dispatch_json(diagnostics.route),
        "socket": codex_dispatch_json(diagnostics.socket),
    })
}

fn codex_event_fields(
    request_id: &str,
    diagnostics: CodexRequestDiagnostics,
    state_changed: bool,
    mut fields: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    fields.insert("schemaVersion".into(), serde_json::json!(1));
    fields.insert("reqId".into(), serde_json::json!(request_id));
    fields.insert(
        "eventSequence".into(),
        serde_json::json!(diagnostics.event_sequence),
    );
    fields.insert("stateChanged".into(), serde_json::json!(state_changed));
    fields
}

fn log_codex_event(
    enabled: bool,
    request_id: &str,
    message: &str,
    diagnostics: CodexRequestDiagnostics,
    state_changed: bool,
    fields: serde_json::Map<String, serde_json::Value>,
) {
    if !enabled {
        return;
    }
    create_logger("codex").info(
        message,
        Some(codex_event_fields(
            request_id,
            diagnostics,
            state_changed,
            fields,
        )),
    );
}

fn codex_request_snapshot_fields(
    request: &CompletedRequest,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    if request.provider.as_deref() != Some("codex") {
        return None;
    }
    let diagnostics = request.codex?;
    let mut fields = serde_json::Map::new();
    fields.insert("schemaVersion".into(), serde_json::json!(1));
    fields.insert("reqId".into(), serde_json::json!(request.request_id));
    fields.insert(
        "endpoint".into(),
        serde_json::json!(request.endpoint.label()),
    );
    fields.insert("outcome".into(), serde_json::json!(request.status.label()));
    fields.insert("httpStatus".into(), serde_json::json!(request.http_status));
    fields.insert(
        "latencyMs".into(),
        serde_json::json!(request.latency.as_millis().min(u128::from(u64::MAX)) as u64),
    );
    fields.insert(
        "streamedBytes".into(),
        serde_json::json!(request.streamed_bytes),
    );
    fields.insert(
        "streamChunks".into(),
        serde_json::json!(request.stream_chunks),
    );
    fields.insert(
        "inputTokens".into(),
        serde_json::json!(request.input_tokens),
    );
    fields.insert(
        "outputTokens".into(),
        serde_json::json!(request.output_tokens),
    );
    fields.insert("diagnostics".into(), codex_diagnostics_json(diagnostics));
    Some(fields)
}

fn log_codex_request_snapshot(request: &CompletedRequest, message: &str) {
    if let Some(fields) = codex_request_snapshot_fields(request) {
        create_logger("codex").info(message, Some(fields));
    }
}

fn log_codex_update_snapshot(enabled: bool, update: &CodexUpdateSnapshot) {
    if enabled
        && update.changed
        && let Some(completed) = update.completed.as_ref()
    {
        log_codex_request_snapshot(completed, "codex_request_diagnostic_update");
    }
}

impl Default for MonitorHandle {
    fn default() -> Self {
        Self::with_diagnostic_logging(DEFAULT_RECENT_LIMIT, cfg!(not(test)))
    }
}

impl MonitorHandle {
    pub fn new(recent_limit: usize) -> Self {
        Self::with_diagnostic_logging(recent_limit, false)
    }

    fn with_diagnostic_logging(recent_limit: usize, diagnostic_logging: bool) -> Self {
        Self {
            store: Arc::new(Mutex::new(MonitorStore {
                started_at: SystemTime::now(),
                active: HashMap::new(),
                recent: VecDeque::new(),
                session_usage: HashMap::new(),
                session_output_buckets: HashMap::new(),
                codex: CodexMetricsStore::default(),
                recent_limit,
            })),
            diagnostic_logging,
        }
    }

    pub fn publish(&self, event: MonitorEvent) {
        let completed = match self.store.lock() {
            Ok(mut store) => store.apply(event),
            Err(_) => None,
        };
        if self.diagnostic_logging
            && let Some(completed) = completed.as_ref()
        {
            log_codex_request_snapshot(completed, "codex_request_diagnostic_summary");
        }
    }

    pub fn snapshot(&self) -> MonitorState {
        match self.store.lock() {
            Ok(store) => store.snapshot(),
            Err(_) => MonitorState {
                started_at: SystemTime::now(),
                sessions: Vec::new(),
                active: Vec::new(),
                recent: Vec::new(),
                codex: CodexMetricsSnapshot::default(),
            },
        }
    }

    pub fn request_started(
        &self,
        request_id: impl Into<String>,
        session_id: Option<String>,
        session_seq: Option<u64>,
        endpoint: EndpointKind,
    ) {
        self.publish(MonitorEvent::RequestStarted {
            request_id: request_id.into(),
            session_id,
            session_seq,
            endpoint,
        });
    }

    pub fn project_resolved(&self, request_id: impl Into<String>, project: impl Into<String>) {
        self.publish(MonitorEvent::ProjectResolved {
            request_id: request_id.into(),
            project: project.into(),
        });
    }

    pub fn session_sequence_resolved(&self, request_id: impl Into<String>, session_seq: u64) {
        self.publish(MonitorEvent::SessionSequenceResolved {
            request_id: request_id.into(),
            session_seq,
        });
    }

    pub fn provider_selected(
        &self,
        request_id: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
        effort: Option<String>,
    ) {
        self.publish(MonitorEvent::ProviderSelected {
            request_id: request_id.into(),
            provider: provider.into(),
            model: model.into(),
            effort,
        });
    }

    pub fn model_resolved(&self, request_id: impl Into<String>, model: impl Into<String>) {
        self.publish(MonitorEvent::ModelResolved {
            request_id: request_id.into(),
            model: model.into(),
        });
    }

    pub fn compaction_started(&self, request_id: impl Into<String>) {
        self.publish(MonitorEvent::CompactionStarted {
            request_id: request_id.into(),
        });
    }

    pub fn upstream_started(&self, request_id: impl Into<String>) {
        self.publish(MonitorEvent::UpstreamStarted {
            request_id: request_id.into(),
        });
    }

    pub fn generation_started(&self, request_id: impl Into<String>) {
        self.publish(MonitorEvent::GenerationStarted {
            request_id: request_id.into(),
        });
    }

    pub fn traffic_capture_path(&self, request_id: impl Into<String>, path: PathBuf) {
        self.publish(MonitorEvent::TrafficCapturePath {
            request_id: request_id.into(),
            path,
        });
    }

    pub fn stream_progress(
        &self,
        request_id: impl Into<String>,
        bytes: u64,
        chunks: u64,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    ) {
        self.publish(MonitorEvent::StreamProgress {
            request_id: request_id.into(),
            bytes,
            chunks,
            input_tokens,
            output_tokens,
        });
    }

    pub fn usage_updated(
        &self,
        request_id: impl Into<String>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    ) {
        self.publish(MonitorEvent::UsageUpdated {
            request_id: request_id.into(),
            input_tokens,
            output_tokens,
        });
    }

    pub fn request_completed(
        &self,
        request_id: impl Into<String>,
        http_status: u16,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    ) {
        self.publish(MonitorEvent::RequestCompleted {
            request_id: request_id.into(),
            http_status,
            input_tokens,
            output_tokens,
        });
    }

    pub fn request_failed(
        &self,
        request_id: impl Into<String>,
        http_status: Option<u16>,
        error: impl Into<String>,
    ) {
        self.publish(MonitorEvent::RequestFailed {
            request_id: request_id.into(),
            http_status,
            error: error.into(),
        });
    }

    pub(crate) fn codex_acceleration_resolved(
        &self,
        request_id: &str,
        requested_speed: Option<&'static str>,
        codex_service_tier: Option<&'static str>,
    ) {
        let acceleration = RequestAcceleration {
            requested_speed,
            codex_service_tier,
        };
        if acceleration.is_empty() {
            return;
        }
        let update = match self.store.lock() {
            Ok(mut store) => store.update_codex_request(request_id, |diagnostics| {
                diagnostics.acceleration = acceleration;
            }),
            Err(_) => None,
        };
        if let Some(update) = update {
            let mut fields = serde_json::Map::new();
            fields.insert("requestedSpeed".into(), serde_json::json!(requested_speed));
            fields.insert("serviceTier".into(), serde_json::json!(codex_service_tier));
            log_codex_event(
                self.diagnostic_logging,
                request_id,
                "codex_acceleration_resolved",
                update.diagnostics,
                update.changed,
                fields,
            );
            log_codex_update_snapshot(self.diagnostic_logging, &update);
        }
    }

    pub(crate) fn codex_request_lane(&self, request_id: &str, responses_lite: bool) {
        let lane = CodexLane::from_responses_lite(responses_lite);
        let update = match self.store.lock() {
            Ok(mut store) => store.update_codex_request(request_id, |diagnostics| {
                diagnostics.lane = Some(lane);
            }),
            Err(_) => None,
        };
        if let Some(update) = update {
            let mut fields = serde_json::Map::new();
            fields.insert("lane".into(), serde_json::json!(lane.label()));
            log_codex_event(
                self.diagnostic_logging,
                request_id,
                "codex_request_lane",
                update.diagnostics,
                update.changed,
                fields,
            );
            log_codex_update_snapshot(self.diagnostic_logging, &update);
        }
    }

    pub(crate) fn codex_previous_id_pending(&self, request_id: &str) {
        let update = match self.store.lock() {
            Ok(mut store) => store.update_codex_request(request_id, |diagnostics| {
                diagnostics.previous_id = CodexRequestPreviousId::Pending;
            }),
            Err(_) => None,
        };
        if let Some(update) = update {
            log_codex_event(
                self.diagnostic_logging,
                request_id,
                "codex_previous_id_pending",
                update.diagnostics,
                update.changed,
                serde_json::Map::new(),
            );
            log_codex_update_snapshot(self.diagnostic_logging, &update);
        }
    }

    pub(crate) fn codex_previous_id_settled(
        &self,
        request_id: &str,
        outcome: CodexPreviousIdOutcome,
    ) {
        let previous_id = CodexRequestPreviousId::from(outcome);
        let update = match self.store.lock() {
            Ok(mut store) => {
                store.codex.record_previous_id(outcome);
                store.update_codex_request(request_id, |diagnostics| {
                    diagnostics.previous_id = previous_id;
                })
            }
            Err(_) => None,
        };
        if let Some(update) = update {
            let mut fields = serde_json::Map::new();
            fields.insert("outcome".into(), serde_json::json!(previous_id.label()));
            log_codex_event(
                self.diagnostic_logging,
                request_id,
                "codex_previous_id_settled",
                update.diagnostics,
                update.changed,
                fields,
            );
            log_codex_update_snapshot(self.diagnostic_logging, &update);
        }
    }

    pub(crate) fn codex_previous_id_cause(&self, request_id: &str, cause: CodexRecoveryCause) {
        let update = match self.store.lock() {
            Ok(mut store) => store.update_codex_request(request_id, |diagnostics| {
                diagnostics.recovery.previous_id_cause.get_or_insert(cause);
            }),
            Err(_) => None,
        };
        if let Some(update) = update {
            let mut fields = serde_json::Map::new();
            fields.insert("cause".into(), serde_json::json!(cause.label()));
            log_codex_event(
                self.diagnostic_logging,
                request_id,
                "codex_previous_id_cause",
                update.diagnostics,
                update.changed,
                fields,
            );
            log_codex_update_snapshot(self.diagnostic_logging, &update);
        }
    }

    pub(crate) fn codex_socket_cause(&self, request_id: &str, cause: CodexRecoveryCause) {
        let update = match self.store.lock() {
            Ok(mut store) => store.update_codex_request(request_id, |diagnostics| {
                diagnostics.recovery.socket_causes.insert(cause);
            }),
            Err(_) => None,
        };
        if let Some(update) = update {
            let mut fields = serde_json::Map::new();
            fields.insert("cause".into(), serde_json::json!(cause.label()));
            log_codex_event(
                self.diagnostic_logging,
                request_id,
                "codex_socket_recovery",
                update.diagnostics,
                update.changed,
                fields,
            );
            log_codex_update_snapshot(self.diagnostic_logging, &update);
        }
    }

    pub(crate) fn codex_append_only(
        &self,
        request_id: &str,
        append_only: CodexAppendOnlyDiagnostics,
    ) {
        let update = match self.store.lock() {
            Ok(mut store) => store.update_codex_request(request_id, |diagnostics| {
                diagnostics.append_only = Some(append_only);
            }),
            Err(_) => None,
        };
        if let Some(update) = update {
            let mut fields = serde_json::Map::new();
            fields.insert(
                "outcome".into(),
                serde_json::json!(append_only.outcome.label()),
            );
            fields.insert(
                "incomingItems".into(),
                serde_json::json!(append_only.incoming_items),
            );
            fields.insert(
                "retainedItems".into(),
                serde_json::json!(append_only.retained_items),
            );
            fields.insert(
                "deltaItems".into(),
                serde_json::json!(append_only.delta_items),
            );
            fields.insert(
                "firstMismatchIndex".into(),
                serde_json::json!(append_only.first_mismatch_index),
            );
            log_codex_event(
                self.diagnostic_logging,
                request_id,
                "codex_append_only_comparison",
                update.diagnostics,
                update.changed,
                fields,
            );
            log_codex_update_snapshot(self.diagnostic_logging, &update);
        }
    }

    pub(crate) fn codex_pool_resolution(&self, request_id: &str, resolution: CodexPoolResolution) {
        let update = match self.store.lock() {
            Ok(mut store) => store.update_codex_request(request_id, |diagnostics| {
                diagnostics.pool.record(resolution);
            }),
            Err(_) => None,
        };
        if let Some(update) = update {
            let mut fields = serde_json::Map::new();
            fields.insert("resolution".into(), serde_json::json!(resolution.label()));
            log_codex_event(
                self.diagnostic_logging,
                request_id,
                "codex_pool_resolution",
                update.diagnostics,
                update.changed,
                fields,
            );
            log_codex_update_snapshot(self.diagnostic_logging, &update);
        }
    }

    pub(crate) fn codex_socket_validation(
        &self,
        request_id: &str,
        failure: Option<CodexSocketValidationFailure>,
        elapsed_ms: u32,
        required_origin: bool,
    ) {
        let update = match self.store.lock() {
            Ok(mut store) => store.update_codex_request(request_id, |diagnostics| {
                diagnostics
                    .validation
                    .record(failure, elapsed_ms, required_origin);
            }),
            Err(_) => None,
        };
        if let Some(update) = update {
            let mut fields = serde_json::Map::new();
            fields.insert(
                "outcome".into(),
                serde_json::json!(if failure.is_some() {
                    "failed"
                } else {
                    "passed"
                }),
            );
            fields.insert(
                "failure".into(),
                serde_json::json!(failure.map(CodexSocketValidationFailure::label)),
            );
            fields.insert("elapsedMs".into(), serde_json::json!(elapsed_ms));
            fields.insert("requiredOrigin".into(), serde_json::json!(required_origin));
            log_codex_event(
                self.diagnostic_logging,
                request_id,
                "codex_socket_validation",
                update.diagnostics,
                update.changed,
                fields,
            );
            log_codex_update_snapshot(self.diagnostic_logging, &update);
        }
    }

    pub(crate) fn codex_websocket_dispatch(
        &self,
        request_id: &str,
        owner: ConversationIdentity,
        route_identity: [u8; 32],
        responses_lite: bool,
        socket_id: u64,
    ) {
        let update_and_outcomes = match self.store.lock() {
            Ok(mut store) => {
                let (route, socket) = store.codex.record_owner_dispatch(
                    owner,
                    route_identity,
                    responses_lite,
                    socket_id,
                );
                let update = store.update_codex_request(request_id, |diagnostics| {
                    diagnostics.lane = Some(CodexLane::from_responses_lite(responses_lite));
                    diagnostics.route.record(route);
                    diagnostics.socket.record(socket);
                });
                update.map(|update| (update, route, socket))
            }
            Err(_) => None,
        };
        if let Some((update, route, socket)) = update_and_outcomes {
            let mut fields = serde_json::Map::new();
            fields.insert("route".into(), serde_json::json!(route.label()));
            fields.insert("socket".into(), serde_json::json!(socket.label()));
            fields.insert(
                "dispatch".into(),
                serde_json::json!(update.diagnostics.route.dispatches),
            );
            log_codex_event(
                self.diagnostic_logging,
                request_id,
                "codex_websocket_dispatch",
                update.diagnostics,
                update.changed,
                fields,
            );
            log_codex_update_snapshot(self.diagnostic_logging, &update);
        }
    }

    pub fn request_abandoned(&self, request_id: impl Into<String>, error: impl Into<String>) {
        self.publish(MonitorEvent::RequestAbandoned {
            request_id: request_id.into(),
            error: error.into(),
        });
    }
}

impl CodexMetricsStore {
    fn record_previous_id(&mut self, outcome: CodexPreviousIdOutcome) {
        let counter = match outcome {
            CodexPreviousIdOutcome::NoCandidate => &mut self.totals.previous_id_no_candidates,
            CodexPreviousIdOutcome::Hit => &mut self.totals.previous_id_hits,
            CodexPreviousIdOutcome::Fallback => &mut self.totals.previous_id_fallbacks,
        };
        *counter = counter.saturating_add(1);
    }

    fn record_owner_dispatch(
        &mut self,
        owner: ConversationIdentity,
        route_identity: [u8; 32],
        responses_lite: bool,
        socket_id: u64,
    ) -> (CodexDispatchOutcome, CodexDispatchOutcome) {
        let now = Instant::now();
        let current = LastCodexOwnerDispatch {
            route_identity,
            responses_lite,
            socket_id,
            updated_at: now,
        };
        let previous = self
            .owner_dispatches
            .remove(&owner)
            .filter(|previous| now.duration_since(previous.updated_at) <= CODEX_OWNER_STATE_TTL);
        self.owner_dispatches.insert(owner, current);

        let (route_outcome, socket_outcome) = if let Some(previous) = previous {
            let route_outcome = if previous.route_identity == route_identity {
                self.totals.route_reuses = self.totals.route_reuses.saturating_add(1);
                CodexDispatchOutcome::Reuse
            } else {
                self.totals.route_switches = self.totals.route_switches.saturating_add(1);
                if previous.responses_lite != responses_lite {
                    self.totals.lane_switches = self.totals.lane_switches.saturating_add(1);
                }
                CodexDispatchOutcome::Switch
            };
            let socket_outcome = if previous.socket_id == socket_id {
                self.totals.socket_reuses = self.totals.socket_reuses.saturating_add(1);
                CodexDispatchOutcome::Reuse
            } else {
                self.totals.socket_switches = self.totals.socket_switches.saturating_add(1);
                CodexDispatchOutcome::Switch
            };
            (route_outcome, socket_outcome)
        } else {
            self.totals.route_baselines = self.totals.route_baselines.saturating_add(1);
            self.totals.socket_baselines = self.totals.socket_baselines.saturating_add(1);
            (
                CodexDispatchOutcome::Baseline,
                CodexDispatchOutcome::Baseline,
            )
        };

        while self.owner_dispatches.len() > MAX_CODEX_OWNER_STATES {
            let oldest = self
                .owner_dispatches
                .iter()
                .min_by_key(|(_, dispatch)| dispatch.updated_at)
                .map(|(owner, _)| owner.clone());
            let Some(oldest) = oldest else {
                break;
            };
            self.owner_dispatches.remove(&oldest);
        }
        (route_outcome, socket_outcome)
    }
}

impl MonitorStore {
    fn update_codex_request(
        &mut self,
        request_id: &str,
        update: impl FnOnce(&mut CodexRequestDiagnostics),
    ) -> Option<CodexUpdateSnapshot> {
        if let Some(active) = self.active.get_mut(request_id) {
            let diagnostics = active.codex.get_or_insert_default();
            let before = *diagnostics;
            update(diagnostics);
            let changed = *diagnostics != before;
            diagnostics.event_sequence = diagnostics.event_sequence.saturating_add(1);
            return Some(CodexUpdateSnapshot {
                diagnostics: *diagnostics,
                completed: None,
                changed,
            });
        }
        let completed = self
            .recent
            .iter_mut()
            .find(|request| request.request_id == request_id)?;
        let diagnostics = completed.codex.get_or_insert_default();
        let before = *diagnostics;
        update(diagnostics);
        let changed = *diagnostics != before;
        diagnostics.event_sequence = diagnostics.event_sequence.saturating_add(1);
        Some(CodexUpdateSnapshot {
            diagnostics: *diagnostics,
            completed: Some(completed.clone()),
            changed,
        })
    }

    fn apply(&mut self, event: MonitorEvent) -> Option<CompletedRequest> {
        match event {
            MonitorEvent::RequestStarted {
                request_id,
                session_id,
                session_seq,
                endpoint,
            } => {
                self.active.insert(
                    request_id.clone(),
                    ActiveRequest {
                        request_id,
                        session_id,
                        session_seq,
                        project: None,
                        provider: None,
                        model: None,
                        effort: None,
                        endpoint,
                        started_at: SystemTime::now(),
                        started_instant: Instant::now(),
                        generation_started_at: None,
                        generation_started_instant: None,
                        generation_initial_output_tokens: 0,
                        generation_finished_at: None,
                        generation_duration: None,
                        status: RequestStatus::Started,
                        streamed_bytes: 0,
                        stream_chunks: 0,
                        input_tokens: None,
                        output_tokens: None,
                        error: None,
                        traffic_capture_path: None,
                        codex: None,
                    },
                );
            }
            MonitorEvent::ProjectResolved {
                request_id,
                project,
            } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.project = Some(project);
                }
            }
            MonitorEvent::SessionSequenceResolved {
                request_id,
                session_seq,
            } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.session_seq = Some(session_seq);
                }
            }
            MonitorEvent::ProviderSelected {
                request_id,
                provider,
                model,
                effort,
            } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.provider = Some(provider);
                    active.model = Some(model);
                    active.effort = effort;
                    active.status = RequestStatus::ProviderSelected;
                }
            }
            MonitorEvent::ModelResolved { request_id, model } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.model = Some(match active.model.take() {
                        Some(incoming) if incoming != model => format!("{incoming} → {model}"),
                        Some(incoming) => incoming,
                        None => model,
                    });
                }
            }
            MonitorEvent::CompactionStarted { request_id } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.status = RequestStatus::Compacting;
                }
            }
            MonitorEvent::UpstreamStarted { request_id } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.status = RequestStatus::Upstream;
                }
            }
            MonitorEvent::GenerationStarted { request_id } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.generation_started_at = Some(SystemTime::now());
                    active.generation_started_instant = Some(Instant::now());
                    active.generation_initial_output_tokens = active.output_tokens.unwrap_or(0);
                    active.generation_finished_at = None;
                    active.generation_duration = None;
                }
            }
            MonitorEvent::TrafficCapturePath { request_id, path } => {
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.traffic_capture_path = Some(path);
                }
            }
            MonitorEvent::StreamProgress {
                request_id,
                bytes,
                chunks,
                input_tokens,
                output_tokens,
            } => {
                let mut usage_update = None;
                let mut history_update = None;
                if let Some(active) = self.active.get_mut(&request_id) {
                    active.status = RequestStatus::Streaming;
                    if active.generation_started_instant.is_none() {
                        active.generation_started_at = Some(SystemTime::now());
                        active.generation_started_instant = Some(Instant::now());
                        active.generation_initial_output_tokens =
                            output_tokens.or(active.output_tokens).unwrap_or(0);
                    } else {
                        active.generation_finished_at = Some(SystemTime::now());
                        active.generation_duration = active
                            .generation_started_instant
                            .map(|started| started.elapsed());
                    }
                    active.streamed_bytes = active.streamed_bytes.saturating_add(bytes);
                    active.stream_chunks = active.stream_chunks.saturating_add(chunks);
                    let input_delta = update_token_count(&mut active.input_tokens, input_tokens);
                    let output_delta = update_token_count(&mut active.output_tokens, output_tokens);
                    usage_update = Some((active.session_id.clone(), input_delta, output_delta));
                } else if let Some(completed) = self
                    .recent
                    .iter_mut()
                    .find(|request| request.request_id == request_id)
                {
                    if let Some(started) = completed.generation_started_instant {
                        completed.generation_finished_at = Some(SystemTime::now());
                        completed.generation_duration = Some(started.elapsed());
                    }
                    completed.streamed_bytes = completed.streamed_bytes.saturating_add(bytes);
                    completed.stream_chunks = completed.stream_chunks.saturating_add(chunks);
                    let input_delta = update_token_count(&mut completed.input_tokens, input_tokens);
                    let output_delta =
                        update_token_count(&mut completed.output_tokens, output_tokens);
                    usage_update = Some((completed.session_id.clone(), input_delta, output_delta));
                    if output_delta > 0 {
                        history_update = Some((
                            completed.session_id.clone(),
                            completed
                                .generation_finished_at
                                .unwrap_or(completed.finished_at),
                            output_delta,
                        ));
                    }
                }
                if let Some((session_id, input_delta, output_delta)) = usage_update {
                    self.record_session_usage(session_id, input_delta, output_delta);
                }
                if let Some((session_id, timestamp, tokens)) = history_update {
                    self.record_session_output(session_id, timestamp, tokens);
                }
            }
            MonitorEvent::UsageUpdated {
                request_id,
                input_tokens,
                output_tokens,
            } => {
                let mut usage_update = None;
                let mut history_update = None;
                if let Some(active) = self.active.get_mut(&request_id) {
                    if output_tokens.is_some()
                        && let Some(started) = active.generation_started_instant
                    {
                        active.generation_finished_at = Some(SystemTime::now());
                        active.generation_duration = Some(started.elapsed());
                    }
                    let input_delta = update_token_count(&mut active.input_tokens, input_tokens);
                    let output_delta = update_token_count(&mut active.output_tokens, output_tokens);
                    usage_update = Some((active.session_id.clone(), input_delta, output_delta));
                } else if let Some(completed) = self
                    .recent
                    .iter_mut()
                    .find(|request| request.request_id == request_id)
                {
                    if output_tokens.is_some()
                        && let Some(started) = completed.generation_started_instant
                    {
                        completed.generation_finished_at = Some(SystemTime::now());
                        completed.generation_duration = Some(started.elapsed());
                    }
                    let input_delta = update_token_count(&mut completed.input_tokens, input_tokens);
                    let output_delta =
                        update_token_count(&mut completed.output_tokens, output_tokens);
                    usage_update = Some((completed.session_id.clone(), input_delta, output_delta));
                    if output_delta > 0 {
                        history_update = Some((
                            completed.session_id.clone(),
                            completed
                                .generation_finished_at
                                .unwrap_or(completed.finished_at),
                            output_delta,
                        ));
                    }
                }
                if let Some((session_id, input_delta, output_delta)) = usage_update {
                    self.record_session_usage(session_id, input_delta, output_delta);
                }
                if let Some((session_id, timestamp, tokens)) = history_update {
                    self.record_session_output(session_id, timestamp, tokens);
                }
            }
            MonitorEvent::RequestCompleted {
                request_id,
                http_status,
                input_tokens,
                output_tokens,
            } => {
                return self.finish(
                    &request_id,
                    RequestStatus::Completed,
                    Some(http_status),
                    input_tokens,
                    output_tokens,
                    None,
                );
            }
            MonitorEvent::RequestFailed {
                request_id,
                http_status,
                error,
            } => {
                return self.finish(
                    &request_id,
                    RequestStatus::Failed,
                    http_status,
                    None,
                    None,
                    Some(error),
                );
            }
            MonitorEvent::RequestAbandoned { request_id, error } => {
                return self.finish_active(
                    &request_id,
                    RequestStatus::Failed,
                    None,
                    None,
                    None,
                    Some(error),
                );
            }
        }
        None
    }

    fn finish_active(
        &mut self,
        request_id: &str,
        status: RequestStatus,
        http_status: Option<u16>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        error: Option<String>,
    ) -> Option<CompletedRequest> {
        if self.active.contains_key(request_id) {
            return self.finish(
                request_id,
                status,
                http_status,
                input_tokens,
                output_tokens,
                error,
            );
        }
        None
    }

    fn finish(
        &mut self,
        request_id: &str,
        status: RequestStatus,
        http_status: Option<u16>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        error: Option<String>,
    ) -> Option<CompletedRequest> {
        let mut active = self
            .active
            .remove(request_id)
            .unwrap_or_else(|| ActiveRequest {
                request_id: request_id.to_string(),
                session_id: None,
                session_seq: None,
                project: None,
                provider: None,
                model: None,
                effort: None,
                endpoint: EndpointKind::Messages,
                started_at: SystemTime::now(),
                started_instant: Instant::now(),
                generation_started_at: None,
                generation_started_instant: None,
                generation_initial_output_tokens: 0,
                generation_finished_at: None,
                generation_duration: None,
                status: RequestStatus::Started,
                streamed_bytes: 0,
                stream_chunks: 0,
                input_tokens: None,
                output_tokens: None,
                error: None,
                traffic_capture_path: None,
                codex: None,
            });
        if let Some(diagnostics) = active.codex.as_mut() {
            diagnostics.finish();
        }
        if output_tokens.is_some()
            && let Some(started) = active.generation_started_instant
        {
            active.generation_finished_at = Some(SystemTime::now());
            active.generation_duration = Some(started.elapsed());
        }
        let input_delta = update_token_count(&mut active.input_tokens, input_tokens);
        let output_delta = update_token_count(&mut active.output_tokens, output_tokens);
        self.record_session_usage(active.session_id.clone(), input_delta, output_delta);
        let completed = CompletedRequest {
            request_id: active.request_id,
            session_id: active.session_id,
            session_seq: active.session_seq,
            project: active.project,
            provider: active.provider,
            model: active.model,
            effort: active.effort,
            endpoint: active.endpoint,
            started_at: active.started_at,
            finished_at: SystemTime::now(),
            generation_started_at: active.generation_started_at,
            generation_started_instant: active.generation_started_instant,
            generation_initial_output_tokens: active.generation_initial_output_tokens,
            generation_finished_at: active.generation_finished_at,
            generation_duration: active.generation_duration,
            status,
            http_status,
            latency: active.started_instant.elapsed(),
            streamed_bytes: active.streamed_bytes,
            stream_chunks: active.stream_chunks,
            input_tokens: active.input_tokens,
            output_tokens: active.output_tokens,
            error: error.or(active.error),
            traffic_capture_path: active.traffic_capture_path,
            codex: active.codex,
        };
        if let Some(tokens) = completed.output_tokens.filter(|tokens| *tokens > 0) {
            self.record_session_output(
                completed.session_id.clone(),
                completed
                    .generation_finished_at
                    .unwrap_or(completed.finished_at),
                tokens,
            );
        }
        self.recent.push_front(completed.clone());
        while self.recent.len() > self.recent_limit {
            self.recent.pop_back();
        }
        Some(completed)
    }

    fn record_session_usage(
        &mut self,
        session_id: Option<String>,
        input_tokens: u64,
        output_tokens: u64,
    ) {
        let usage = self.session_usage.entry(session_id).or_default();
        usage.input_tokens = usage.input_tokens.saturating_add(input_tokens);
        usage.output_tokens = usage.output_tokens.saturating_add(output_tokens);
    }

    fn record_session_output(
        &mut self,
        session_id: Option<String>,
        timestamp: SystemTime,
        tokens: u64,
    ) {
        let bucket = session_token_bucket(timestamp);
        let buckets = self.session_output_buckets.entry(session_id).or_default();
        match buckets.binary_search_by_key(&bucket, |(bucket, _)| *bucket) {
            Ok(index) => buckets[index].1 = buckets[index].1.saturating_add(tokens),
            Err(index) => buckets.insert(index, (bucket, tokens)),
        }
    }

    fn snapshot(&self) -> MonitorState {
        let mut active: Vec<_> = self.active.values().cloned().collect();
        active.sort_by_key(|request| request.started_at);
        let sessions = session_summaries(
            &active,
            &self.recent,
            &self.session_usage,
            &self.session_output_buckets,
        );
        MonitorState {
            started_at: self.started_at,
            sessions,
            active,
            recent: self.recent.iter().cloned().collect(),
            codex: self.codex.totals,
        }
    }
}

fn session_summaries(
    active: &[ActiveRequest],
    recent: &VecDeque<CompletedRequest>,
    session_usage: &HashMap<Option<String>, SessionUsage>,
    session_output_buckets: &HashMap<Option<String>, Vec<(u64, u64)>>,
) -> Vec<SessionSummary> {
    let mut sessions: HashMap<Option<String>, SessionSummary> = HashMap::new();
    for request in recent.iter().rev() {
        let entry = sessions
            .entry(request.session_id.clone())
            .or_insert_with(|| SessionSummary {
                session_id: request.session_id.clone(),
                project: request.project.clone(),
                active_count: 0,
                request_count: 0,
                failure_count: 0,
                provider: None,
                model: None,
                effort: None,
                last_seen: request.finished_at,
                input_tokens: 0,
                output_tokens: 0,
                output_token_samples: Vec::new(),
                rate_output_tokens: 0,
                generation_duration: Duration::ZERO,
                last_status: "-".to_string(),
            });
        entry.request_count += 1;
        if request.status == RequestStatus::Failed {
            entry.failure_count += 1;
        }
        entry.project = request.project.clone().or(entry.project.clone());
        entry.provider = request.provider.clone().or(entry.provider.clone());
        entry.model = request.model.clone().or(entry.model.clone());
        entry.effort = request.effort.clone().or(entry.effort.clone());
        entry.last_seen = max_system_time(entry.last_seen, request.finished_at);
        if let (Some(tokens), Some(duration)) = (
            request
                .output_tokens
                .and_then(|tokens| tokens.checked_sub(request.generation_initial_output_tokens))
                .filter(|tokens| *tokens > 0),
            request
                .generation_duration
                .filter(|duration| !duration.is_zero()),
        ) {
            entry.rate_output_tokens = entry.rate_output_tokens.saturating_add(tokens);
            entry.generation_duration = entry.generation_duration.saturating_add(duration);
        }
        entry.last_status = request.status.label().to_string();
    }

    for request in active {
        let entry = sessions
            .entry(request.session_id.clone())
            .or_insert_with(|| SessionSummary {
                session_id: request.session_id.clone(),
                project: request.project.clone(),
                active_count: 0,
                request_count: 0,
                failure_count: 0,
                provider: None,
                model: None,
                effort: None,
                last_seen: request.started_at,
                input_tokens: 0,
                output_tokens: 0,
                output_token_samples: Vec::new(),
                rate_output_tokens: 0,
                generation_duration: Duration::ZERO,
                last_status: "-".to_string(),
            });
        entry.active_count += 1;
        entry.request_count += 1;
        entry.project = request.project.clone().or(entry.project.clone());
        entry.provider = request.provider.clone().or(entry.provider.clone());
        entry.model = request.model.clone().or(entry.model.clone());
        entry.effort = request.effort.clone().or(entry.effort.clone());
        entry.last_seen = max_system_time(entry.last_seen, request.started_at);
        if let (Some(tokens), Some(duration)) = (
            request
                .output_tokens
                .and_then(|tokens| tokens.checked_sub(request.generation_initial_output_tokens))
                .filter(|tokens| *tokens > 0),
            request
                .generation_duration
                .filter(|duration| !duration.is_zero()),
        ) {
            entry.rate_output_tokens = entry.rate_output_tokens.saturating_add(tokens);
            entry.generation_duration = entry.generation_duration.saturating_add(duration);
        }
        entry.last_status = request.status.label().to_string();
    }

    for (session_id, session) in &mut sessions {
        if let Some(usage) = session_usage.get(session_id) {
            session.input_tokens = usage.input_tokens;
            session.output_tokens = usage.output_tokens;
        }
        if let Some(buckets) = session_output_buckets.get(session_id) {
            session.output_token_samples = buckets
                .iter()
                .map(|(bucket, tokens)| (session_token_bucket_start(*bucket), *tokens))
                .collect();
        }
    }

    let mut out: Vec<_> = sessions.into_values().collect();
    out.sort_by_key(SessionSummary::label);
    out
}

fn session_token_bucket(timestamp: SystemTime) -> u64 {
    timestamp
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
        / SESSION_TOKEN_BUCKET_SECS
}

fn session_token_bucket_start(bucket: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(bucket.saturating_mul(SESSION_TOKEN_BUCKET_SECS))
}

fn update_token_count(current: &mut Option<u64>, incoming: Option<u64>) -> u64 {
    let Some(incoming) = incoming else {
        return 0;
    };
    let previous = current.unwrap_or(0);
    if incoming > previous || current.is_none() {
        *current = Some(incoming);
    }
    incoming.saturating_sub(previous)
}

fn max_system_time(left: SystemTime, right: SystemTime) -> SystemTime {
    if right.duration_since(left).is_ok() {
        right
    } else {
        left
    }
}

pub fn throughput(
    output_tokens: Option<u64>,
    streamed_bytes: u64,
    stream_chunks: u64,
    elapsed: Duration,
) -> Throughput {
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        return Throughput::None;
    }
    if let Some(tokens) = output_tokens.filter(|tokens| *tokens > 0) {
        return Throughput::TokensPerSecond(tokens as f64 / secs);
    }
    if streamed_bytes > 0 {
        return Throughput::BytesPerSecond(streamed_bytes as f64 / secs);
    }
    if stream_chunks > 0 {
        return Throughput::EventsPerSecond(stream_chunks as f64 / secs);
    }
    Throughput::None
}

pub fn usage_from_anthropic_sse(bytes: &[u8]) -> (Option<u64>, Option<u64>) {
    let text = String::from_utf8_lossy(bytes);
    let mut input_tokens = None;
    let mut output_tokens = None;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(data.trim()) else {
            continue;
        };
        for usage in [
            value.pointer("/usage"),
            value.pointer("/delta/usage"),
            value.pointer("/message/usage"),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(tokens) = usage.get("input_tokens").and_then(|value| value.as_u64()) {
                input_tokens = Some(tokens);
            }
            if let Some(tokens) = usage.get("output_tokens").and_then(|value| value.as_u64()) {
                output_tokens = Some(tokens);
            }
        }
    }
    (input_tokens, output_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_metrics_count_previous_id_outcomes_and_owner_transitions() {
        let monitor = MonitorHandle::new(10);
        monitor.codex_previous_id_settled("prev-none", CodexPreviousIdOutcome::NoCandidate);
        monitor.codex_previous_id_settled("prev-hit", CodexPreviousIdOutcome::Hit);
        monitor.codex_previous_id_settled("prev-fallback", CodexPreviousIdOutcome::Fallback);

        let main = ConversationIdentity::Main("session-a".into());
        let agent = ConversationIdentity::Agent("session-a".into(), "agent-a".into());
        let sibling = ConversationIdentity::Agent("session-a".into(), "agent-b".into());
        let same_agent_other_session =
            ConversationIdentity::Agent("session-b".into(), "agent-a".into());
        let route_a = [1; 32];
        let route_b = [2; 32];
        let route_c = [3; 32];

        monitor.codex_websocket_dispatch("main-a", main.clone(), route_a, true, 11);
        monitor.codex_websocket_dispatch("agent-a", agent.clone(), route_a, true, 21);
        monitor.codex_websocket_dispatch("sibling-a", sibling.clone(), route_a, true, 31);
        monitor.codex_websocket_dispatch(
            "other-a",
            same_agent_other_session.clone(),
            route_a,
            true,
            41,
        );
        assert_eq!(
            monitor.snapshot().codex,
            CodexMetricsSnapshot {
                previous_id_no_candidates: 1,
                previous_id_hits: 1,
                previous_id_fallbacks: 1,
                route_baselines: 4,
                socket_baselines: 4,
                ..CodexMetricsSnapshot::default()
            }
        );

        monitor.codex_websocket_dispatch("main-b", main, route_a, true, 11);
        monitor.codex_websocket_dispatch("agent-b", agent, route_a, true, 22);
        monitor.codex_websocket_dispatch("sibling-b", sibling, route_b, true, 31);
        monitor.codex_websocket_dispatch("other-b", same_agent_other_session, route_c, false, 42);

        assert_eq!(
            monitor.snapshot().codex,
            CodexMetricsSnapshot {
                previous_id_no_candidates: 1,
                previous_id_hits: 1,
                previous_id_fallbacks: 1,
                route_baselines: 4,
                route_reuses: 2,
                route_switches: 2,
                lane_switches: 1,
                socket_baselines: 4,
                socket_reuses: 2,
                socket_switches: 2,
            }
        );
    }

    #[test]
    fn codex_metrics_rebuild_baselines_after_owner_state_expires() {
        let monitor = MonitorHandle::new(10);
        let main = ConversationIdentity::Main("session-a".into());
        let agent = ConversationIdentity::Agent("session-a".into(), "agent-a".into());
        let route_a = [1; 32];
        let route_b = [2; 32];

        monitor.codex_websocket_dispatch("main-a", main.clone(), route_a, true, 11);
        monitor.codex_websocket_dispatch("agent-a", agent.clone(), route_a, true, 21);
        {
            let mut store = monitor.store.lock().unwrap();
            store
                .codex
                .owner_dispatches
                .get_mut(&main)
                .unwrap()
                .updated_at = Instant::now() - CODEX_OWNER_STATE_TTL - Duration::from_secs(1);
        }
        monitor.codex_websocket_dispatch("main-b", main, route_b, false, 12);
        monitor.codex_websocket_dispatch("agent-b", agent.clone(), route_a, true, 21);
        {
            let mut store = monitor.store.lock().unwrap();
            store
                .codex
                .owner_dispatches
                .get_mut(&agent)
                .unwrap()
                .updated_at = Instant::now() - CODEX_OWNER_STATE_TTL - Duration::from_secs(1);
        }
        monitor.codex_websocket_dispatch("agent-c", agent, route_b, false, 22);

        assert_eq!(
            monitor.snapshot().codex,
            CodexMetricsSnapshot {
                route_baselines: 4,
                route_reuses: 1,
                socket_baselines: 4,
                socket_reuses: 1,
                ..CodexMetricsSnapshot::default()
            }
        );
    }

    #[test]
    fn codex_request_diagnostics_follow_request_across_retries_and_completion() {
        let monitor = MonitorHandle::new(10);
        let owner = ConversationIdentity::Main("session-a".into());
        let route_a = [1; 32];
        let route_b = [2; 32];

        monitor.request_started(
            "r-codex",
            Some("session-a".to_string()),
            Some(1),
            EndpointKind::Messages,
        );
        monitor.codex_request_lane("r-codex", true);
        monitor.codex_previous_id_pending("r-codex");
        monitor.codex_websocket_dispatch("r-codex", owner.clone(), route_a, true, 11);
        monitor.codex_websocket_dispatch("r-codex", owner.clone(), route_a, true, 11);
        monitor.codex_websocket_dispatch("r-codex", owner, route_b, false, 12);

        let state = monitor.snapshot();
        let diagnostics = state.active[0].codex_diagnostics().unwrap();
        assert_eq!(diagnostics.lane, Some(CodexLane::Full));
        assert_eq!(diagnostics.previous_id, CodexRequestPreviousId::Pending);
        assert_eq!(
            diagnostics.route,
            CodexDispatchSummary {
                first: Some(CodexDispatchOutcome::Baseline),
                latest: Some(CodexDispatchOutcome::Switch),
                dispatches: 3,
                baselines: 1,
                reuses: 1,
                switches: 1,
            }
        );
        assert_eq!(diagnostics.socket, diagnostics.route);

        monitor.request_completed("r-codex", 200, None, None);
        let state = monitor.snapshot();
        assert!(state.active.is_empty());
        assert_eq!(
            state.recent[0].codex_diagnostics().unwrap().previous_id,
            CodexRequestPreviousId::Unsettled
        );

        monitor.codex_previous_id_settled("r-codex", CodexPreviousIdOutcome::Hit);
        assert_eq!(
            monitor.snapshot().recent[0]
                .codex_diagnostics()
                .unwrap()
                .previous_id,
            CodexRequestPreviousId::Hit
        );
    }

    #[test]
    fn codex_acceleration_follows_request_into_completion_and_diagnostics() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r-fast", None, None, EndpointKind::Messages);
        monitor.provider_selected("r-fast", "codex", "gpt-5.5", None);
        monitor.codex_acceleration_resolved("r-fast", Some("fast"), Some("priority"));

        let active = monitor.snapshot();
        let diagnostics = *active.active[0].codex_diagnostics().unwrap();
        assert_eq!(
            diagnostics.acceleration,
            RequestAcceleration {
                requested_speed: Some("fast"),
                codex_service_tier: Some("priority"),
            }
        );
        let json = codex_diagnostics_json(diagnostics);
        assert_eq!(json["requestedSpeed"], "fast");
        assert_eq!(json["serviceTier"], "priority");

        monitor.request_completed("r-fast", 200, None, None);
        assert_eq!(
            monitor.snapshot().recent[0]
                .codex_diagnostics()
                .unwrap()
                .acceleration,
            diagnostics.acceleration
        );
    }

    #[test]
    fn ordinary_request_does_not_create_acceleration_diagnostics() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r-standard", None, None, EndpointKind::Messages);
        monitor.codex_acceleration_resolved("r-standard", None, None);

        assert!(monitor.snapshot().active[0].codex_diagnostics().is_none());
    }

    #[test]
    fn codex_recovery_causes_are_first_wins_deduped_and_late_safe() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r-recovery", None, None, EndpointKind::Messages);
        monitor.codex_previous_id_cause("r-recovery", CodexRecoveryCause::PromptChanged);
        monitor.codex_previous_id_cause("r-recovery", CodexRecoveryCause::AuthRejection);
        monitor.codex_socket_cause("r-recovery", CodexRecoveryCause::TransportFailure);
        monitor.codex_socket_cause("r-recovery", CodexRecoveryCause::OriginSocketMissing);
        monitor.codex_socket_cause("r-recovery", CodexRecoveryCause::TransportFailure);

        let active = monitor.snapshot();
        let diagnostics = active.active[0].codex_diagnostics().unwrap();
        let recovery = diagnostics.recovery;
        assert_eq!(diagnostics.event_sequence, 5);
        assert_eq!(
            recovery.previous_id_cause,
            Some(CodexRecoveryCause::PromptChanged)
        );
        assert_eq!(
            recovery.socket_causes.iter().collect::<Vec<_>>(),
            vec![
                CodexRecoveryCause::OriginSocketMissing,
                CodexRecoveryCause::TransportFailure,
            ]
        );

        monitor.request_completed("r-recovery", 200, None, None);
        monitor.codex_previous_id_cause("r-recovery", CodexRecoveryCause::MissingState);
        monitor.codex_socket_cause("r-recovery", CodexRecoveryCause::AuthRejection);
        let recent = monitor.snapshot();
        let diagnostics = recent.recent[0].codex_diagnostics().unwrap();
        let recovery = diagnostics.recovery;
        assert_eq!(diagnostics.event_sequence, 7);
        assert_eq!(
            recovery.previous_id_cause,
            Some(CodexRecoveryCause::PromptChanged)
        );
        assert_eq!(
            recovery.socket_causes.iter().collect::<Vec<_>>(),
            vec![
                CodexRecoveryCause::AuthRejection,
                CodexRecoveryCause::OriginSocketMissing,
                CodexRecoveryCause::TransportFailure,
            ]
        );
    }

    #[test]
    fn codex_detailed_diagnostics_are_bounded_and_privacy_safe() {
        let monitor = MonitorHandle::new(10);
        let owner = ConversationIdentity::Main("private-session".into());
        monitor.request_started(
            "r-detail",
            Some("private-session".into()),
            Some(9),
            EndpointKind::Messages,
        );
        monitor.provider_selected("r-detail", "codex", "private-model-alias", None);
        monitor.codex_append_only(
            "r-detail",
            CodexAppendOnlyDiagnostics {
                outcome: CodexAppendOnlyOutcome::FirstMismatch,
                incoming_items: 14,
                retained_items: 11,
                delta_items: 3,
                first_mismatch_index: Some(6),
            },
        );
        monitor.codex_pool_resolution("r-detail", CodexPoolResolution::ExactOriginMatched);
        monitor.codex_socket_validation("r-detail", None, 7, true);
        monitor.codex_socket_validation(
            "r-detail",
            Some(CodexSocketValidationFailure::UnexpectedText),
            12,
            true,
        );
        monitor.codex_websocket_dispatch("r-detail", owner, [9; 32], true, 9_999);
        monitor.request_failed("r-detail", Some(502), "private upstream error body");
        monitor.codex_pool_resolution("r-detail", CodexPoolResolution::ExactOriginUnavailable);

        let state = monitor.snapshot();
        let request = &state.recent[0];
        let diagnostics = request.codex_diagnostics().unwrap();
        assert_eq!(
            diagnostics.append_only,
            Some(CodexAppendOnlyDiagnostics {
                outcome: CodexAppendOnlyOutcome::FirstMismatch,
                incoming_items: 14,
                retained_items: 11,
                delta_items: 3,
                first_mismatch_index: Some(6),
            })
        );
        assert_eq!(diagnostics.pool.lookups, 2);
        assert_eq!(diagnostics.pool.hits, 1);
        assert_eq!(diagnostics.pool.misses, 1);
        assert_eq!(diagnostics.pool.exact_matches, 1);
        assert_eq!(diagnostics.pool.exact_unavailable, 1);
        assert_eq!(diagnostics.validation.attempts, 2);
        assert_eq!(diagnostics.validation.successes, 1);
        assert_eq!(diagnostics.validation.failures, 1);
        assert_eq!(
            diagnostics.validation.latest_failure,
            Some(CodexSocketValidationFailure::UnexpectedText)
        );
        assert_eq!(diagnostics.validation.latest_elapsed_ms, 12);
        assert!(diagnostics.validation.latest_required_origin);

        let mut event_specific = serde_json::Map::new();
        event_specific.insert(
            "cause".into(),
            serde_json::json!(CodexRecoveryCause::SocketValidationFailed.label()),
        );
        let event_fields = serde_json::Value::Object(codex_event_fields(
            "r-detail",
            *diagnostics,
            false,
            event_specific,
        ));
        assert_eq!(event_fields["schemaVersion"], 1);
        assert_eq!(event_fields["reqId"], "r-detail");
        assert_eq!(event_fields["eventSequence"], 6);
        assert_eq!(event_fields["stateChanged"], false);
        assert_eq!(event_fields["cause"], "socket_validation_failed");

        let fields = serde_json::Value::Object(codex_request_snapshot_fields(request).unwrap());
        assert_eq!(fields["schemaVersion"], 1);
        assert_eq!(
            fields["diagnostics"]["appendOnly"]["outcome"],
            "first_mismatch"
        );
        assert_eq!(
            fields["diagnostics"]["validation"]["latestFailure"],
            "unexpected_text"
        );
        assert_eq!(
            fields["diagnostics"]["pool"]["latest"],
            "exact_origin_unavailable"
        );
        let serialized = fields.to_string();
        assert!(!serialized.contains("private-session"));
        assert!(!serialized.contains("private-model-alias"));
        assert!(!serialized.contains("private upstream error body"));
        assert!(!serialized.contains("9999"));
    }

    #[test]
    fn codex_dispatch_does_not_infer_causes_and_evicted_updates_are_noops() {
        let monitor = MonitorHandle::new(1);
        let owner = ConversationIdentity::Main("session-recovery".into());
        monitor.request_started("r-old", None, None, EndpointKind::Messages);
        monitor.codex_websocket_dispatch("r-old", owner.clone(), [1; 32], true, 11);
        monitor.codex_websocket_dispatch("r-old", owner, [2; 32], false, 12);
        assert_eq!(
            monitor.snapshot().active[0]
                .codex_diagnostics()
                .unwrap()
                .recovery,
            CodexRecoveryDiagnostics::default()
        );
        monitor.request_completed("r-old", 200, None, None);

        monitor.request_started("r-new", None, None, EndpointKind::Messages);
        monitor.request_completed("r-new", 200, None, None);
        monitor.codex_previous_id_cause("r-old", CodexRecoveryCause::RouteChanged);
        monitor.codex_socket_cause("r-old", CodexRecoveryCause::Cancelled);

        let state = monitor.snapshot();
        assert_eq!(state.recent.len(), 1);
        assert_eq!(state.recent[0].request_id, "r-new");
        assert!(state.recent[0].codex_diagnostics().is_none());
    }

    #[test]
    fn started_requests_appear_active() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "r1",
            Some("s1".to_string()),
            Some(3),
            EndpointKind::Messages,
        );
        let state = monitor.snapshot();
        assert_eq!(state.active.len(), 1);
        assert_eq!(state.active[0].request_id, "r1");
        assert_eq!(state.active[0].session_id.as_deref(), Some("s1"));
        assert_eq!(state.active[0].session_seq, Some(3));
    }

    #[test]
    fn resolved_model_appends_to_incoming_alias() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.provider_selected("r1", "codex", "claude-sonnet-4-6", None);
        monitor.model_resolved("r1", "gpt-5.4");

        let state = monitor.snapshot();
        assert_eq!(
            state.active[0].model.as_deref(),
            Some("claude-sonnet-4-6 → gpt-5.4")
        );
    }

    #[test]
    fn identical_resolved_model_is_shown_once() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.provider_selected("r1", "codex", "gpt-5.6-sol", None);
        monitor.model_resolved("r1", "gpt-5.6-sol");

        let state = monitor.snapshot();
        assert_eq!(state.active[0].model.as_deref(), Some("gpt-5.6-sol"));
    }

    #[test]
    fn compaction_started_marks_request_compacting() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.provider_selected("r1", "codex", "gpt-5.6-sol", None);
        monitor.compaction_started("r1");

        let state = monitor.snapshot();
        assert_eq!(state.active[0].status, RequestStatus::Compacting);
        assert_eq!(state.sessions[0].last_status, "compacting");
    }

    #[test]
    fn generation_baseline_pairs_total_usage_with_the_full_observed_interval() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.generation_started("r1");
        monitor.stream_progress("r1", 50, 1, Some(1_225), Some(141));
        monitor.request_completed("r1", 200, None, None);

        let request = &monitor.snapshot().recent[0];

        assert!(
            request
                .generation_duration
                .is_some_and(|duration| !duration.is_zero())
        );
        assert!(matches!(request.rate(), Throughput::TokensPerSecond(_)));
    }

    #[test]
    fn first_stream_progress_has_no_rate_without_an_interval() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.stream_progress("r1", 50, 1, Some(1_225), Some(141));

        let state = monitor.snapshot();
        assert_eq!(state.active.len(), 1);
        assert_eq!(state.active[0].rate(), Throughput::None);
    }

    #[test]
    fn late_stream_progress_extends_stream_timing_without_extending_request_latency() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.stream_progress("r1", 100, 1, Some(0), Some(0));
        monitor.request_completed("r1", 200, None, None);
        let completed = monitor.snapshot().recent[0].clone();
        monitor.stream_progress("r1", 50, 1, Some(1_225), Some(141));

        let state = monitor.snapshot();
        assert!(state.active.is_empty());
        assert_eq!(state.recent[0].streamed_bytes, 150);
        assert_eq!(state.recent[0].stream_chunks, 2);
        assert_eq!(state.recent[0].input_tokens, Some(1_225));
        assert_eq!(state.recent[0].output_tokens, Some(141));
        assert_eq!(state.recent[0].finished_at, completed.finished_at);
        assert_eq!(state.recent[0].latency, completed.latency);
        assert!(state.recent[0].generation_duration > completed.generation_duration);
        assert!(matches!(
            state.recent[0].rate(),
            Throughput::TokensPerSecond(_)
        ));
        assert_eq!(
            state.sessions[0]
                .output_token_samples
                .iter()
                .map(|(_, tokens)| *tokens)
                .sum::<u64>(),
            141
        );
    }

    #[test]
    fn completed_requests_leave_active_and_enter_recent() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.provider_selected("r1", "codex", "gpt-5.5", Some("high".to_string()));
        monitor.request_completed("r1", 200, Some(10), Some(20));
        let state = monitor.snapshot();
        assert!(state.active.is_empty());
        assert_eq!(state.recent.len(), 1);
        assert_eq!(state.recent[0].provider.as_deref(), Some("codex"));
        assert_eq!(state.recent[0].effort.as_deref(), Some("high"));
        assert_eq!(state.recent[0].output_tokens, Some(20));
    }

    #[test]
    fn failed_requests_preserve_error_summary() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.request_failed("r1", Some(400), "Unknown model");
        let state = monitor.snapshot();
        assert_eq!(state.recent[0].status, RequestStatus::Failed);
        assert_eq!(state.recent[0].http_status, Some(400));
        assert_eq!(state.recent[0].error.as_deref(), Some("Unknown model"));
    }

    #[test]
    fn abandoned_requests_leave_active_once() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.request_abandoned("r1", "request dropped");
        monitor.request_abandoned("r1", "request dropped again");
        let state = monitor.snapshot();
        assert!(state.active.is_empty());
        assert_eq!(state.recent.len(), 1);
        assert_eq!(state.recent[0].status, RequestStatus::Failed);
        assert_eq!(state.recent[0].http_status, None);
        assert_eq!(state.recent[0].error.as_deref(), Some("request dropped"));
    }

    #[test]
    fn completed_requests_ignore_late_abandonment() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started("r1", None, None, EndpointKind::Messages);
        monitor.request_completed("r1", 200, None, None);
        monitor.request_abandoned("r1", "request dropped");
        let state = monitor.snapshot();
        assert!(state.active.is_empty());
        assert_eq!(state.recent.len(), 1);
        assert_eq!(state.recent[0].status, RequestStatus::Completed);
    }

    #[test]
    fn bounded_recent_history_drops_oldest() {
        let monitor = MonitorHandle::new(2);
        for id in ["r1", "r2", "r3"] {
            monitor.request_started(id, None, None, EndpointKind::Messages);
            monitor.request_completed(id, 200, None, None);
        }
        let state = monitor.snapshot();
        let ids: Vec<_> = state
            .recent
            .iter()
            .map(|request| request.request_id.as_str())
            .collect();
        assert_eq!(ids, vec!["r3", "r2"]);
    }

    #[test]
    fn throughput_selects_best_available_signal() {
        let elapsed = Duration::from_secs(2);
        assert_eq!(
            throughput(Some(84), 1024, 10, elapsed),
            Throughput::TokensPerSecond(42.0)
        );
        assert_eq!(
            throughput(None, 2048, 10, elapsed),
            Throughput::BytesPerSecond(1024.0)
        );
        assert_eq!(
            throughput(None, 0, 36, elapsed),
            Throughput::EventsPerSecond(18.0)
        );
    }

    #[test]
    fn sse_usage_extracts_final_message_delta_tokens() {
        let sse = br#"event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":0,"output_tokens":0}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":12,"output_tokens":48}}

"#;
        assert_eq!(usage_from_anthropic_sse(sse), (Some(12), Some(48)));
    }

    fn completed_request(
        request_id: &str,
        session_id: &str,
        output_tokens: u64,
        latency: Duration,
        generation_duration: Option<Duration>,
    ) -> CompletedRequest {
        CompletedRequest {
            request_id: request_id.to_string(),
            session_id: Some(session_id.to_string()),
            session_seq: None,
            project: None,
            provider: Some("codex".to_string()),
            model: Some("gpt-5.6-sol".to_string()),
            effort: None,
            endpoint: EndpointKind::Messages,
            started_at: SystemTime::UNIX_EPOCH,
            finished_at: SystemTime::UNIX_EPOCH + latency,
            generation_started_at: generation_duration.map(|_| SystemTime::UNIX_EPOCH),
            generation_started_instant: None,
            generation_initial_output_tokens: 0,
            generation_finished_at: generation_duration
                .map(|duration| SystemTime::UNIX_EPOCH + duration),
            generation_duration,
            status: RequestStatus::Completed,
            http_status: Some(200),
            latency,
            streamed_bytes: 0,
            stream_chunks: 0,
            input_tokens: None,
            output_tokens: Some(output_tokens),
            error: None,
            traffic_capture_path: None,
            codex: None,
        }
    }

    fn session_summaries_for_requests(recent: &VecDeque<CompletedRequest>) -> Vec<SessionSummary> {
        let mut usage = HashMap::<Option<String>, SessionUsage>::new();
        for request in recent {
            let entry = usage.entry(request.session_id.clone()).or_default();
            entry.input_tokens = entry
                .input_tokens
                .saturating_add(request.input_tokens.unwrap_or(0));
            entry.output_tokens = entry
                .output_tokens
                .saturating_add(request.output_tokens.unwrap_or(0));
        }
        session_summaries(&[], recent, &usage, &HashMap::new())
    }

    #[test]
    fn completed_request_rate_uses_stream_interval_instead_of_request_latency() {
        let request = completed_request(
            "r1",
            "s1",
            120,
            Duration::from_secs(30),
            Some(Duration::from_secs(4)),
        );

        assert_eq!(request.rate(), Throughput::TokensPerSecond(30.0));
    }

    #[test]
    fn request_rate_uses_token_delta_from_the_initial_observation() {
        let mut request = completed_request(
            "r1",
            "s1",
            120,
            Duration::from_secs(30),
            Some(Duration::from_secs(4)),
        );
        request.generation_initial_output_tokens = 20;

        assert_eq!(request.rate(), Throughput::TokensPerSecond(25.0));
    }

    #[test]
    fn session_rate_combines_request_tokens_and_generation_intervals() {
        let recent = VecDeque::from([
            completed_request(
                "r2",
                "s1",
                50,
                Duration::from_secs(40),
                Some(Duration::from_secs(1)),
            ),
            completed_request(
                "r1",
                "s1",
                100,
                Duration::from_secs(20),
                Some(Duration::from_secs(4)),
            ),
        ]);

        let sessions = session_summaries_for_requests(&recent);

        assert_eq!(sessions[0].output_tokens, 150);
        assert_eq!(sessions[0].generation_duration, Duration::from_secs(5));
        assert_eq!(sessions[0].rate(), Throughput::TokensPerSecond(30.0));
    }

    #[test]
    fn output_without_observed_stream_interval_has_no_output_rate() {
        let request = completed_request("r1", "s1", 120, Duration::from_secs(30), None);
        let recent = VecDeque::from([request.clone()]);

        assert_eq!(request.rate(), Throughput::None);
        assert_eq!(
            session_summaries_for_requests(&recent)[0].rate(),
            Throughput::None
        );
    }

    #[test]
    fn session_rate_excludes_interval_without_output_usage() {
        let mut tokenless = completed_request(
            "tokenless",
            "s1",
            0,
            Duration::from_secs(30),
            Some(Duration::from_secs(100)),
        );
        tokenless.output_tokens = None;
        let recent = VecDeque::from([
            tokenless,
            completed_request(
                "measured",
                "s1",
                100,
                Duration::from_secs(20),
                Some(Duration::from_secs(4)),
            ),
        ]);

        assert_eq!(
            session_summaries_for_requests(&recent)[0].rate(),
            Throughput::TokensPerSecond(25.0)
        );
    }

    #[test]
    fn session_rate_excludes_output_without_a_matching_stream_interval() {
        let recent = VecDeque::from([
            completed_request("buffered", "s1", 900, Duration::from_secs(30), None),
            completed_request(
                "streamed",
                "s1",
                100,
                Duration::from_secs(20),
                Some(Duration::from_secs(4)),
            ),
        ]);

        let session = &session_summaries_for_requests(&recent)[0];

        assert_eq!(session.output_tokens, 1_000);
        assert_eq!(session.rate(), Throughput::TokensPerSecond(25.0));
    }

    #[test]
    fn session_summaries_group_recent_and_active_requests() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "r1",
            Some("s1".to_string()),
            Some(1),
            EndpointKind::Messages,
        );
        monitor.project_resolved("r1", "example");
        monitor.provider_selected("r1", "codex", "gpt-5.5", None);
        monitor.request_completed("r1", 200, Some(10), Some(20));
        monitor.request_started(
            "r2",
            Some("s1".to_string()),
            Some(2),
            EndpointKind::Messages,
        );
        monitor.provider_selected("r2", "codex", "gpt-5.5", Some("xhigh".to_string()));
        let state = monitor.snapshot();
        assert_eq!(state.sessions.len(), 1);
        assert_eq!(state.sessions[0].label(), "s1");
        assert_eq!(state.sessions[0].project.as_deref(), Some("example"));
        assert_eq!(state.sessions[0].request_count, 2);
        assert_eq!(state.sessions[0].active_count, 1);
        assert_eq!(state.sessions[0].effort.as_deref(), Some("xhigh"));
        assert_eq!(state.sessions[0].output_tokens, 20);
        assert_eq!(
            state.sessions[0]
                .output_token_samples
                .iter()
                .map(|(_, tokens)| *tokens)
                .collect::<Vec<_>>(),
            vec![20]
        );
    }

    #[test]
    fn session_output_history_survives_request_eviction() {
        let monitor = MonitorHandle::new(1);
        for (request_id, tokens) in [("oldest", 20), ("newest", 80)] {
            monitor.request_started(
                request_id,
                Some("s1".to_string()),
                None,
                EndpointKind::Messages,
            );
            monitor.request_completed(request_id, 200, Some(tokens * 10), Some(tokens));
        }

        let state = monitor.snapshot();

        assert_eq!(state.recent.len(), 1);
        assert_eq!(state.sessions[0].input_tokens, 1_000);
        assert_eq!(state.sessions[0].output_tokens, 100);
        assert_eq!(
            state.sessions[0]
                .output_token_samples
                .iter()
                .map(|(_, tokens)| *tokens)
                .sum::<u64>(),
            100
        );
    }

    #[test]
    fn session_usage_ignores_decreasing_request_observations() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "r1",
            Some("s1".to_string()),
            Some(1),
            EndpointKind::Messages,
        );
        monitor.usage_updated("r1", Some(100), Some(20));
        monitor.usage_updated("r1", Some(90), Some(15));
        monitor.usage_updated("r1", Some(120), Some(25));

        let active = monitor.snapshot();
        assert_eq!(active.active[0].input_tokens, Some(120));
        assert_eq!(active.active[0].output_tokens, Some(25));
        assert_eq!(active.sessions[0].input_tokens, 120);
        assert_eq!(active.sessions[0].output_tokens, 25);

        monitor.request_completed("r1", 200, Some(80), Some(10));
        let completed = monitor.snapshot();
        assert_eq!(completed.recent[0].input_tokens, Some(120));
        assert_eq!(completed.recent[0].output_tokens, Some(25));
        assert_eq!(completed.sessions[0].input_tokens, 120);
        assert_eq!(completed.sessions[0].output_tokens, 25);
    }

    #[test]
    fn compaction_preserves_cumulative_session_usage() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "before",
            Some("s1".to_string()),
            Some(1),
            EndpointKind::Messages,
        );
        monitor.request_completed("before", 200, Some(100), Some(20));
        monitor.request_started(
            "compact",
            Some("s1".to_string()),
            Some(2),
            EndpointKind::Messages,
        );
        monitor.compaction_started("compact");
        monitor.request_completed("compact", 200, Some(40), Some(10));

        let state = monitor.snapshot();
        assert_eq!(state.sessions[0].input_tokens, 140);
        assert_eq!(state.sessions[0].output_tokens, 30);
    }

    #[test]
    fn session_sequence_restart_preserves_cumulative_usage() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "before",
            Some("s1".to_string()),
            None,
            EndpointKind::Messages,
        );
        monitor.session_sequence_resolved("before", 7);
        monitor.request_completed("before", 200, Some(100), Some(20));

        monitor.request_started(
            "after",
            Some("s1".to_string()),
            None,
            EndpointKind::Messages,
        );
        monitor.session_sequence_resolved("after", 1);
        monitor.request_completed("after", 200, Some(25), Some(5));

        let state = monitor.snapshot();
        assert_eq!(state.sessions[0].input_tokens, 125);
        assert_eq!(state.sessions[0].output_tokens, 25);
    }

    #[test]
    fn session_order_is_stable_across_activity() {
        let monitor = MonitorHandle::new(10);
        monitor.request_started(
            "r1",
            Some("session-b".to_string()),
            Some(1),
            EndpointKind::Messages,
        );
        monitor.request_started(
            "r2",
            Some("session-a".to_string()),
            Some(1),
            EndpointKind::Messages,
        );

        let first: Vec<_> = monitor
            .snapshot()
            .sessions
            .iter()
            .map(SessionSummary::label)
            .collect();
        monitor.request_completed("r1", 200, None, None);
        monitor.request_started(
            "r3",
            Some("session-b".to_string()),
            Some(2),
            EndpointKind::Messages,
        );
        let second: Vec<_> = monitor
            .snapshot()
            .sessions
            .iter()
            .map(SessionSummary::label)
            .collect();

        assert_eq!(first, vec!["session-a", "session-b"]);
        assert_eq!(second, first);
    }
}
