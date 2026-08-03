use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodexFailureKind {
    RateLimit,
    Overloaded,
    Transient,
    Permanent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CodexEventFailure {
    pub kind: CodexFailureKind,
    pub explicit_status: Option<u16>,
    pub status: u16,
    pub message: String,
    pub retry_after: Option<String>,
}

impl CodexEventFailure {
    pub fn retryable(&self) -> bool {
        !matches!(self.kind, CodexFailureKind::Permanent)
    }
}

pub(crate) fn is_terminal_rate_limit_event(payload: &Value) -> bool {
    payload.get("type").and_then(Value::as_str) == Some("codex.rate_limits")
        && payload
            .pointer("/rate_limits/limit_reached")
            .and_then(Value::as_bool)
            == Some(true)
        && payload
            .pointer("/credits/has_credits")
            .and_then(Value::as_bool)
            != Some(true)
        && payload
            .pointer("/credits/unlimited")
            .and_then(Value::as_bool)
            != Some(true)
}

pub(crate) fn event_error(payload: &Value) -> Option<&Value> {
    payload
        .get("error")
        .or_else(|| payload.pointer("/response/error"))
}

pub(crate) fn classify_event_failure(payload: &Value) -> Option<CodexEventFailure> {
    let event_type = payload.get("type").and_then(Value::as_str)?;
    if event_type == "codex.rate_limits" {
        if !is_terminal_rate_limit_event(payload) {
            return None;
        }
        return Some(CodexEventFailure {
            kind: CodexFailureKind::RateLimit,
            explicit_status: Some(429),
            status: 429,
            message: "rate limit reached".to_string(),
            retry_after: scalar_string(payload.pointer("/rate_limits/primary/reset_after_seconds")),
        });
    }
    if !matches!(event_type, "response.failed" | "response.error" | "error") {
        return None;
    }

    let error = event_error(payload);
    let explicit_status = numeric_status(payload)
        .or_else(|| {
            error
                .and_then(|value| value.get("status"))
                .and_then(Value::as_u64)
        })
        .and_then(|status| u16::try_from(status).ok());
    let message = error
        .and_then(|value| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("Upstream error")
        .to_string();
    let code = error
        .and_then(|value| value.get("code"))
        .and_then(Value::as_str);
    let error_type = error
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str);
    let lower = message.to_ascii_lowercase();

    let kind = if explicit_status == Some(429) || lower.contains("rate limit") {
        CodexFailureKind::RateLimit
    } else if explicit_status == Some(529)
        || code == Some("overloaded_error")
        || error_type == Some("overloaded_error")
        || lower.contains("overloaded")
    {
        CodexFailureKind::Overloaded
    } else if explicit_status.is_some_and(|status| matches!(status, 500 | 502 | 503 | 504))
        || matches!(
            code,
            Some("server_error" | "internal_server_error" | "internal_error")
        )
        || matches!(
            error_type,
            Some("server_error" | "internal_server_error" | "internal_error")
        )
        || retryable_message(&lower)
    {
        CodexFailureKind::Transient
    } else {
        CodexFailureKind::Permanent
    };
    let status = explicit_status.unwrap_or(match kind {
        CodexFailureKind::RateLimit => 429,
        CodexFailureKind::Overloaded => 529,
        CodexFailureKind::Transient => 503,
        CodexFailureKind::Permanent => 500,
    });
    let retry_after = error
        .and_then(|value| value.get("retry_after"))
        .and_then(scalar_string_value)
        .or_else(|| {
            error
                .and_then(|value| value.get("retry_after_seconds"))
                .and_then(scalar_string_value)
        })
        .or_else(|| scalar_string(payload.get("retry_after_seconds")))
        .or_else(|| scalar_string(payload.pointer("/headers/retry-after")))
        .or_else(|| scalar_string(payload.pointer("/headers/Retry-After")));

    Some(CodexEventFailure {
        kind,
        explicit_status,
        status,
        message,
        retry_after,
    })
}

pub(crate) fn failure_with_status(
    payload: &Value,
    expected_status: u16,
) -> Option<CodexEventFailure> {
    classify_event_failure(payload).filter(|failure| failure.status == expected_status)
}

pub(crate) fn first_failure_with_status(
    body: &[u8],
    expected_status: u16,
) -> Option<CodexEventFailure> {
    for event in crate::anthropic::sse::parse_sse_events(body) {
        if event.data == "[DONE]" {
            continue;
        }
        let Ok(payload) = serde_json::from_str::<Value>(&event.data) else {
            continue;
        };
        if let Some(failure) = failure_with_status(&payload, expected_status) {
            return Some(failure);
        }
    }
    None
}

pub(crate) struct BoundedAuthFailureDetector {
    format: AuthFailureFormat,
    pending: Vec<u8>,
    discarding_oversized_sse_frame: bool,
    finished: bool,
    detected: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthFailureFormat {
    Json,
    Sse,
}

impl BoundedAuthFailureDetector {
    pub(crate) fn json() -> Self {
        Self::new(AuthFailureFormat::Json)
    }

    pub(crate) fn sse() -> Self {
        Self::new(AuthFailureFormat::Sse)
    }

    fn new(format: AuthFailureFormat) -> Self {
        Self {
            format,
            pending: Vec::new(),
            discarding_oversized_sse_frame: false,
            finished: false,
            detected: false,
        }
    }

    pub(crate) fn observe(&mut self, chunk: &[u8]) -> bool {
        if self.finished || self.detected {
            return false;
        }
        match self.format {
            AuthFailureFormat::Json => self.observe_json(chunk),
            AuthFailureFormat::Sse => self.observe_sse(chunk),
        }
    }

    pub(crate) fn finish(&mut self) -> bool {
        if self.finished || self.detected {
            return false;
        }
        self.finished = true;
        let detected = match self.format {
            AuthFailureFormat::Json => json_has_auth_failure(&self.pending),
            AuthFailureFormat::Sse if !self.discarding_oversized_sse_frame => {
                sse_has_auth_failure(&self.pending)
            }
            AuthFailureFormat::Sse => false,
        };
        self.detected = detected;
        detected
    }

    fn observe_json(&mut self, chunk: &[u8]) -> bool {
        let limit = crate::traffic::MAX_STREAM_CAPTURE_FRAME_BYTES;
        if self.pending.len().saturating_add(chunk.len()) > limit {
            self.pending.clear();
            self.finished = true;
            return false;
        }
        self.pending.extend_from_slice(chunk);
        if json_has_auth_failure(&self.pending) {
            self.detected = true;
            return true;
        }
        false
    }

    fn observe_sse(&mut self, chunk: &[u8]) -> bool {
        for byte in chunk {
            self.pending.push(*byte);
            if let Some(separator_len) = boundary_suffix_len(&self.pending) {
                if !self.discarding_oversized_sse_frame && sse_has_auth_failure(&self.pending) {
                    self.detected = true;
                    return true;
                }
                self.pending.clear();
                self.discarding_oversized_sse_frame = false;
                debug_assert!(separator_len <= 4);
            } else if self.discarding_oversized_sse_frame {
                retain_sse_boundary_prefix(&mut self.pending);
            } else if self.pending.len() > crate::traffic::MAX_STREAM_CAPTURE_FRAME_BYTES {
                self.discarding_oversized_sse_frame = true;
                retain_sse_boundary_prefix(&mut self.pending);
            }
        }
        false
    }
}

fn json_has_auth_failure(bytes: &[u8]) -> bool {
    serde_json::from_slice::<Value>(bytes)
        .ok()
        .and_then(|payload| failure_with_status(&payload, 401))
        .is_some()
}

fn sse_has_auth_failure(bytes: &[u8]) -> bool {
    first_failure_with_status(bytes, 401).is_some()
}

fn boundary_suffix_len(bytes: &[u8]) -> Option<usize> {
    if bytes.ends_with(b"\r\n\r\n") {
        Some(4)
    } else if bytes.ends_with(b"\n\n") || bytes.ends_with(b"\r\r") {
        Some(2)
    } else {
        None
    }
}

fn retain_sse_boundary_prefix(bytes: &mut Vec<u8>) {
    let keep = bytes.len().min(3);
    if bytes.len() > keep {
        bytes.drain(..bytes.len() - keep);
    }
}

pub(crate) fn first_retryable_failure(body: &[u8]) -> Option<CodexEventFailure> {
    for event in crate::anthropic::sse::parse_sse_events(body) {
        if event.data == "[DONE]" {
            continue;
        }
        let Ok(payload) = serde_json::from_str::<Value>(&event.data) else {
            continue;
        };
        if let Some(failure) = classify_event_failure(&payload)
            && failure.retryable()
        {
            return Some(failure);
        }
    }
    None
}

pub(crate) fn numeric_status(payload: &Value) -> Option<u64> {
    payload
        .get("status")
        .and_then(Value::as_u64)
        .or_else(|| payload.get("status_code").and_then(Value::as_u64))
}

fn scalar_string(value: Option<&Value>) -> Option<String> {
    value.and_then(scalar_string_value)
}

fn scalar_string_value(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn retryable_message(message: &str) -> bool {
    [
        "server error",
        "internal server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "temporarily unavailable",
        "you can retry your request",
        "socket connection was closed unexpectedly",
        "connection closed unexpectedly",
        "operation timed out",
        "connection reset",
        "connection closed",
        "timed out",
        "timeout",
        "econnreset",
        "epipe",
        "etimedout",
        "und_err_socket",
        "fetch failed",
        "unexpected eof",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_retryable_failure_kinds() {
        let rate = classify_event_failure(&serde_json::json!({
            "type": "codex.rate_limits",
            "rate_limits": {"limit_reached": true, "primary": {"reset_after_seconds": 1.5}}
        }))
        .unwrap();
        assert_eq!(rate.kind, CodexFailureKind::RateLimit);
        assert_eq!(rate.retry_after.as_deref(), Some("1.5"));

        let overload = classify_event_failure(&serde_json::json!({
            "type": "response.failed",
            "response": {"error": {"type": "overloaded_error", "message": "busy"}}
        }))
        .unwrap();
        assert_eq!(overload.status, 529);
        assert!(overload.retryable());
    }

    #[test]
    fn terminal_rate_limit_honors_credits() {
        // No credits field at all: legacy payload stays terminal.
        assert!(is_terminal_rate_limit_event(&serde_json::json!({
            "type": "codex.rate_limits",
            "rate_limits": {"limit_reached": true}
        })));

        // Credits exhausted: terminal.
        assert!(is_terminal_rate_limit_event(&serde_json::json!({
            "type": "codex.rate_limits",
            "rate_limits": {"limit_reached": true},
            "credits": {"has_credits": false, "unlimited": false}
        })));

        // Usable credits remain: informational.
        assert!(!is_terminal_rate_limit_event(&serde_json::json!({
            "type": "codex.rate_limits",
            "rate_limits": {"limit_reached": true},
            "credits": {"has_credits": true, "unlimited": false}
        })));

        // Unlimited plan: informational.
        assert!(!is_terminal_rate_limit_event(&serde_json::json!({
            "type": "codex.rate_limits",
            "rate_limits": {"limit_reached": true},
            "credits": {"has_credits": false, "unlimited": true}
        })));

        // Limit not reached: never terminal, credits irrelevant.
        assert!(!is_terminal_rate_limit_event(&serde_json::json!({
            "type": "codex.rate_limits",
            "rate_limits": {"limit_reached": false},
            "credits": {"has_credits": false, "unlimited": false}
        })));

        // Wrong event type never matches.
        assert!(!is_terminal_rate_limit_event(&serde_json::json!({
            "type": "response.completed",
            "rate_limits": {"limit_reached": true}
        })));
    }

    #[test]
    fn classifier_skips_credited_rate_limit_snapshots() {
        assert!(
            classify_event_failure(&serde_json::json!({
                "type": "codex.rate_limits",
                "rate_limits": {"limit_reached": true},
                "credits": {"has_credits": true, "unlimited": false}
            }))
            .is_none()
        );
    }

    #[test]
    fn ignores_informational_and_permanent_events() {
        assert!(
            classify_event_failure(&serde_json::json!({
                "type": "codex.rate_limits",
                "rate_limits": {"limit_reached": false}
            }))
            .is_none()
        );
        let failure = classify_event_failure(&serde_json::json!({
            "type": "error",
            "error": {"status": 400, "message": "bad request"}
        }))
        .unwrap();
        assert!(!failure.retryable());
    }

    #[test]
    fn extracts_explicit_unauthorized_failures_from_payload_and_sse() {
        let payload = serde_json::json!({
            "type": "response.failed",
            "status_code": 401,
            "response": {
                "error": {
                    "status": 401,
                    "message": "route credential rejected",
                    "retry_after": 2
                }
            }
        });
        let failure = failure_with_status(&payload, 401).unwrap();
        assert_eq!(failure.explicit_status, Some(401));
        assert_eq!(failure.status, 401);
        assert_eq!(failure.message, "route credential rejected");
        assert_eq!(failure.retry_after.as_deref(), Some("2"));
        assert!(!failure.retryable());

        let body = format!(
            "data: {{\"type\":\"response.created\"}}\n\ndata: {payload}\n\ndata: [DONE]\n\n"
        );
        assert_eq!(
            first_failure_with_status(body.as_bytes(), 401),
            Some(failure)
        );
        assert!(first_failure_with_status(body.as_bytes(), 403).is_none());
    }

    #[test]
    fn unauthorized_requires_an_explicit_401_status() {
        assert!(
            failure_with_status(
                &serde_json::json!({
                    "type": "response.failed",
                    "response": {"error": {"message": "unauthorized"}}
                }),
                401
            )
            .is_none()
        );
    }

    #[test]
    fn bounded_auth_detector_handles_split_json_and_sse_frames() {
        let mut json = BoundedAuthFailureDetector::json();
        assert!(!json.observe(br#"{"type":"response.failed","status_"#));
        assert!(
            json.observe(br#"code":401,"response":{"error":{"status":401,"message":"expired"}}}"#)
        );
        assert!(!json.finish());

        let mut sse = BoundedAuthFailureDetector::sse();
        assert!(!sse.observe(
            b"event: response.failed\r\ndata: {\"type\":\"response.failed\",\"status_code\":4"
        ));
        assert!(!sse.observe(
            b"01,\"response\":{\"error\":{\"status\":401,\"message\":\"expired\"}}}\r\n\r"
        ));
        assert!(sse.observe(b"\n"));
        assert!(!sse.observe(b"data: duplicate\n\n"));
    }

    #[test]
    fn bounded_auth_detector_discards_oversized_frames_and_recovers() {
        let mut json = BoundedAuthFailureDetector::json();
        assert!(!json.observe(&vec![
            b'x';
            crate::traffic::MAX_STREAM_CAPTURE_FRAME_BYTES + 1
        ]));
        assert!(
            !json
                .observe(br#"{"type":"response.failed","status_code":401,"error":{"status":401}}"#)
        );

        let mut sse = BoundedAuthFailureDetector::sse();
        assert!(!sse.observe(&vec![
            b'x';
            crate::traffic::MAX_STREAM_CAPTURE_FRAME_BYTES + 1
        ]));
        assert!(!sse.observe(b"\n\n"));
        assert!(sse.observe(
            b"data: {\"type\":\"error\",\"status_code\":401,\"error\":{\"status\":401}}\n\n"
        ));
    }
}
