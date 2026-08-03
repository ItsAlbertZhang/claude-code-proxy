use std::{collections::VecDeque, io, pin::Pin, time::Duration};

use axum::{body::Body, response::Response};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use http::{HeaderMap, StatusCode};
use serde_json::{Value, json};

use crate::providers::codex::client::InBandAuthRefreshDetector;
use crate::providers::codex::native::NativeResponseOutcome;
use crate::{provider::RequestContext, traffic::MAX_SSE_CAPTURE_BYTES};

use super::{
    ChatError, MAX_CHAT_OUTPUT_BYTES, MAX_CHAT_SSE_FRAME_BYTES, MAX_CHAT_UPSTREAM_BYTES,
    response::CompletionState,
};

type UpstreamStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

pub fn streaming_response(
    upstream: reqwest::Response,
    ctx: RequestContext,
    model: String,
    include_usage: bool,
    body_idle_timeout_ms: u64,
) -> Response {
    streaming_response_inner(
        upstream,
        ctx,
        model,
        include_usage,
        body_idle_timeout_ms,
        None,
    )
}

pub(crate) fn streaming_response_with_auth_refresh(
    upstream: reqwest::Response,
    ctx: RequestContext,
    model: String,
    include_usage: bool,
    body_idle_timeout_ms: u64,
    auth_refresh: InBandAuthRefreshDetector,
) -> Response {
    streaming_response_inner(
        upstream,
        ctx,
        model,
        include_usage,
        body_idle_timeout_ms,
        Some(auth_refresh),
    )
}

fn streaming_response_inner(
    upstream: reqwest::Response,
    ctx: RequestContext,
    model: String,
    include_usage: bool,
    body_idle_timeout_ms: u64,
    auth_refresh: Option<InBandAuthRefreshDetector>,
) -> Response {
    let outcome = NativeResponseOutcome::default();
    let state = StreamState {
        upstream: Box::pin(upstream.bytes_stream()),
        decoder: ChatSseDecoder::default(),
        output: VecDeque::new(),
        completion: CompletionState::new_streaming(&model),
        role_sent: false,
        include_usage,
        ended: false,
        generation_started: false,
        ctx,
        outcome: outcome.clone(),
        auth_refresh,
        body_idle_timeout_ms,
        translated_output_bytes: 0,
        raw: Vec::new(),
        raw_truncated: 0,
    };
    let stream =
        futures_util::stream::unfold(Some(state), |state| async move { next_frame(state).await });
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        http::header::CACHE_CONTROL,
        http::HeaderValue::from_static("no-cache"),
    );
    response.extensions_mut().insert(outcome);
    response
}

async fn next_frame(
    state: Option<StreamState>,
) -> Option<(Result<Bytes, io::Error>, Option<StreamState>)> {
    let mut state = state?;
    loop {
        if let Some(frame) = state.output.pop_front() {
            let next = if state.ended && state.output.is_empty() {
                state.finish_capture("complete");
                None
            } else {
                Some(state)
            };
            return Some((Ok(frame), next));
        }
        if state.ended {
            state.finish_capture("complete");
            return None;
        }
        match tokio::time::timeout(
            Duration::from_millis(state.body_idle_timeout_ms),
            state.upstream.next(),
        )
        .await
        {
            Ok(Some(Ok(chunk))) => state.observe_chunk(&chunk),
            Ok(Some(Err(error))) => {
                state.fail(ChatError::upstream(format!(
                    "Codex response body read failed: {error}"
                )));
            }
            Ok(None) => {
                if let Some(auth_refresh) = state.auth_refresh.as_mut() {
                    auth_refresh.finish();
                }
                if let Err(error) = state.decoder.finish() {
                    state.fail(error);
                } else if !state.completion.completed {
                    state.fail(ChatError::upstream(
                        "Codex event stream ended before completion",
                    ));
                } else {
                    state.finish_success();
                }
            }
            Err(_) => {
                state.fail(ChatError::timeout(format!(
                    "Timed out waiting {}ms for the next Codex response body chunk",
                    state.body_idle_timeout_ms
                )));
            }
        }
    }
}

struct StreamState {
    upstream: UpstreamStream,
    decoder: ChatSseDecoder,
    output: VecDeque<Bytes>,
    completion: CompletionState,
    role_sent: bool,
    include_usage: bool,
    ended: bool,
    generation_started: bool,
    ctx: RequestContext,
    outcome: NativeResponseOutcome,
    auth_refresh: Option<InBandAuthRefreshDetector>,
    body_idle_timeout_ms: u64,
    translated_output_bytes: usize,
    raw: Vec<u8>,
    raw_truncated: u64,
}

impl StreamState {
    fn observe_chunk(&mut self, chunk: &[u8]) {
        if let Some(auth_refresh) = self.auth_refresh.as_mut() {
            auth_refresh.observe(chunk);
        }
        if !chunk.is_empty() && !self.generation_started {
            if let Some(monitor) = self.ctx.monitor.as_ref() {
                monitor.generation_started(&self.ctx.req_id);
            }
            self.generation_started = true;
        }
        let remaining = MAX_SSE_CAPTURE_BYTES.saturating_sub(self.raw.len());
        let captured = remaining.min(chunk.len());
        self.raw.extend_from_slice(&chunk[..captured]);
        self.raw_truncated = self
            .raw_truncated
            .saturating_add((chunk.len() - captured) as u64);
        let events = match self.decoder.observe(chunk) {
            Ok(events) => events,
            Err(error) => {
                self.fail(error);
                Vec::new()
            }
        };
        let event_count = events.len();
        for data in events {
            self.observe_event(&data);
            if self.ended {
                break;
            }
        }
        if let Some(monitor) = self.ctx.monitor.as_ref() {
            monitor.stream_progress(
                &self.ctx.req_id,
                chunk.len() as u64,
                event_count as u64,
                Some(self.completion.usage.prompt_tokens),
                Some(self.completion.usage.completion_tokens),
            );
        }
    }

    fn observe_event(&mut self, data: &str) {
        if data == "[DONE]" {
            return;
        }
        let event: Value = match serde_json::from_str(data) {
            Ok(event) => event,
            Err(_) => {
                self.fail(ChatError::upstream(
                    "Codex returned malformed JSON in its event stream",
                ));
                return;
            }
        };
        if let Some(traffic) = self.ctx.traffic.as_deref() {
            traffic.write_json_event("040-upstream-event", &event);
        }
        match self.completion.observe(&event) {
            Ok(Some(delta)) => {
                if !self.role_sent {
                    self.role_sent = true;
                    let role = sse(json!({
                        "id": self.completion.id,
                        "object": "chat.completion.chunk",
                        "created": self.completion.created,
                        "model": self.completion.model,
                        "choices": [{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],
                    }));
                    if !self.push_output(role) {
                        self.fail(ChatError::upstream(
                            "Codex translated output exceeded the configured limit",
                        ));
                        return;
                    }
                }
                let content = sse(json!({
                    "id": self.completion.id,
                    "object": "chat.completion.chunk",
                    "created": self.completion.created,
                    "model": self.completion.model,
                    "choices": [{"index":0,"delta":{"content":delta},"finish_reason":null}],
                }));
                if !self.push_output(content) {
                    self.fail(ChatError::upstream(
                        "Codex translated output exceeded the configured limit",
                    ));
                }
            }
            Ok(None) if self.completion.completed => self.finish_success(),
            Ok(None) => {}
            Err(error) => self.fail(error),
        }
    }

    fn finish_success(&mut self) {
        if self.ended {
            return;
        }
        if !self.completion.has_output_text() {
            self.fail(ChatError::upstream("Codex completed without output text"));
            return;
        }
        let mut terminal = json!({
            "id": self.completion.id,
            "object": "chat.completion.chunk",
            "created": self.completion.created,
            "model": self.completion.model,
            "choices": [{"index":0,"delta":{},"finish_reason":self.completion.finish_reason}],
        });
        if self.include_usage {
            terminal["usage"] = self.completion.usage.value();
        }
        if let Some(monitor) = self.ctx.monitor.as_ref() {
            monitor.usage_updated(
                &self.ctx.req_id,
                Some(self.completion.usage.prompt_tokens),
                Some(self.completion.usage.completion_tokens),
            );
        }
        let done = Bytes::from_static(b"data: [DONE]\n\n");
        let terminal = sse(terminal);
        let additional = terminal.len().saturating_add(done.len());
        if self.translated_output_bytes.saturating_add(additional) > MAX_CHAT_OUTPUT_BYTES {
            self.fail(ChatError::upstream(
                "Codex translated output exceeded the configured limit",
            ));
            return;
        }
        self.translated_output_bytes += additional;
        self.output.push_back(terminal);
        self.output.push_back(done);
        self.ended = true;
    }

    fn push_output(&mut self, frame: Bytes) -> bool {
        if self.translated_output_bytes.saturating_add(frame.len()) > MAX_CHAT_OUTPUT_BYTES {
            return false;
        }
        self.translated_output_bytes += frame.len();
        self.output.push_back(frame);
        true
    }

    fn fail(&mut self, error: ChatError) {
        if self.ended {
            return;
        }
        self.outcome.fail(error.message.clone());
        self.output.push_back(sse(error.value()));
        self.output
            .push_back(Bytes::from_static(b"data: [DONE]\n\n"));
        self.ended = true;
    }

    fn finish_capture(&mut self, capture_outcome: &str) {
        let Some(traffic) = self.ctx.traffic.as_deref() else {
            return;
        };
        if !self.raw.is_empty() {
            traffic.write_bytes("032-upstream-response-body.sse", &self.raw);
        }
        traffic.write_json(
            "033-chat-completions-response-capture",
            &json!({
                "outcome": capture_outcome,
                "capturedBytes": self.raw.len(),
                "truncatedBytes": self.raw_truncated,
                "inputTokens": self.completion.usage.prompt_tokens,
                "outputTokens": self.completion.usage.completion_tokens,
            }),
        );
    }
}

impl Drop for StreamState {
    fn drop(&mut self) {
        if !self.ended {
            self.finish_capture("downstream_cancelled");
        }
    }
}

fn sse(value: Value) -> Bytes {
    Bytes::from(format!(
        "data: {}\n\n",
        serde_json::to_string(&value).unwrap()
    ))
}

#[derive(Default)]
pub(crate) struct ChatSseDecoder {
    pending: Vec<u8>,
    total_bytes: usize,
}

impl ChatSseDecoder {
    pub(crate) fn observe(&mut self, chunk: &[u8]) -> Result<Vec<String>, ChatError> {
        if self.total_bytes.saturating_add(chunk.len()) > MAX_CHAT_UPSTREAM_BYTES {
            return Err(ChatError::upstream(
                "Codex response body exceeded the configured limit",
            ));
        }
        self.total_bytes += chunk.len();
        self.pending.extend_from_slice(chunk);

        let mut events = Vec::new();
        while let Some((end, separator_len)) = find_boundary(&self.pending) {
            if end > MAX_CHAT_SSE_FRAME_BYTES {
                return Err(ChatError::upstream(
                    "Codex event-stream frame exceeded the configured limit",
                ));
            }
            let frame = self
                .pending
                .drain(..end + separator_len)
                .collect::<Vec<_>>();
            events.extend(
                crate::anthropic::sse::parse_sse_events(&frame)
                    .into_iter()
                    .map(|event| event.data),
            );
        }
        if self.pending.len() > MAX_CHAT_SSE_FRAME_BYTES {
            return Err(ChatError::upstream(
                "Codex event-stream frame exceeded the configured limit",
            ));
        }
        Ok(events)
    }

    pub(crate) fn finish(&self) -> Result<(), ChatError> {
        if self.pending.iter().any(|byte| !byte.is_ascii_whitespace()) {
            return Err(ChatError::upstream(
                "Codex event stream ended with an incomplete frame",
            ));
        }
        Ok(())
    }
}

fn find_boundary(bytes: &[u8]) -> Option<(usize, usize)> {
    for index in 0..bytes.len() {
        if bytes[index..].starts_with(b"\r\n\r\n") {
            return Some((index, 4));
        }
        if bytes[index..].starts_with(b"\n\n") || bytes[index..].starts_with(b"\r\r") {
            return Some((index, 2));
        }
    }
    None
}

pub fn response_headers(upstream: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for name in [
        "retry-after",
        "x-request-id",
        "openai-processing-ms",
        "openai-version",
    ] {
        if let Some(value) = upstream.get(name) {
            headers.insert(name, value.clone());
        }
    }
    for (name, value) in upstream {
        if name.as_str().starts_with("x-ratelimit-") {
            headers.append(name.clone(), value.clone());
        }
    }
    headers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_rejects_oversized_incomplete_frame() {
        let mut decoder = ChatSseDecoder::default();
        let error = decoder
            .observe(&vec![b'x'; MAX_CHAT_SSE_FRAME_BYTES + 1])
            .unwrap_err();
        assert!(error.message.contains("frame exceeded"));
    }

    #[test]
    fn decoder_rejects_total_bytes_across_many_frames() {
        let mut decoder = ChatSseDecoder {
            pending: Vec::new(),
            total_bytes: MAX_CHAT_UPSTREAM_BYTES,
        };
        let error = decoder.observe(b"x").unwrap_err();
        assert!(error.message.contains("body exceeded"));
    }

    #[test]
    fn boundary_supports_split_safe_delimiters() {
        assert_eq!(find_boundary(b"data: {}\n\nnext"), Some((8, 2)));
        assert_eq!(find_boundary(b"data: {}\r\n\r\nnext"), Some((8, 4)));
        assert_eq!(find_boundary(b"data: {}"), None);
    }
}
