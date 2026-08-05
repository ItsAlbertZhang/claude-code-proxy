use crate::anthropic::schema::MessagesRequest;
use crate::monitor::MonitorHandle;
use crate::request_identity::{ConversationIdentity, RequestPurpose, RequestScope};
use crate::traffic::TrafficCapture;
use anyhow::Result;
use async_trait::async_trait;
use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use clap::Subcommand;
use std::sync::Arc;

#[derive(Debug, Clone, Subcommand)]
pub enum AuthCommand {
    /// Sign in using browser-based authentication
    Login,
    /// Sign in using a device code
    Device,
    /// Show the current authentication status
    Status,
    /// Delete stored authentication credentials
    Logout,
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &'static str;
    fn supported_models(&self) -> Vec<String>;
    fn cli(&self) -> &'static dyn CliHandlers;
    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response;

    async fn handle_messages_with_conversation_identity(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
        conversation_identity: Option<ConversationIdentity>,
    ) -> Response {
        let identity = compatible_explicit_identity(&ctx, conversation_identity);
        let _ = identity;
        self.handle_messages(body, ctx).await
    }

    async fn handle_messages_with_claude_fast_intent(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
        conversation_identity: Option<ConversationIdentity>,
        claude_fast_intent: bool,
    ) -> Response {
        let _ = claude_fast_intent;
        self.handle_messages_with_conversation_identity(body, ctx, conversation_identity)
            .await
    }

    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response;

    async fn generate_anthropic_stream(
        &self,
        _body: MessagesRequest,
        _ctx: RequestContext,
    ) -> Result<Generation, ProviderError> {
        Err(ProviderError::new(
            StatusCode::NOT_IMPLEMENTED,
            ProviderErrorKind::InvalidRequest,
            format!(
                "provider '{}' does not support OpenAI-compatible generation",
                self.name()
            ),
        ))
    }

    async fn generate_anthropic_stream_with_conversation_identity(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
        conversation_identity: Option<ConversationIdentity>,
    ) -> Result<Generation, ProviderError> {
        let identity = compatible_explicit_identity(&ctx, conversation_identity);
        let _ = identity;
        self.generate_anthropic_stream(body, ctx).await
    }
}

pub enum GenerationBody {
    BufferedSse(Bytes),
    LiveSse(Body),
}

pub struct Generation {
    pub body: GenerationBody,
    pub resolved_model: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorKind {
    Authentication,
    Permission,
    RateLimit,
    InvalidRequest,
    Api,
}

#[derive(Debug, Clone)]
pub struct ProviderError {
    pub status: StatusCode,
    pub kind: ProviderErrorKind,
    pub message: String,
    pub retry_after: Option<String>,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ProviderError {
    pub fn new(status: StatusCode, kind: ProviderErrorKind, message: impl Into<String>) -> Self {
        Self {
            status,
            kind,
            message: message.into(),
            retry_after: None,
            param: None,
            code: None,
        }
    }

    pub fn error_type(&self) -> &'static str {
        match self.kind {
            ProviderErrorKind::Authentication => "authentication_error",
            ProviderErrorKind::Permission => "permission_error",
            ProviderErrorKind::RateLimit => "rate_limit_error",
            ProviderErrorKind::InvalidRequest => "invalid_request_error",
            ProviderErrorKind::Api => "api_error",
        }
    }
}

pub trait CliHandlers: Send + Sync {
    fn login(&self) -> Result<()>;
    fn device(&self) -> Result<()>;
    fn status(&self) -> Result<()>;
    fn logout(&self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct RequestContext {
    pub req_id: String,
    pub session_id: Option<String>,
    pub session_seq: Option<u64>,
    pub provider: String,
    pub traffic: Option<Arc<TrafficCapture>>,
    pub monitor: Option<MonitorHandle>,
}

#[derive(Debug, Clone)]
pub(crate) struct ScopedRequestContext {
    legacy: RequestContext,
    scope: RequestScope,
}

impl ScopedRequestContext {
    pub(crate) fn new(legacy: RequestContext, scope: RequestScope) -> Self {
        Self { legacy, scope }
    }

    pub(crate) fn into_parts(self) -> (RequestContext, RequestScope) {
        (self.legacy, self.scope)
    }
}

pub(crate) fn compatible_explicit_identity(
    ctx: &RequestContext,
    identity: Option<ConversationIdentity>,
) -> Option<ConversationIdentity> {
    let identity = identity.and_then(|identity| identity.validated())?;
    let legacy = ctx
        .session_id
        .as_deref()
        .and_then(ConversationIdentity::from_legacy_main)?;
    (identity.session_component() == legacy.session_component()).then_some(identity)
}

pub(crate) fn legacy_scope(ctx: &RequestContext, purpose: RequestPurpose) -> RequestScope {
    RequestScope::legacy(ctx.session_id.as_deref(), purpose)
}
