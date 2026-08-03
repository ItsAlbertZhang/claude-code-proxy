use base64::Engine;
use http::HeaderMap;
use sha2::{Digest, Sha256};

pub const CLAUDE_SESSION_HEADER: &str = "x-claude-code-session-id";
pub const CLAUDE_AGENT_HEADER: &str = "x-claude-code-agent-id";
pub const CLAUDE_PARENT_AGENT_HEADER: &str = "x-claude-code-parent-agent-id";

const OPENAI_SESSION_HEADER: &str = "session_id";
const OPENAI_REQUEST_HEADER: &str = "x-client-request-id";
const MAX_IDENTITY_LEN: usize = 512;
const OPAQUE_LANE_VERSION: &[u8] = b"ccp-opaque-lane-v1";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConversationIdentity {
    Main(String),
    Agent(String, String),
}

impl ConversationIdentity {
    pub fn from_headers(headers: &HeaderMap) -> Option<Self> {
        RequestScope::from_headers(headers, RequestPurpose::Conversation)
            .identity
            .clone()
    }

    pub(crate) fn validated(&self) -> Option<Self> {
        match self {
            Self::Main(session) if valid_identity_text(session) => {
                Some(Self::Main(trim_ows(session).to_string()))
            }
            Self::Agent(session, agent)
                if valid_identity_text(session) && valid_identity_text(agent) =>
            {
                Some(Self::Agent(
                    trim_ows(session).to_string(),
                    trim_ows(agent).to_string(),
                ))
            }
            Self::Main(_) | Self::Agent(_, _) => None,
        }
    }

    pub(crate) fn from_legacy_main(value: &str) -> Option<Self> {
        valid_identity_text(value).then(|| Self::Main(trim_ows(value).to_string()))
    }

    pub(crate) fn session_component(&self) -> &str {
        match self {
            Self::Main(session) | Self::Agent(session, _) => session,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPurpose {
    Conversation,
    CountTokens,
    AutoReview,
    Auxiliary,
}

impl RequestPurpose {
    pub fn is_conversational(self) -> bool {
        matches!(self, Self::Conversation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestScope {
    identity: Option<ConversationIdentity>,
    parent_agent_id: Option<String>,
    purpose: RequestPurpose,
    claude_identity_headers_present: bool,
}

impl RequestScope {
    /// Parses the complete Claude identity tuple exactly once.
    pub fn from_headers(headers: &HeaderMap, purpose: RequestPurpose) -> Self {
        let claude_identity_headers_present = [
            CLAUDE_SESSION_HEADER,
            CLAUDE_AGENT_HEADER,
            CLAUDE_PARENT_AGENT_HEADER,
        ]
        .into_iter()
        .any(|name| headers.get_all(name).iter().next().is_some());
        let session = read_identity_header(headers, CLAUDE_SESSION_HEADER);
        let agent = read_identity_header(headers, CLAUDE_AGENT_HEADER);
        let parent = read_identity_header(headers, CLAUDE_PARENT_AGENT_HEADER);

        let tuple_valid = !session.is_invalid() && !agent.is_invalid() && !parent.is_invalid();
        let identity = tuple_valid
            .then(|| match (session.value(), agent.value(), parent.value()) {
                (Some(session_id), Some(agent_id), _) => Some(ConversationIdentity::Agent(
                    session_id.to_string(),
                    agent_id.to_string(),
                )),
                (Some(session_id), None, None) => {
                    Some(ConversationIdentity::Main(session_id.to_string()))
                }
                _ => None,
            })
            .flatten();
        let parent_agent_id = tuple_valid
            .then(|| parent.value().map(str::to_string))
            .flatten();

        Self {
            identity,
            parent_agent_id,
            purpose,
            claude_identity_headers_present,
        }
    }

    /// Parses native OpenAI identity without allowing malformed Claude headers
    /// to fall through to a legacy identity.
    pub fn from_openai_headers(headers: &HeaderMap, purpose: RequestPurpose) -> Self {
        let claude = Self::from_headers(headers, purpose);
        if claude.claude_identity_headers_present {
            return claude;
        }

        let session = read_identity_header(headers, OPENAI_SESSION_HEADER);
        let request = read_identity_header(headers, OPENAI_REQUEST_HEADER);
        let identity = match session {
            ParsedHeader::Valid(value) => Some(ConversationIdentity::Main(value.to_string())),
            ParsedHeader::Invalid => None,
            ParsedHeader::Missing => match request {
                ParsedHeader::Valid(value) => Some(ConversationIdentity::Main(value.to_string())),
                ParsedHeader::Missing | ParsedHeader::Invalid => None,
            },
        };
        Self {
            identity,
            parent_agent_id: None,
            purpose,
            claude_identity_headers_present: false,
        }
    }

    pub fn legacy(session_id: Option<&str>, purpose: RequestPurpose) -> Self {
        Self {
            identity: session_id.and_then(ConversationIdentity::from_legacy_main),
            parent_agent_id: None,
            purpose,
            claude_identity_headers_present: false,
        }
    }

    pub fn from_conversation_identity(
        identity: Option<ConversationIdentity>,
        purpose: RequestPurpose,
    ) -> Self {
        Self {
            identity: identity.and_then(|identity| identity.validated()),
            parent_agent_id: None,
            purpose,
            claude_identity_headers_present: false,
        }
    }

    pub fn identity(&self) -> Option<&ConversationIdentity> {
        self.identity.as_ref()
    }

    pub fn conversational_lane(&self) -> Option<&ConversationIdentity> {
        self.purpose
            .is_conversational()
            .then_some(self.identity.as_ref())
            .flatten()
    }

    pub fn purpose(&self) -> RequestPurpose {
        self.purpose
    }

    pub(crate) fn with_purpose(mut self, purpose: RequestPurpose) -> Self {
        self.purpose = purpose;
        self
    }

    pub fn parent_agent_id(&self) -> Option<&str> {
        self.parent_agent_id.as_deref()
    }

    pub fn claude_identity_headers_present(&self) -> bool {
        self.claude_identity_headers_present
    }

    pub(crate) fn provider_lane(&self, domain: LaneDomain) -> Option<OpaqueLane> {
        self.conversational_lane()
            .map(|identity| OpaqueLane::derive(domain, identity))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum LaneDomain {
    CodexConversation,
}

impl LaneDomain {
    fn label(self) -> &'static [u8] {
        match self {
            Self::CodexConversation => b"codex-conversation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OpaqueLane([u8; 32]);

impl OpaqueLane {
    fn derive(domain: LaneDomain, identity: &ConversationIdentity) -> Self {
        let mut digest = Sha256::new();
        update_length_prefixed(&mut digest, OPAQUE_LANE_VERSION);
        update_length_prefixed(&mut digest, domain.label());
        match identity {
            ConversationIdentity::Main(session) => {
                update_length_prefixed(&mut digest, b"main");
                update_length_prefixed(&mut digest, session.as_bytes());
            }
            ConversationIdentity::Agent(session, agent) => {
                update_length_prefixed(&mut digest, b"agent");
                update_length_prefixed(&mut digest, session.as_bytes());
                update_length_prefixed(&mut digest, agent.as_bytes());
            }
        }
        Self(digest.finalize().into())
    }

    pub(crate) fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub(crate) fn encode(&self) -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(self.0)
    }

    #[cfg(test)]
    pub(crate) fn decode(value: &str) -> Option<Self> {
        let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(value)
            .ok()?;
        Some(Self(decoded.try_into().ok()?))
    }
}

fn update_length_prefixed(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value);
}

#[derive(Debug)]
enum ParsedHeader<'a> {
    Missing,
    Valid(&'a str),
    Invalid,
}

impl ParsedHeader<'_> {
    fn value(&self) -> Option<&str> {
        match self {
            Self::Valid(value) => Some(value),
            Self::Missing | Self::Invalid => None,
        }
    }

    fn is_invalid(&self) -> bool {
        matches!(self, Self::Invalid)
    }
}

fn read_identity_header<'a>(headers: &'a HeaderMap, name: &str) -> ParsedHeader<'a> {
    let mut values = headers.get_all(name).iter();
    let Some(value) = values.next() else {
        return ParsedHeader::Missing;
    };
    if values.next().is_some() {
        return ParsedHeader::Invalid;
    }

    let Ok(value) = value.to_str() else {
        return ParsedHeader::Invalid;
    };
    let value = trim_ows(value);
    if !valid_trimmed_identity_text(value) {
        return ParsedHeader::Invalid;
    }

    ParsedHeader::Valid(value)
}

fn trim_ows(value: &str) -> &str {
    value.trim_matches(|character| matches!(character, ' ' | '\t'))
}

fn valid_identity_text(value: &str) -> bool {
    valid_trimmed_identity_text(trim_ows(value))
}

fn valid_trimmed_identity_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDENTITY_LEN
        && !value.contains(',')
        && value.bytes().all(|byte| (0x21..=0x7e).contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::{HeaderName, HeaderValue};

    fn headers(values: &[(&str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in values {
            headers.append(
                HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        headers
    }

    #[test]
    fn main_and_agent_lanes_are_distinct() {
        let main = RequestScope::from_headers(
            &headers(&[(CLAUDE_SESSION_HEADER, "session-a")]),
            RequestPurpose::Conversation,
        );
        let agent = RequestScope::from_headers(
            &headers(&[
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "agent-a"),
            ]),
            RequestPurpose::Conversation,
        );
        let sibling = RequestScope::from_headers(
            &headers(&[
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "agent-b"),
            ]),
            RequestPurpose::Conversation,
        );
        assert_ne!(main.conversational_lane(), agent.conversational_lane());
        assert_ne!(agent.conversational_lane(), sibling.conversational_lane());
    }

    #[test]
    fn nested_agent_uses_child_as_lane_and_parent_as_lineage() {
        let scope = RequestScope::from_headers(
            &headers(&[
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "agent-child"),
                (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
            ]),
            RequestPurpose::Conversation,
        );
        assert_eq!(
            scope.identity(),
            Some(&ConversationIdentity::Agent(
                "session-a".to_string(),
                "agent-child".to_string()
            ))
        );
        assert_eq!(scope.parent_agent_id(), Some("agent-parent"));
    }

    #[test]
    fn empty_identity_is_stateless() {
        let scope = RequestScope::from_headers(&HeaderMap::new(), RequestPurpose::Conversation);
        assert!(scope.conversational_lane().is_none());
        assert!(!scope.claude_identity_headers_present());
    }

    #[test]
    fn ambiguous_identity_tuples_are_stateless() {
        for values in [
            vec![(CLAUDE_AGENT_HEADER, "agent-a")],
            vec![(CLAUDE_PARENT_AGENT_HEADER, "agent-parent")],
            vec![
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
            ],
            vec![
                (CLAUDE_AGENT_HEADER, "agent-a"),
                (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
            ],
        ] {
            assert!(
                RequestScope::from_headers(&headers(&values), RequestPurpose::Conversation)
                    .conversational_lane()
                    .is_none()
            );
        }
    }

    #[test]
    fn malformed_identity_headers_are_stateless() {
        let malformed = ["", "   ", "two values", "two\tvalues"];
        // http::HeaderValue rejects DEL before ingress; keep the validator
        // characterization explicit for legacy string adapters.
        assert!(!valid_identity_text("x\u{7f}"));
        for field in [
            CLAUDE_SESSION_HEADER,
            CLAUDE_AGENT_HEADER,
            CLAUDE_PARENT_AGENT_HEADER,
        ] {
            for value in malformed {
                let mut values = vec![
                    (CLAUDE_SESSION_HEADER, "session-a"),
                    (CLAUDE_AGENT_HEADER, "agent-a"),
                    (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
                ];
                values
                    .iter_mut()
                    .find(|(name, _)| *name == field)
                    .unwrap()
                    .1 = value;
                assert!(
                    RequestScope::from_headers(&headers(&values), RequestPurpose::Conversation)
                        .conversational_lane()
                        .is_none(),
                    "field={field} value={value:?}"
                );
            }
        }
    }

    #[test]
    fn comma_in_identity_header_is_rejected() {
        for field in [
            CLAUDE_SESSION_HEADER,
            CLAUDE_AGENT_HEADER,
            CLAUDE_PARENT_AGENT_HEADER,
        ] {
            let mut values = vec![
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "agent-a"),
                (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
            ];
            values
                .iter_mut()
                .find(|(name, _)| *name == field)
                .unwrap()
                .1 = "a,b";
            assert!(
                RequestScope::from_headers(&headers(&values), RequestPurpose::Conversation)
                    .conversational_lane()
                    .is_none()
            );
        }
    }

    #[test]
    fn duplicate_identity_headers_fall_back_to_stateless() {
        for field in [
            CLAUDE_SESSION_HEADER,
            CLAUDE_AGENT_HEADER,
            CLAUDE_PARENT_AGENT_HEADER,
        ] {
            let mut values = headers(&[
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "agent-a"),
                (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
            ]);
            values.append(field, HeaderValue::from_static("duplicate"));
            assert!(
                RequestScope::from_headers(&values, RequestPurpose::Conversation)
                    .conversational_lane()
                    .is_none()
            );
        }
    }

    #[test]
    fn opaque_tokens_are_stable_domain_separated_and_non_revealing() {
        let scope = RequestScope::from_headers(
            &headers(&[
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "agent-a"),
            ]),
            RequestPurpose::Conversation,
        );
        let sibling = RequestScope::from_headers(
            &headers(&[
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "agent-b"),
            ]),
            RequestPurpose::Conversation,
        );
        let lane = scope.provider_lane(LaneDomain::CodexConversation).unwrap();
        assert_eq!(
            lane,
            scope.provider_lane(LaneDomain::CodexConversation).unwrap()
        );
        assert_ne!(
            lane,
            sibling
                .provider_lane(LaneDomain::CodexConversation)
                .unwrap()
        );
        assert!(!lane.encode().contains("session-a"));
        assert!(!lane.encode().contains("agent-a"));
    }

    #[test]
    fn explicit_identity_is_canonicalized_before_lane_derivation() {
        let canonical = RequestScope::from_conversation_identity(
            Some(ConversationIdentity::Agent(
                "session-a".to_string(),
                "agent-a".to_string(),
            )),
            RequestPurpose::Conversation,
        );
        let padded = RequestScope::from_conversation_identity(
            Some(ConversationIdentity::Agent(
                " \tsession-a\t ".to_string(),
                "\tagent-a ".to_string(),
            )),
            RequestPurpose::Conversation,
        );
        assert_eq!(canonical.identity(), padded.identity());
        assert_eq!(
            canonical.provider_lane(LaneDomain::CodexConversation),
            padded.provider_lane(LaneDomain::CodexConversation)
        );
    }

    #[test]
    fn opaque_lane_encoding_round_trips_exactly() {
        let scope = RequestScope::legacy(Some("session-a"), RequestPurpose::Conversation);
        let lane = scope.provider_lane(LaneDomain::CodexConversation).unwrap();
        assert_eq!(OpaqueLane::decode(&lane.encode()), Some(lane));
        assert!(OpaqueLane::decode("not-an-opaque-lane").is_none());
    }

    #[test]
    fn auxiliary_scope_has_no_conversational_lane() {
        for purpose in [
            RequestPurpose::CountTokens,
            RequestPurpose::AutoReview,
            RequestPurpose::Auxiliary,
        ] {
            let scope = RequestScope::from_headers(
                &headers(&[(CLAUDE_SESSION_HEADER, "session-a")]),
                purpose,
            );
            assert!(scope.identity().is_some());
            assert!(scope.conversational_lane().is_none());
            assert!(scope.provider_lane(LaneDomain::CodexConversation).is_none());
        }
    }

    #[test]
    fn malformed_agent_never_downgrades_to_main_and_fallback_is_suppressed() {
        let scope = RequestScope::from_openai_headers(
            &headers(&[
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "malformed agent"),
                (OPENAI_SESSION_HEADER, "legacy-session"),
            ]),
            RequestPurpose::Conversation,
        );
        assert!(scope.identity().is_none());
        assert!(scope.claude_identity_headers_present());
    }

    #[test]
    fn openai_fallback_uses_the_same_strict_validation() {
        let valid = RequestScope::from_openai_headers(
            &headers(&[(OPENAI_SESSION_HEADER, " \tlegacy-session\t ")]),
            RequestPurpose::Conversation,
        );
        assert_eq!(
            valid.identity(),
            Some(&ConversationIdentity::Main("legacy-session".to_string()))
        );
        let invalid = RequestScope::from_openai_headers(
            &headers(&[
                (OPENAI_SESSION_HEADER, "bad session"),
                (OPENAI_REQUEST_HEADER, "valid-fallback"),
            ]),
            RequestPurpose::Conversation,
        );
        assert!(invalid.identity().is_none());
    }

    #[test]
    fn nontext_and_oversized_fields_invalidate_the_tuple() {
        for field in [
            CLAUDE_SESSION_HEADER,
            CLAUDE_AGENT_HEADER,
            CLAUDE_PARENT_AGENT_HEADER,
        ] {
            let mut nontext = headers(&[
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "agent-a"),
                (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
            ]);
            nontext.insert(field, HeaderValue::from_bytes(&[0x80]).unwrap());
            assert!(
                RequestScope::from_headers(&nontext, RequestPurpose::Conversation)
                    .identity()
                    .is_none()
            );

            let oversized = "x".repeat(MAX_IDENTITY_LEN + 1);
            let mut values = vec![
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "agent-a"),
                (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
            ];
            values
                .iter_mut()
                .find(|(name, _)| *name == field)
                .unwrap()
                .1 = &oversized;
            assert!(
                RequestScope::from_headers(&headers(&values), RequestPurpose::Conversation)
                    .identity()
                    .is_none()
            );
        }
    }
}
