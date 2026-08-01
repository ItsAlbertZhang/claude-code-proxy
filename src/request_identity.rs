use base64::Engine;
use http::HeaderMap;
use sha2::{Digest, Sha256};

pub const CLAUDE_SESSION_HEADER: &str = "x-claude-code-session-id";
pub const CLAUDE_AGENT_HEADER: &str = "x-claude-code-agent-id";
pub const CLAUDE_PARENT_AGENT_HEADER: &str = "x-claude-code-parent-agent-id";

const MAX_IDENTITY_LEN: usize = 512;
const OPAQUE_TOKEN_VERSION: &str = "ccp-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPurpose {
    Conversation,
    CountTokens,
    AutoReview,
}

impl RequestPurpose {
    pub fn is_conversational(self) -> bool {
        matches!(self, Self::Conversation)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentLane {
    Main,
    Agent(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AgentLaneKey {
    Main {
        session_id: String,
    },
    Agent {
        session_id: String,
        agent_id: String,
    },
}

impl AgentLaneKey {
    pub fn main(session_id: impl Into<String>) -> Self {
        Self::Main {
            session_id: session_id.into(),
        }
    }

    pub fn session_id(&self) -> &str {
        match self {
            Self::Main { session_id } | Self::Agent { session_id, .. } => session_id,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Main { .. } => "main",
            Self::Agent { .. } => "agent",
        }
    }

    pub fn opaque_token(&self, domain: &str) -> String {
        let mut digest = Sha256::new();
        digest.update(OPAQUE_TOKEN_VERSION.as_bytes());
        digest.update([0]);
        digest.update(domain.as_bytes());
        digest.update([0]);
        digest.update(b"claude-code");
        digest.update([0]);
        digest.update(self.session_id().as_bytes());
        digest.update([0]);
        match self {
            Self::Main { .. } => digest.update(b"main"),
            Self::Agent { agent_id, .. } => {
                digest.update(b"agent");
                digest.update([0]);
                digest.update(agent_id.as_bytes());
            }
        }
        format!(
            "{OPAQUE_TOKEN_VERSION}-{}",
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest.finalize())
        )
    }

    pub fn fingerprint(&self) -> String {
        self.opaque_token("observability")
            .chars()
            .take(19)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestIdentity {
    session_id: Option<String>,
    agent_id: Option<String>,
    parent_agent_id: Option<String>,
    lane: Option<AgentLaneKey>,
}

impl RequestIdentity {
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let session = read_identity_header(headers, CLAUDE_SESSION_HEADER);
        let agent = read_identity_header(headers, CLAUDE_AGENT_HEADER);
        let parent = read_identity_header(headers, CLAUDE_PARENT_AGENT_HEADER);

        let session_id = session.value().map(str::to_string);
        let agent_id = agent.value().map(str::to_string);
        let parent_agent_id = parent.value().map(str::to_string);
        let headers_valid = session.is_valid() && agent.is_valid() && parent.is_valid();
        let lane = if headers_valid {
            match (
                session_id.as_ref(),
                agent_id.as_ref(),
                parent_agent_id.as_ref(),
            ) {
                (Some(session_id), Some(agent_id), _) => Some(AgentLaneKey::Agent {
                    session_id: session_id.clone(),
                    agent_id: agent_id.clone(),
                }),
                (Some(session_id), None, None) => Some(AgentLaneKey::main(session_id.clone())),
                _ => None,
            }
        } else {
            None
        };

        Self {
            session_id,
            agent_id,
            parent_agent_id,
            lane,
        }
    }

    pub fn legacy_main(session_id: Option<&str>) -> Self {
        let session_id = session_id.map(str::to_string);
        let lane = session_id
            .as_ref()
            .map(|value| AgentLaneKey::main(value.clone()));
        Self {
            session_id,
            agent_id: None,
            parent_agent_id: None,
            lane,
        }
    }

    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    pub fn agent_id(&self) -> Option<&str> {
        self.agent_id.as_deref()
    }

    pub fn parent_agent_id(&self) -> Option<&str> {
        self.parent_agent_id.as_deref()
    }

    pub fn lane(&self) -> Option<&AgentLaneKey> {
        self.lane.as_ref()
    }

    pub fn is_stateful(&self) -> bool {
        self.lane.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestScope {
    pub identity: RequestIdentity,
    pub purpose: RequestPurpose,
}

impl RequestScope {
    pub fn from_headers(headers: &HeaderMap, purpose: RequestPurpose) -> Self {
        Self {
            identity: RequestIdentity::from_headers(headers),
            purpose,
        }
    }

    pub fn legacy(session_id: Option<&str>, purpose: RequestPurpose) -> Self {
        Self {
            identity: RequestIdentity::legacy_main(session_id),
            purpose,
        }
    }

    pub fn lane(&self) -> Option<&AgentLaneKey> {
        self.identity.lane()
    }

    pub fn conversational_lane(&self) -> Option<&AgentLaneKey> {
        self.purpose
            .is_conversational()
            .then(|| self.lane())
            .flatten()
    }

    pub fn lane_token(&self, domain: &str) -> Option<String> {
        self.conversational_lane()
            .map(|lane| lane.opaque_token(domain))
    }
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

    fn is_valid(&self) -> bool {
        !matches!(self, Self::Invalid)
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
    let value = value.trim();
    if value.is_empty()
        || value.len() > MAX_IDENTITY_LEN
        || value.contains(',')
        || !value.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return ParsedHeader::Invalid;
    }
    ParsedHeader::Valid(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;

    fn headers(values: &[(&str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in values {
            headers.insert(
                http::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        headers
    }

    #[test]
    fn main_and_agent_lanes_are_distinct() {
        let main = RequestIdentity::from_headers(&headers(&[(CLAUDE_SESSION_HEADER, "session-a")]));
        let agent = RequestIdentity::from_headers(&headers(&[
            (CLAUDE_SESSION_HEADER, "session-a"),
            (CLAUDE_AGENT_HEADER, "agent-a"),
        ]));
        let sibling = RequestIdentity::from_headers(&headers(&[
            (CLAUDE_SESSION_HEADER, "session-a"),
            (CLAUDE_AGENT_HEADER, "agent-b"),
        ]));

        assert!(matches!(
            main.lane(),
            Some(AgentLaneKey::Main { session_id }) if session_id == "session-a"
        ));
        assert_ne!(main.lane(), agent.lane());
        assert_ne!(agent.lane(), sibling.lane());
        assert_eq!(agent.agent_id(), Some("agent-a"));
    }

    #[test]
    fn nested_agent_uses_child_as_lane_and_parent_as_lineage() {
        let identity = RequestIdentity::from_headers(&headers(&[
            (CLAUDE_SESSION_HEADER, "session-a"),
            (CLAUDE_AGENT_HEADER, "agent-child"),
            (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
        ]));

        assert!(matches!(
            identity.lane(),
            Some(AgentLaneKey::Agent { agent_id, .. }) if agent_id == "agent-child"
        ));
        assert_eq!(identity.parent_agent_id(), Some("agent-parent"));
    }

    #[test]
    fn empty_identity_is_stateless() {
        let identity = RequestIdentity::from_headers(&HeaderMap::new());

        assert!(identity.lane().is_none());
    }

    #[test]
    fn ambiguous_identity_tuples_are_stateless() {
        let missing_session =
            RequestIdentity::from_headers(&headers(&[(CLAUDE_AGENT_HEADER, "agent-a")]));
        let parent_without_agent = RequestIdentity::from_headers(&headers(&[
            (CLAUDE_SESSION_HEADER, "session-a"),
            (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
        ]));

        assert!(missing_session.lane().is_none());
        assert!(parent_without_agent.lane().is_none());
    }

    #[test]
    fn malformed_identity_headers_are_stateless() {
        let identity =
            RequestIdentity::from_headers(&headers(&[(CLAUDE_SESSION_HEADER, "session a")]));

        assert!(identity.session_id().is_none());
        assert!(identity.lane().is_none());
    }

    #[test]
    fn comma_in_identity_header_is_rejected() {
        for value in ["first,second", "first,", ",second", "first,first"] {
            let session = RequestIdentity::from_headers(&headers(&[
                (CLAUDE_SESSION_HEADER, value),
                (CLAUDE_AGENT_HEADER, "agent-child"),
                (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
            ]));
            assert!(session.session_id().is_none(), "value={value}");
            assert!(session.lane().is_none(), "value={value}");

            let agent = RequestIdentity::from_headers(&headers(&[
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, value),
                (CLAUDE_PARENT_AGENT_HEADER, "agent-parent"),
            ]));
            assert!(agent.agent_id().is_none(), "value={value}");
            assert!(agent.lane().is_none(), "value={value}");

            let parent = RequestIdentity::from_headers(&headers(&[
                (CLAUDE_SESSION_HEADER, "session-a"),
                (CLAUDE_AGENT_HEADER, "agent-child"),
                (CLAUDE_PARENT_AGENT_HEADER, value),
            ]));
            assert!(parent.parent_agent_id().is_none(), "value={value}");
            assert!(parent.lane().is_none(), "value={value}");
        }
    }

    #[test]
    fn duplicate_identity_headers_fall_back_to_stateless() {
        let mut headers = headers(&[(CLAUDE_SESSION_HEADER, "session-a")]);
        headers.append(CLAUDE_SESSION_HEADER, HeaderValue::from_static("session-b"));

        assert!(RequestIdentity::from_headers(&headers).lane().is_none());
    }

    #[test]
    fn opaque_tokens_are_stable_and_domain_separated() {
        let lane = AgentLaneKey::Agent {
            session_id: "session-a".to_string(),
            agent_id: "agent-a".to_string(),
        };

        assert_eq!(lane.opaque_token("codex"), lane.opaque_token("codex"));
        assert_ne!(lane.opaque_token("codex"), lane.opaque_token("kimi"));
        assert!(!lane.opaque_token("codex").contains("agent-a"));
    }

    #[test]
    fn auxiliary_scope_has_no_conversational_lane() {
        let scope = RequestScope::from_headers(
            &headers(&[(CLAUDE_SESSION_HEADER, "session-a")]),
            RequestPurpose::CountTokens,
        );

        assert!(scope.lane().is_some());
        assert!(scope.conversational_lane().is_none());
    }
}
