use std::error::Error;
use std::fmt;

use base64::Engine;
use sha2::{Digest, Sha256, Sha384};
use url::Url;

use crate::request_identity::OpaqueLane;

use super::auth::token_store::StoredAuth;
use super::translate::request::{ResponsesRequest, request_uses_responses_lite};

const ROUTE_KEY_VERSION: &[u8] = b"ccp-codex-route-v1";
const CONVERSATION_KEY_VERSION: &[u8] = b"ccp-codex-conversation-v1";
const PROMPT_CACHE_NAMESPACE_VERSION: &[u8] = b"ccp-codex-prompt-cache-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ProtocolLane {
    ResponsesFull,
    ResponsesLite,
}

impl ProtocolLane {
    pub(crate) fn from_uses_responses_lite(uses_responses_lite: bool) -> Self {
        if uses_responses_lite {
            Self::ResponsesLite
        } else {
            Self::ResponsesFull
        }
    }

    pub(crate) fn uses_responses_lite(self) -> bool {
        matches!(self, Self::ResponsesLite)
    }

    fn label(self) -> &'static [u8] {
        match self {
            Self::ResponsesFull => b"responses-full",
            Self::ResponsesLite => b"responses-lite",
        }
    }
}

impl fmt::Display for ProtocolLane {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResponsesFull => formatter.write_str("Responses Full"),
            Self::ResponsesLite => formatter.write_str("Responses Lite"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CredentialGeneration([u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PrincipalFingerprint([u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CodexRouteIdentity([u8; 32]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CodexConversationKey([u8; 48]);

impl CodexConversationKey {
    pub(crate) fn encode(self) -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SocketPoolKey {
    route_identity: CodexRouteIdentity,
    conversation_key: CodexConversationKey,
}

#[derive(Clone)]
pub(crate) struct CodexBoundRoute {
    auth: StoredAuth,
    canonical_endpoint: Url,
    account_id: Option<String>,
    principal: PrincipalFingerprint,
    credential_generation: CredentialGeneration,
    protocol: ProtocolLane,
    lane: Option<OpaqueLane>,
    route_identity: CodexRouteIdentity,
    conversation_key: Option<CodexConversationKey>,
    socket_pool_key: Option<SocketPoolKey>,
}

impl fmt::Debug for CodexBoundRoute {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexBoundRoute")
            .field("canonical_endpoint", &self.canonical_endpoint)
            .field("has_known_account", &self.account_id.is_some())
            .field("principal", &self.principal)
            .field("credential_generation", &self.credential_generation)
            .field("protocol", &self.protocol)
            .field("lane", &self.lane)
            .field("route_identity", &self.route_identity)
            .field("conversation_key", &self.conversation_key)
            .field("socket_pool_key", &self.socket_pool_key)
            .finish_non_exhaustive()
    }
}

impl CodexBoundRoute {
    pub(crate) fn new(
        auth: StoredAuth,
        endpoint: &str,
        protocol: ProtocolLane,
        lane: Option<OpaqueLane>,
    ) -> Result<Self, url::ParseError> {
        let canonical_endpoint = canonicalize_endpoint(endpoint)?;
        let account_id = auth.account_id.clone();
        let principal = PrincipalFingerprint(match account_id.as_deref() {
            Some(account_id) => hash32(b"principal", &[b"account", account_id.as_bytes()]),
            None => hash32(b"principal", &[b"access-token", auth.access.as_bytes()]),
        });
        let credential_generation =
            CredentialGeneration(hash32(b"credential-generation", &[auth.access.as_bytes()]));
        let route_identity = CodexRouteIdentity(hash32(
            b"route-identity",
            &[
                canonical_endpoint.as_str().as_bytes(),
                &principal.0,
                &credential_generation.0,
                protocol.label(),
            ],
        ));
        let conversation_key = lane.map(|lane| {
            let mut digest = Sha384::new();
            update_length_prefixed(&mut digest, CONVERSATION_KEY_VERSION);
            update_length_prefixed(&mut digest, b"conversation-key");
            update_length_prefixed(&mut digest, &route_identity.0);
            update_length_prefixed(&mut digest, lane.as_bytes());
            CodexConversationKey(digest.finalize().into())
        });
        let socket_pool_key = conversation_key.map(|conversation_key| SocketPoolKey {
            route_identity,
            conversation_key,
        });

        Ok(Self {
            auth,
            canonical_endpoint,
            account_id,
            principal,
            credential_generation,
            protocol,
            lane,
            route_identity,
            conversation_key,
            socket_pool_key,
        })
    }

    pub(crate) fn auth(&self) -> &StoredAuth {
        &self.auth
    }

    pub(crate) fn canonical_endpoint(&self) -> &Url {
        &self.canonical_endpoint
    }

    pub(crate) fn account_id(&self) -> Option<&str> {
        self.account_id.as_deref()
    }

    pub(crate) fn credential_generation(&self) -> CredentialGeneration {
        self.credential_generation
    }

    pub(crate) fn protocol(&self) -> ProtocolLane {
        self.protocol
    }

    pub(crate) fn lane(&self) -> Option<OpaqueLane> {
        self.lane
    }

    pub(crate) fn route_identity(&self) -> CodexRouteIdentity {
        self.route_identity
    }

    pub(crate) fn route_identity_key(&self) -> [u8; 32] {
        self.route_identity.0
    }

    pub(crate) fn conversation_key(&self) -> Option<CodexConversationKey> {
        self.conversation_key
    }

    pub(crate) fn conversation_key_encoded(&self) -> Option<String> {
        self.conversation_key.map(CodexConversationKey::encode)
    }

    pub(crate) fn socket_pool_key(&self) -> Option<SocketPoolKey> {
        self.socket_pool_key
    }

    /// Preserves the bound origin, principal, credentials, and protocol while
    /// removing all conversation identity for detached auxiliary work.
    pub(crate) fn auxiliary(&self) -> Self {
        let mut route = self.clone();
        route.lane = None;
        route.conversation_key = None;
        route.socket_pool_key = None;
        route
    }

    pub(crate) fn validate_responses_request(
        &self,
        request: &ResponsesRequest,
    ) -> Result<(), CodexProtocolError> {
        self.validate_protocol(ProtocolLane::from_uses_responses_lite(
            request_uses_responses_lite(request),
        ))
    }

    pub(crate) fn validate_protocol(&self, actual: ProtocolLane) -> Result<(), CodexProtocolError> {
        if self.protocol == actual {
            Ok(())
        } else {
            Err(CodexProtocolError {
                expected: self.protocol,
                actual,
            })
        }
    }

    pub(crate) fn namespace_prompt_cache_key(&self, caller_key: &str) -> Option<String> {
        let conversation_key = self.conversation_key?;
        let mut digest = Sha384::new();
        update_length_prefixed(&mut digest, PROMPT_CACHE_NAMESPACE_VERSION);
        update_length_prefixed(&mut digest, b"caller-prompt-cache-key");
        update_length_prefixed(&mut digest, &conversation_key.0);
        update_length_prefixed(&mut digest, caller_key.as_bytes());
        Some(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest.finalize()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CodexProtocolError {
    expected: ProtocolLane,
    actual: ProtocolLane,
}

impl fmt::Display for CodexProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Codex route protocol mismatch: route uses {}, payload uses {}",
            self.expected, self.actual
        )
    }
}

impl Error for CodexProtocolError {}

fn canonicalize_endpoint(endpoint: &str) -> Result<Url, url::ParseError> {
    let mut endpoint = Url::parse(endpoint)?;
    let canonical_path = endpoint.path().trim_end_matches('/').to_string();
    endpoint.set_path(if canonical_path.is_empty() {
        "/"
    } else {
        &canonical_path
    });
    endpoint.set_fragment(None);
    if endpoint.query() == Some("") {
        endpoint.set_query(None);
    }
    Ok(endpoint)
}

fn hash32(domain: &[u8], components: &[&[u8]]) -> [u8; 32] {
    let mut digest = Sha256::new();
    update_length_prefixed(&mut digest, ROUTE_KEY_VERSION);
    update_length_prefixed(&mut digest, domain);
    for component in components {
        update_length_prefixed(&mut digest, component);
    }
    digest.finalize().into()
}

fn update_length_prefixed<D: Digest>(digest: &mut D, value: &[u8]) {
    Digest::update(digest, (value.len() as u64).to_be_bytes());
    Digest::update(digest, value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_identity::{LaneDomain, RequestPurpose, RequestScope};

    fn auth(access: &str, account_id: Option<&str>) -> StoredAuth {
        StoredAuth {
            access: access.to_string(),
            refresh: "refresh-secret".to_string(),
            expires: u64::MAX,
            account_id: account_id.map(str::to_string),
        }
    }

    fn lane(session: &str) -> OpaqueLane {
        RequestScope::legacy(Some(session), RequestPurpose::Conversation)
            .provider_lane(LaneDomain::CodexConversation)
            .unwrap()
    }

    fn route(
        endpoint: &str,
        account: Option<&str>,
        access: &str,
        protocol: ProtocolLane,
        lane: Option<OpaqueLane>,
    ) -> CodexBoundRoute {
        CodexBoundRoute::new(auth(access, account), endpoint, protocol, lane).unwrap()
    }

    #[test]
    fn conversation_binding_rolls_over_route_account_credential_and_protocol() {
        let lane = lane("session-a");
        let base = route(
            "https://example.test/backend-api/codex",
            Some("account-a"),
            "token-a",
            ProtocolLane::ResponsesFull,
            Some(lane),
        );
        for changed in [
            route(
                "https://other.test/backend-api/codex",
                Some("account-a"),
                "token-a",
                ProtocolLane::ResponsesFull,
                Some(lane),
            ),
            route(
                "https://example.test/backend-api/codex",
                Some("account-b"),
                "token-a",
                ProtocolLane::ResponsesFull,
                Some(lane),
            ),
            route(
                "https://example.test/backend-api/codex",
                Some("account-a"),
                "token-b",
                ProtocolLane::ResponsesFull,
                Some(lane),
            ),
            route(
                "https://example.test/backend-api/codex",
                Some("account-a"),
                "token-a",
                ProtocolLane::ResponsesLite,
                Some(lane),
            ),
        ] {
            assert_ne!(base.route_identity(), changed.route_identity());
            assert_ne!(base.conversation_key(), changed.conversation_key());
        }
    }

    #[test]
    fn canonical_equivalent_endpoints_have_equal_route_identity() {
        let lane = lane("session-a");
        let without_slash = route(
            "https://EXAMPLE.test:443/backend-api/codex?",
            Some("account-a"),
            "token-a",
            ProtocolLane::ResponsesFull,
            Some(lane),
        );
        let with_slash = route(
            "https://example.test/backend-api/codex/",
            Some("account-a"),
            "token-a",
            ProtocolLane::ResponsesFull,
            Some(lane),
        );
        assert_eq!(without_slash.route_identity(), with_slash.route_identity());
        assert_eq!(
            without_slash.conversation_key(),
            with_slash.conversation_key()
        );
    }

    #[test]
    fn token_only_rotation_changes_conversation_binding_and_socket_pool_key() {
        let lane = Some(lane("session-a"));
        let before = route(
            "https://example.test/backend-api/codex",
            None,
            "token-a",
            ProtocolLane::ResponsesFull,
            lane,
        );
        let after = route(
            "https://example.test/backend-api/codex",
            None,
            "token-b",
            ProtocolLane::ResponsesFull,
            lane,
        );
        assert_ne!(before.conversation_key(), after.conversation_key());
        assert_ne!(before.socket_pool_key(), after.socket_pool_key());
    }

    #[test]
    fn bound_conversation_key_fits_upstream_prompt_cache_limit() {
        let route = route(
            "https://example.test/backend-api/codex",
            Some("raw-account-secret"),
            "raw-access-secret",
            ProtocolLane::ResponsesFull,
            Some(lane("raw-session-secret")),
        );
        let encoded = route.conversation_key_encoded().unwrap();
        assert_eq!(encoded.len(), 64);
        assert!(
            encoded
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        );
        for raw in [
            "raw-account-secret",
            "raw-access-secret",
            "raw-session-secret",
        ] {
            assert!(!encoded.contains(raw));
        }
    }

    #[test]
    fn socket_key_is_bound_to_lane_route_and_credential() {
        let lane_a = lane("session-a");
        let lane_b = lane("session-b");
        let base = route(
            "https://example.test/backend-api/codex",
            Some("account-a"),
            "token-a",
            ProtocolLane::ResponsesFull,
            Some(lane_a),
        );
        let sibling = route(
            "https://example.test/backend-api/codex",
            Some("account-a"),
            "token-a",
            ProtocolLane::ResponsesFull,
            Some(lane_b),
        );
        let rotated = route(
            "https://example.test/backend-api/codex",
            Some("account-a"),
            "token-b",
            ProtocolLane::ResponsesFull,
            Some(lane_a),
        );
        assert_ne!(base.socket_pool_key(), sibling.socket_pool_key());
        assert_ne!(base.socket_pool_key(), rotated.socket_pool_key());
        assert!(
            route(
                "https://example.test/backend-api/codex",
                Some("account-a"),
                "token-a",
                ProtocolLane::ResponsesFull,
                None,
            )
            .socket_pool_key()
            .is_none()
        );
    }

    #[test]
    fn auxiliary_route_preserves_origin_and_auth_but_drops_conversation_identity() {
        let bound = route(
            "https://example.test/backend-api/codex",
            Some("account-a"),
            "token-a",
            ProtocolLane::ResponsesFull,
            Some(lane("session-a")),
        );
        let auxiliary = bound.auxiliary();

        assert_eq!(auxiliary.canonical_endpoint(), bound.canonical_endpoint());
        assert_eq!(auxiliary.account_id(), bound.account_id());
        assert_eq!(auxiliary.auth().access, bound.auth().access);
        assert_eq!(auxiliary.route_identity(), bound.route_identity());
        assert_eq!(auxiliary.protocol(), bound.protocol());
        assert!(auxiliary.lane().is_none());
        assert!(auxiliary.conversation_key().is_none());
        assert!(auxiliary.socket_pool_key().is_none());
        assert!(auxiliary.namespace_prompt_cache_key("caller-key").is_none());
    }

    #[test]
    fn caller_prompt_cache_key_is_namespaced_by_bound_conversation() {
        let route_a = route(
            "https://example.test/backend-api/codex",
            Some("account-a"),
            "token-a",
            ProtocolLane::ResponsesFull,
            Some(lane("session-a")),
        );
        let route_b = route(
            "https://example.test/backend-api/codex",
            Some("account-a"),
            "token-a",
            ProtocolLane::ResponsesFull,
            Some(lane("session-b")),
        );
        let key_a = route_a.namespace_prompt_cache_key("caller-key").unwrap();
        let key_b = route_b.namespace_prompt_cache_key("caller-key").unwrap();
        assert_eq!(key_a.len(), 64);
        assert_ne!(key_a, key_b);
        assert!(!key_a.contains("caller-key"));
        assert!(
            route(
                "https://example.test/backend-api/codex",
                Some("account-a"),
                "token-a",
                ProtocolLane::ResponsesFull,
                None,
            )
            .namespace_prompt_cache_key("caller-key")
            .is_none()
        );
    }
}
