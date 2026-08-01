use base64::Engine;
use sha2::{Digest, Sha256};

use super::auth::token_store::StoredAuth;

const KEY_VERSION: &str = "codex-state-v1";
const PROMPT_CACHE_KEY_MAX_LEN: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolLane {
    ResponsesFull,
    ResponsesLite,
}

impl ProtocolLane {
    pub fn from_responses_lite(use_responses_lite: bool) -> Self {
        if use_responses_lite {
            Self::ResponsesLite
        } else {
            Self::ResponsesFull
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ResponsesFull => "responses-full",
            Self::ResponsesLite => "responses-lite",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConversationBinding(String);

impl ConversationBinding {
    pub fn for_request(endpoint: &str, auth: &StoredAuth, protocol_lane: ProtocolLane) -> Self {
        let principal = auth
            .account_id
            .as_deref()
            .map(str::to_string)
            .unwrap_or_else(|| fingerprint("principal", auth.access.as_bytes()));
        let canonical_endpoint = endpoint.trim_end_matches('/');
        let credential_generation = fingerprint("credential", auth.access.as_bytes());
        let mut material = Vec::new();
        for value in [
            canonical_endpoint,
            &principal,
            &credential_generation,
            protocol_lane.label(),
        ] {
            material.extend_from_slice(value.as_bytes());
            material.push(0);
        }
        Self(fingerprint("conversation-binding", &material))
    }

    pub fn bind_lane(&self, lane_token: &str) -> String {
        let mut material = Vec::new();
        material.extend_from_slice(lane_token.as_bytes());
        material.push(0);
        material.extend_from_slice(self.0.as_bytes());
        let key = format!(
            "codex-conversation-v1{}",
            fingerprint("bound-conversation", &material)
        );
        debug_assert!(key.len() <= PROMPT_CACHE_KEY_MAX_LEN);
        key
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SocketPoolKey(String);

impl SocketPoolKey {
    pub fn for_request(
        lane_token: &str,
        endpoint: &str,
        auth: &StoredAuth,
        protocol_lane: ProtocolLane,
    ) -> Self {
        let principal = auth
            .account_id
            .as_deref()
            .map(str::to_string)
            .unwrap_or_else(|| fingerprint("principal", auth.access.as_bytes()));
        let credential_generation = fingerprint("credential", auth.access.as_bytes());
        let canonical_endpoint = endpoint.trim_end_matches('/');
        let mut material = Vec::new();
        for value in [
            KEY_VERSION,
            lane_token,
            canonical_endpoint,
            &principal,
            &credential_generation,
            protocol_lane.label(),
        ] {
            material.extend_from_slice(value.as_bytes());
            material.push(0);
        }
        Self(format!(
            "codex-pool-v1-{}",
            fingerprint("socket-pool", &material)
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn fingerprint(domain: &str, value: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(KEY_VERSION.as_bytes());
    digest.update([0]);
    digest.update(domain.as_bytes());
    digest.update([0]);
    digest.update(value);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auth(account_id: Option<&str>, access: &str) -> StoredAuth {
        StoredAuth {
            access: access.to_string(),
            refresh: "refresh".to_string(),
            expires: 1,
            account_id: account_id.map(str::to_string),
        }
    }

    #[test]
    fn conversation_binding_rolls_over_route_account_credential_and_protocol() {
        let bind = |endpoint: &str, account: Option<&str>, access: &str, protocol: ProtocolLane| {
            ConversationBinding::for_request(endpoint, &auth(account, access), protocol)
                .bind_lane("lane-a")
        };
        let baseline = bind(
            "https://example.test/responses",
            Some("acct-a"),
            "token-a",
            ProtocolLane::ResponsesLite,
        );
        assert_eq!(
            baseline,
            bind(
                "https://example.test/responses/",
                Some("acct-a"),
                "token-a",
                ProtocolLane::ResponsesLite,
            )
        );
        assert_ne!(
            baseline,
            bind(
                "https://other.test/responses",
                Some("acct-a"),
                "token-a",
                ProtocolLane::ResponsesLite,
            )
        );
        assert_ne!(
            baseline,
            bind(
                "https://example.test/responses",
                Some("acct-b"),
                "token-a",
                ProtocolLane::ResponsesLite,
            )
        );
        assert_ne!(
            baseline,
            bind(
                "https://example.test/responses",
                Some("acct-a"),
                "token-b",
                ProtocolLane::ResponsesLite,
            )
        );
        assert_ne!(
            baseline,
            bind(
                "https://example.test/responses",
                Some("acct-a"),
                "token-a",
                ProtocolLane::ResponsesFull,
            )
        );
        assert!(!baseline.contains("acct-a"));
        assert!(!baseline.contains("token-a"));
    }

    #[test]
    fn token_only_rotation_changes_conversation_binding_and_socket_pool_key() {
        let endpoint = "https://example.test/responses";
        let protocol = ProtocolLane::ResponsesLite;
        let before = auth(Some("acct-a"), "token-before");
        let after = auth(Some("acct-a"), "token-after");

        assert_ne!(
            ConversationBinding::for_request(endpoint, &before, protocol),
            ConversationBinding::for_request(endpoint, &after, protocol),
        );
        assert_ne!(
            SocketPoolKey::for_request("lane-a", endpoint, &before, protocol),
            SocketPoolKey::for_request("lane-a", endpoint, &after, protocol),
        );
    }

    #[test]
    fn bound_conversation_key_fits_upstream_prompt_cache_limit() {
        let key = ConversationBinding::for_request(
            "https://example.test/responses",
            &auth(Some("acct-a"), "token-a"),
            ProtocolLane::ResponsesLite,
        )
        .bind_lane("lane-a");

        assert_eq!(key.len(), PROMPT_CACHE_KEY_MAX_LEN);
        assert!(key.is_ascii());
        let digest = key
            .strip_prefix("codex-conversation-v1")
            .expect("bounded prompt cache key prefix");
        assert_eq!(digest.len(), 43);
        assert!(
            digest
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        );
    }

    #[test]
    fn socket_key_is_bound_to_lane_route_and_credential() {
        let baseline = SocketPoolKey::for_request(
            "lane-a",
            "https://example.test/responses",
            &auth(Some("acct-a"), "token-a"),
            ProtocolLane::ResponsesLite,
        );
        let same = SocketPoolKey::for_request(
            "lane-a",
            "https://example.test/responses/",
            &auth(Some("acct-a"), "token-a"),
            ProtocolLane::ResponsesLite,
        );
        assert_eq!(baseline, same);
        assert_ne!(
            baseline,
            SocketPoolKey::for_request(
                "lane-b",
                "https://example.test/responses",
                &auth(Some("acct-a"), "token-a"),
                ProtocolLane::ResponsesLite,
            )
        );
        assert_ne!(
            baseline,
            SocketPoolKey::for_request(
                "lane-a",
                "https://other.test/responses",
                &auth(Some("acct-a"), "token-a"),
                ProtocolLane::ResponsesLite,
            )
        );
        assert_ne!(
            baseline,
            SocketPoolKey::for_request(
                "lane-a",
                "https://example.test/responses",
                &auth(Some("acct-b"), "token-a"),
                ProtocolLane::ResponsesLite,
            )
        );
        assert_ne!(
            baseline,
            SocketPoolKey::for_request(
                "lane-a",
                "https://example.test/responses",
                &auth(Some("acct-a"), "token-b"),
                ProtocolLane::ResponsesLite,
            )
        );
        assert_ne!(
            baseline,
            SocketPoolKey::for_request(
                "lane-a",
                "https://example.test/responses",
                &auth(Some("acct-a"), "token-a"),
                ProtocolLane::ResponsesFull,
            )
        );
        assert!(!baseline.as_str().contains("acct-a"));
        assert!(!baseline.as_str().contains("token-a"));
    }
}
