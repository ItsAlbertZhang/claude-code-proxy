use crate::config::AliasProvider;
use crate::registry::normalize_incoming_model;
use crate::request_identity::{ConversationIdentity, RequestScope};
use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

const SESSION_IDLE_TTL_MS: u64 = 30 * 60 * 1000;
pub const MAX_SESSIONS: usize = 10_000;

#[derive(Debug, Clone)]
pub struct SessionState {
    pub seq: u64,
    pub affinity_provider: Option<AliasProvider>,
    pub last_seen: u64,
}

#[derive(Default)]
struct SessionStore {
    map: HashMap<ConversationIdentity, SessionState>,
    order: VecDeque<ConversationIdentity>,
}

static SESSIONS: LazyLock<Mutex<SessionStore>> =
    LazyLock::new(|| Mutex::new(SessionStore::default()));

fn now_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis() as u64
}

pub(crate) fn existing_conversation(
    identity: Option<&ConversationIdentity>,
    now: u64,
) -> Option<SessionState> {
    let identity = identity?.validated()?;
    let mut store = SESSIONS.lock().expect("session lock");
    let state = store.map.get(&identity).cloned()?;
    if now.saturating_sub(state.last_seen) > SESSION_IDLE_TTL_MS {
        store.map.remove(&identity);
        store.order.retain(|item| item != &identity);
        return None;
    }
    Some(state)
}

pub(crate) fn existing_conversation_now(
    identity: Option<&ConversationIdentity>,
) -> Option<SessionState> {
    existing_conversation(identity, now_millis())
}

pub fn existing_session(session_id: Option<&str>, now: u64) -> Option<SessionState> {
    let identity = session_id.and_then(ConversationIdentity::from_legacy_main);
    existing_conversation(identity.as_ref(), now)
}

pub fn existing_session_now(session_id: Option<&str>) -> Option<SessionState> {
    existing_session(session_id, now_millis())
}

pub fn record_session_request(
    session_id: Option<&str>,
    prior: Option<&SessionState>,
    provider_name: &str,
    model: &str,
    now: u64,
) -> Option<SessionState> {
    let identity = session_id.and_then(ConversationIdentity::from_legacy_main);
    record_conversation_request(identity.as_ref(), prior, provider_name, model, true, now)
}

pub(crate) fn record_session_request_with_affinity_update(
    session_id: Option<&str>,
    prior: Option<&SessionState>,
    provider_name: &str,
    model: &str,
    update_affinity: bool,
    now: u64,
) -> Option<SessionState> {
    let identity = session_id.and_then(ConversationIdentity::from_legacy_main);
    record_conversation_request(
        identity.as_ref(),
        prior,
        provider_name,
        model,
        update_affinity,
        now,
    )
}

pub(crate) fn record_scoped_request(
    scope: &RequestScope,
    prior: Option<&SessionState>,
    provider_name: &str,
    model: &str,
    update_affinity: bool,
    now: u64,
) -> Option<SessionState> {
    record_conversation_request(
        scope.conversational_lane(),
        prior,
        provider_name,
        model,
        update_affinity,
        now,
    )
}

pub(crate) fn record_conversation_request(
    identity: Option<&ConversationIdentity>,
    prior: Option<&SessionState>,
    provider_name: &str,
    model: &str,
    update_affinity: bool,
    now: u64,
) -> Option<SessionState> {
    let identity = identity?.validated()?;
    let mut store = SESSIONS.lock().expect("session lock");
    let stored = store
        .map
        .get(&identity)
        .cloned()
        .filter(|state| now.saturating_sub(state.last_seen) <= SESSION_IDLE_TTL_MS);
    if stored.is_none() && store.map.remove(&identity).is_some() {
        store.order.retain(|item| item != &identity);
    }
    let mut next = stored
        .or_else(|| {
            prior
                .filter(|state| now.saturating_sub(state.last_seen) <= SESSION_IDLE_TTL_MS)
                .cloned()
        })
        .unwrap_or(SessionState {
            seq: 0,
            affinity_provider: None,
            last_seen: now,
        });
    next.seq = next
        .seq
        .checked_add(1)
        .expect("conversation sequence exhausted");
    next.last_seen = now;
    if update_affinity
        && is_alias_routable_provider(provider_name)
        && !crate::registry::is_anthropic_alias(normalize_incoming_model(model).as_str())
    {
        next.affinity_provider = Some(match provider_name {
            "codex" => AliasProvider::Codex,
            "kimi" => AliasProvider::Kimi,
            _ => next.affinity_provider.unwrap_or(AliasProvider::Codex),
        });
    }

    if !store.map.contains_key(&identity) {
        store.order.push_back(identity.clone());
    }
    store.map.insert(identity, next.clone());

    while store.order.len() > MAX_SESSIONS {
        if let Some(evict) = store.order.pop_front() {
            store.map.remove(&evict);
        } else {
            break;
        }
    }

    Some(next)
}

fn is_alias_routable_provider(name: &str) -> bool {
    matches!(name, "codex" | "kimi")
}

#[cfg(test)]
pub fn reset_sessions_for_test() {
    let mut store = SESSIONS.lock().expect("session lock");
    store.map.clear();
    store.order.clear();
}

pub fn affinity_provider_from_session(session: &SessionState) -> Option<AliasProvider> {
    session.affinity_provider
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_identity::{RequestPurpose, RequestScope};

    #[test]
    fn concurrent_requests_increment_latest_sequence() {
        reset_sessions_for_test();
        let session_id = "session-concurrent-sequence-test";
        let initial = record_session_request(Some(session_id), None, "codex", "gpt-5.6-sol", 1)
            .expect("initial session");
        let handles: Vec<_> = (0..16)
            .map(|offset| {
                let stale = initial.clone();
                std::thread::spawn(move || {
                    record_session_request(
                        Some(session_id),
                        Some(&stale),
                        "codex",
                        "gpt-5.6-sol",
                        2 + offset,
                    )
                    .expect("recorded session")
                    .seq
                })
            })
            .collect();
        let mut sequences: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().expect("session thread"))
            .collect();
        sequences.sort_unstable();
        assert_eq!(sequences, (2..=17).collect::<Vec<_>>());
    }

    #[test]
    fn agent_lanes_keep_affinity_and_sequence_independent() {
        reset_sessions_for_test();
        let main = ConversationIdentity::Main("shared-session".to_string());
        let first =
            ConversationIdentity::Agent("shared-session".to_string(), "agent-one".to_string());
        let second =
            ConversationIdentity::Agent("shared-session".to_string(), "agent-two".to_string());
        let first_state =
            record_conversation_request(Some(&first), None, "codex", "gpt-5.6-sol", true, 1)
                .unwrap();
        let second_state =
            record_conversation_request(Some(&second), None, "kimi", "kimi-for-coding", true, 2)
                .unwrap();
        let main_state =
            record_conversation_request(Some(&main), None, "codex", "gpt-5.6-sol", true, 3)
                .unwrap();
        assert_eq!(first_state.seq, 1);
        assert_eq!(second_state.seq, 1);
        assert_eq!(main_state.seq, 1);
        assert_eq!(first_state.affinity_provider, Some(AliasProvider::Codex));
        assert_eq!(second_state.affinity_provider, Some(AliasProvider::Kimi));
    }

    #[test]
    fn auxiliary_request_does_not_mutate_session_state() {
        reset_sessions_for_test();
        let identity = ConversationIdentity::Main("session-auxiliary".to_string());
        let initial =
            record_conversation_request(Some(&identity), None, "codex", "gpt-5.6-sol", true, 1)
                .unwrap();
        for purpose in [
            RequestPurpose::CountTokens,
            RequestPurpose::AutoReview,
            RequestPurpose::Auxiliary,
        ] {
            let scope = RequestScope::from_conversation_identity(Some(identity.clone()), purpose);
            assert!(
                record_scoped_request(&scope, Some(&initial), "kimi", "kimi-for-coding", true, 2,)
                    .is_none()
            );
        }
        let after = existing_conversation(Some(&identity), 3).unwrap();
        assert_eq!(after.seq, 1);
        assert_eq!(after.last_seen, 1);
        assert_eq!(after.affinity_provider, Some(AliasProvider::Codex));
    }
}
