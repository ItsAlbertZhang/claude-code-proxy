use crate::config::AliasProvider;
use crate::registry::normalize_incoming_model;
use crate::request_identity::AgentLaneKey;
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
    map: HashMap<AgentLaneKey, SessionState>,
    order: VecDeque<AgentLaneKey>,
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

pub fn existing_session(session_id: Option<&str>, now: u64) -> Option<SessionState> {
    let lane = session_id.map(AgentLaneKey::main);
    existing_session_for_lane(lane.as_ref(), now)
}

pub fn existing_session_for_lane(lane: Option<&AgentLaneKey>, now: u64) -> Option<SessionState> {
    let lane = lane?;
    let mut store = SESSIONS.lock().expect("session lock");
    let state = store.map.get(lane).cloned()?;
    if now.saturating_sub(state.last_seen) > SESSION_IDLE_TTL_MS {
        store.map.remove(lane);
        store.order.retain(|item| item != lane);
        return None;
    }
    Some(state)
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
    record_session_request_with_affinity_update(session_id, prior, provider_name, model, true, now)
}

pub(crate) fn record_session_request_with_affinity_update(
    session_id: Option<&str>,
    prior: Option<&SessionState>,
    provider_name: &str,
    model: &str,
    update_affinity: bool,
    now: u64,
) -> Option<SessionState> {
    let lane = session_id.map(AgentLaneKey::main);
    record_session_request_for_lane(
        lane.as_ref(),
        prior,
        provider_name,
        model,
        update_affinity,
        now,
    )
}

pub(crate) fn record_session_request_for_lane(
    lane: Option<&AgentLaneKey>,
    prior: Option<&SessionState>,
    provider_name: &str,
    model: &str,
    update_affinity: bool,
    now: u64,
) -> Option<SessionState> {
    let lane = lane?;
    if !update_affinity {
        let store = SESSIONS.lock().expect("session lock");
        return store
            .map
            .get(lane)
            .filter(|state| now.saturating_sub(state.last_seen) <= SESSION_IDLE_TTL_MS)
            .cloned()
            .or_else(|| {
                prior
                    .filter(|state| now.saturating_sub(state.last_seen) <= SESSION_IDLE_TTL_MS)
                    .cloned()
            });
    }
    let mut store = SESSIONS.lock().expect("session lock");
    let stored = store
        .map
        .get(lane)
        .cloned()
        .filter(|state| now.saturating_sub(state.last_seen) <= SESSION_IDLE_TTL_MS);
    if stored.is_none() && store.map.remove(lane).is_some() {
        store.order.retain(|item| item != lane);
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
    next.seq += 1;
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

    if !store.map.contains_key(lane) {
        store.order.push_back(lane.clone());
    }
    store.map.insert(lane.clone(), next.clone());

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

    #[test]
    fn concurrent_requests_increment_latest_sequence() {
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
        let lane_a = AgentLaneKey::Agent {
            session_id: "shared-session".to_string(),
            agent_id: "agent-a".to_string(),
        };
        let lane_b = AgentLaneKey::Agent {
            session_id: "shared-session".to_string(),
            agent_id: "agent-b".to_string(),
        };

        let a =
            record_session_request_for_lane(Some(&lane_a), None, "codex", "gpt-5.6-sol", true, 1)
                .unwrap();
        let b = record_session_request_for_lane(
            Some(&lane_b),
            None,
            "kimi",
            "kimi-for-coding",
            true,
            2,
        )
        .unwrap();
        let a2 = record_session_request_for_lane(
            Some(&lane_a),
            Some(&a),
            "codex",
            "gpt-5.6-sol",
            true,
            3,
        )
        .unwrap();

        assert_eq!(a2.seq, 2);
        assert_eq!(a2.affinity_provider, Some(AliasProvider::Codex));
        assert_eq!(b.seq, 1);
        assert_eq!(b.affinity_provider, Some(AliasProvider::Kimi));
        assert_eq!(
            existing_session_for_lane(Some(&lane_b), 4)
                .unwrap()
                .affinity_provider,
            Some(AliasProvider::Kimi)
        );
    }

    #[test]
    fn auxiliary_request_does_not_mutate_session_state() {
        let session_id = "session-affinity-auxiliary-request-test";
        let initial = record_session_request(Some(session_id), None, "codex", "gpt-5.6-sol", 1)
            .expect("initial session");
        assert_eq!(initial.affinity_provider, Some(AliasProvider::Codex));

        let after_review = record_session_request_with_affinity_update(
            Some(session_id),
            Some(&initial),
            "kimi",
            "kimi-for-coding",
            false,
            2,
        )
        .expect("existing session");
        assert_eq!(after_review.seq, 1);
        assert_eq!(after_review.last_seen, 1);
        assert_eq!(after_review.affinity_provider, Some(AliasProvider::Codex));
        assert_eq!(existing_session(Some(session_id), 2).unwrap().last_seen, 1);
    }
}
