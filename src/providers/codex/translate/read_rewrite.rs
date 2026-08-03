use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use once_cell::sync::Lazy;
use serde_json::Value;

use crate::request_identity::OpaqueLane;

const MAX_REWRITE_NOTES: usize = 4_096;
const READ_OFFSET_REWRITE_THRESHOLD: i64 = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadOffsetRewrite {
    pub offset: i64,
    pub file_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ReadRewriteScope {
    Stable(OpaqueLane),
    Legacy,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReadRewriteKey {
    scope: ReadRewriteScope,
    call_id: String,
}

#[derive(Debug, Default)]
struct RewriteStore {
    order: VecDeque<ReadRewriteKey>,
    entries: HashMap<ReadRewriteKey, ReadOffsetRewrite>,
}

static READ_OFFSET_REWRITES: Lazy<Mutex<RewriteStore>> =
    Lazy::new(|| Mutex::new(RewriteStore::default()));

/// Compatibility wrapper using a dedicated legacy namespace.
pub fn sanitize_read_args(name: &str, args: &str, call_id: Option<&str>) -> String {
    sanitize_read_args_in_scope(name, args, call_id, Some(ReadRewriteScope::Legacy))
}

pub(crate) fn sanitize_read_args_scoped(
    name: &str,
    args: &str,
    call_id: Option<&str>,
    lane: Option<OpaqueLane>,
) -> String {
    sanitize_read_args_in_scope(name, args, call_id, lane.map(ReadRewriteScope::Stable))
}

fn sanitize_read_args_in_scope(
    name: &str,
    args: &str,
    call_id: Option<&str>,
    scope: Option<ReadRewriteScope>,
) -> String {
    if name != "Read" || args.is_empty() {
        return args.to_string();
    }

    let parsed: Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return args.to_string(),
    };
    let obj = match parsed.as_object() {
        Some(o) => o,
        None => return args.to_string(),
    };

    let mut sanitized = obj.clone();
    let mut changed = false;

    let has_empty_pages = obj
        .get("pages")
        .and_then(|v| v.as_str())
        .is_some_and(|s| s.is_empty());
    if has_empty_pages {
        sanitized.remove("pages");
        changed = true;
    }

    if let Some(offset) = obj.get("offset").and_then(|v| v.as_i64())
        && offset >= READ_OFFSET_REWRITE_THRESHOLD
    {
        sanitized.remove("offset");
        changed = true;
        if let (Some(scope), Some(call_id)) = (scope, call_id.filter(|id| !id.is_empty())) {
            record_read_offset_rewrite(
                ReadRewriteKey {
                    scope,
                    call_id: call_id.to_string(),
                },
                ReadOffsetRewrite {
                    offset,
                    file_path: obj
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                },
            );
        }
    }

    if changed {
        serde_json::to_string(&sanitized).unwrap_or_else(|_| args.to_string())
    } else {
        args.to_string()
    }
}

/// Compatibility wrapper using the same registry's legacy namespace.
pub fn read_offset_rewrite(call_id: &str) -> Option<ReadOffsetRewrite> {
    read_offset_rewrite_in_scope(ReadRewriteScope::Legacy, call_id)
}

pub(crate) fn read_offset_rewrite_scoped(
    lane: Option<OpaqueLane>,
    call_id: &str,
) -> Option<ReadOffsetRewrite> {
    read_offset_rewrite_in_scope(ReadRewriteScope::Stable(lane?), call_id)
}

fn read_offset_rewrite_in_scope(
    scope: ReadRewriteScope,
    call_id: &str,
) -> Option<ReadOffsetRewrite> {
    let key = ReadRewriteKey {
        scope,
        call_id: call_id.to_string(),
    };
    READ_OFFSET_REWRITES
        .lock()
        .ok()
        .and_then(|store| store.entries.get(&key).cloned())
}

fn record_read_offset_rewrite(key: ReadRewriteKey, note: ReadOffsetRewrite) {
    let Ok(mut store) = READ_OFFSET_REWRITES.lock() else {
        return;
    };

    if !store.entries.contains_key(&key) {
        store.order.push_back(key.clone());
    }
    store.entries.insert(key, note);

    while store.entries.len() > MAX_REWRITE_NOTES {
        let Some(oldest) = store.order.pop_front() else {
            break;
        };
        store.entries.remove(&oldest);
    }
}

#[cfg(test)]
pub(crate) static READ_REWRITE_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
fn clear_rewrites() {
    let mut store = READ_OFFSET_REWRITES.lock().unwrap();
    store.entries.clear();
    store.order.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_identity::{ConversationIdentity, LaneDomain, RequestPurpose, RequestScope};

    fn lane(session: &str, agent: &str) -> OpaqueLane {
        RequestScope::from_conversation_identity(
            Some(ConversationIdentity::Agent(
                session.to_string(),
                agent.to_string(),
            )),
            RequestPurpose::Conversation,
        )
        .provider_lane(LaneDomain::CodexReadRewrite)
        .unwrap()
    }

    #[test]
    fn sanitize_read_args_removes_empty_pages() {
        let _lock = READ_REWRITE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let args = r#"{"file_path":"/tmp/a","pages":""}"#;
        let sanitized = sanitize_read_args("Read", args, None);
        let parsed: Value = serde_json::from_str(&sanitized).unwrap();
        assert!(parsed.get("pages").is_none());
        assert_eq!(
            parsed.get("file_path").and_then(|v| v.as_str()),
            Some("/tmp/a")
        );
    }

    #[test]
    fn sanitize_read_args_drops_and_records_absurd_offset() {
        let _lock = READ_REWRITE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_rewrites();
        let args = r#"{"file_path":"/tmp/a","offset":1300000,"limit":20}"#;
        let sanitized = sanitize_read_args("Read", args, Some("call_rewrite_test"));
        let parsed: Value = serde_json::from_str(&sanitized).unwrap();
        assert!(parsed.get("offset").is_none());
        assert_eq!(parsed.get("limit").and_then(|v| v.as_i64()), Some(20));

        let note = read_offset_rewrite("call_rewrite_test").unwrap();
        assert_eq!(note.offset, 1_300_000);
        assert_eq!(note.file_path.as_deref(), Some("/tmp/a"));
    }

    #[test]
    fn sanitize_read_args_keeps_normal_offset() {
        let _lock = READ_REWRITE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_rewrites();
        let args = r#"{"file_path":"/tmp/a","offset":1300,"limit":20}"#;
        let sanitized = sanitize_read_args("Read", args, Some("call_keep_test"));
        let parsed: Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed.get("offset").and_then(|v| v.as_i64()), Some(1_300));
        assert!(read_offset_rewrite("call_keep_test").is_none());
    }

    #[test]
    fn read_rewrite_lookup_survives_route_rollover_without_crossing_lanes() {
        let _lock = READ_REWRITE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_rewrites();
        let first = lane("session", "agent-one");
        let second = lane("session", "agent-two");
        let args = r#"{"file_path":"/tmp/a","offset":1300000}"#;
        sanitize_read_args_scoped("Read", args, Some("same-call"), Some(first));

        assert!(read_offset_rewrite_scoped(Some(first), "same-call").is_some());
        assert!(read_offset_rewrite_scoped(Some(second), "same-call").is_none());
        assert!(read_offset_rewrite("same-call").is_none());
        // The key has no route component, so a route rebuild uses the same lane.
        assert!(read_offset_rewrite_scoped(Some(first), "same-call").is_some());
    }

    #[test]
    fn stateless_sanitization_records_nothing() {
        let _lock = READ_REWRITE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_rewrites();
        let args = r#"{"file_path":"/tmp/a","offset":1300000}"#;
        let sanitized = sanitize_read_args_scoped("Read", args, Some("stateless"), None);
        assert!(!sanitized.contains("offset"));
        assert!(read_offset_rewrite_scoped(None, "stateless").is_none());
        assert_eq!(READ_OFFSET_REWRITES.lock().unwrap().entries.len(), 0);
    }

    #[test]
    fn rewrite_registry_evicts_at_4096_entries() {
        let _lock = READ_REWRITE_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        clear_rewrites();
        let lane = lane("session", "bounded");
        let args = r#"{"offset":1300000}"#;
        for index in 0..=MAX_REWRITE_NOTES {
            sanitize_read_args_scoped("Read", args, Some(&format!("call-{index}")), Some(lane));
        }
        let store = READ_OFFSET_REWRITES.lock().unwrap();
        assert_eq!(store.entries.len(), MAX_REWRITE_NOTES);
        drop(store);
        assert!(read_offset_rewrite_scoped(Some(lane), "call-0").is_none());
        assert!(
            read_offset_rewrite_scoped(Some(lane), &format!("call-{MAX_REWRITE_NOTES}")).is_some()
        );
    }
}
