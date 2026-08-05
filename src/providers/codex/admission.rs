use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use once_cell::sync::Lazy;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use crate::request_identity::{LaneDomain, OpaqueLane, RequestScope};

#[allow(dead_code)]
static ADMISSION: Lazy<Mutex<HashMap<OpaqueLane, Weak<AsyncMutex<()>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Serializes conversational Messages turns within one canonical Codex lane.
/// Stateless and non-conversational scopes intentionally bypass admission.
#[allow(dead_code)]
pub(crate) async fn admit_messages(scope: &RequestScope) -> Option<OwnedMutexGuard<()>> {
    let lane = scope.provider_lane(LaneDomain::CodexConversation)?;
    let admission = {
        let mut registry = ADMISSION.lock().unwrap();
        registry.retain(|_, entry| entry.strong_count() != 0);
        match registry.get(&lane).and_then(Weak::upgrade) {
            Some(admission) => admission,
            None => {
                let admission = Arc::new(AsyncMutex::new(()));
                registry.insert(lane, Arc::downgrade(&admission));
                admission
            }
        }
    };

    Some(admission.lock_owned().await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_identity::{ConversationIdentity, RequestPurpose};
    use tokio::sync::{Notify, oneshot};

    static TEST_LOCK: AsyncMutex<()> = AsyncMutex::const_new(());

    fn scope(identity: ConversationIdentity, purpose: RequestPurpose) -> RequestScope {
        RequestScope::from_conversation_identity(Some(identity), purpose)
    }

    fn main_scope(session: &str) -> RequestScope {
        scope(
            ConversationIdentity::Main(session.to_string()),
            RequestPurpose::Conversation,
        )
    }

    fn agent_scope(session: &str, agent: &str) -> RequestScope {
        scope(
            ConversationIdentity::Agent(session.to_string(), agent.to_string()),
            RequestPurpose::Conversation,
        )
    }

    fn clear_registry() {
        ADMISSION.lock().unwrap().clear();
    }

    #[tokio::test]
    async fn same_lane_waiters_are_serialized_and_retained() {
        let _test = TEST_LOCK.lock().await;
        clear_registry();
        let scope = main_scope("same-lane");
        let lane = scope.provider_lane(LaneDomain::CodexConversation).unwrap();
        let first = admit_messages(&scope).await.unwrap();
        let started = Arc::new(Notify::new());
        let waiter_started = started.clone();
        let (acquired_tx, mut acquired_rx) = oneshot::channel();
        let waiter = tokio::spawn(async move {
            waiter_started.notify_one();
            let guard = admit_messages(&scope).await.unwrap();
            let _ = acquired_tx.send(());
            guard
        });

        started.notified().await;
        assert!(matches!(
            acquired_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        assert!(
            ADMISSION
                .lock()
                .unwrap()
                .get(&lane)
                .and_then(Weak::upgrade)
                .is_some(),
            "the weak entry must remain live while a waiter is queued"
        );
        let independent = admit_messages(&main_scope("independent-lane"))
            .await
            .unwrap();
        drop(independent);

        drop(first);
        acquired_rx.await.unwrap();
        drop(waiter.await.unwrap());
    }

    #[tokio::test]
    async fn sibling_agent_lanes_are_independent() {
        let _test = TEST_LOCK.lock().await;
        clear_registry();
        let first = admit_messages(&agent_scope("session", "agent-a"))
            .await
            .unwrap();
        let sibling = admit_messages(&agent_scope("session", "agent-b"))
            .await
            .unwrap();
        drop(sibling);
        drop(first);
    }

    #[tokio::test]
    async fn non_conversational_and_stateless_scopes_bypass_admission() {
        let _test = TEST_LOCK.lock().await;
        clear_registry();
        let auxiliary = scope(
            ConversationIdentity::Main("auxiliary".to_string()),
            RequestPurpose::Auxiliary,
        );
        let stateless =
            RequestScope::from_conversation_identity(None, RequestPurpose::Conversation);

        assert!(admit_messages(&auxiliary).await.is_none());
        assert!(admit_messages(&stateless).await.is_none());
        assert!(ADMISSION.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn expired_weak_entries_are_cleaned_on_next_admission() {
        let _test = TEST_LOCK.lock().await;
        clear_registry();
        let first_scope = main_scope("expired");
        let first_lane = first_scope
            .provider_lane(LaneDomain::CodexConversation)
            .unwrap();
        drop(admit_messages(&first_scope).await.unwrap());
        assert!(ADMISSION.lock().unwrap().contains_key(&first_lane));

        let second_scope = main_scope("current");
        let second_lane = second_scope
            .provider_lane(LaneDomain::CodexConversation)
            .unwrap();
        let second = admit_messages(&second_scope).await.unwrap();
        let registry = ADMISSION.lock().unwrap();
        assert!(!registry.contains_key(&first_lane));
        assert!(registry.contains_key(&second_lane));
        drop(registry);
        drop(second);
    }
}
