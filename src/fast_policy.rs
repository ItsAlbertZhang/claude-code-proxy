use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum FastPolicy {
    #[default]
    Passthrough = 0,
    ForceFast = 1,
    ForceOff = 2,
}

impl FastPolicy {
    fn from_raw(value: u8) -> Self {
        match value {
            1 => Self::ForceFast,
            2 => Self::ForceOff,
            _ => Self::Passthrough,
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Passthrough => Self::ForceFast,
            Self::ForceFast => Self::ForceOff,
            Self::ForceOff => Self::Passthrough,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Passthrough => "AUTO",
            Self::ForceFast => "ON",
            Self::ForceOff => "OFF",
        }
    }

    pub fn resolve(self, requested_fast: bool) -> bool {
        match self {
            Self::Passthrough => requested_fast,
            Self::ForceFast => true,
            Self::ForceOff => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClaudeFastDecision {
    requested_fast: bool,
    policy: FastPolicy,
}

impl ClaudeFastDecision {
    pub fn new(requested_fast: bool, policy: FastPolicy) -> Self {
        Self {
            requested_fast,
            policy,
        }
    }

    pub fn passthrough(requested_fast: bool) -> Self {
        Self::new(requested_fast, FastPolicy::Passthrough)
    }

    pub fn requested_fast(self) -> bool {
        self.requested_fast
    }

    pub fn policy(self) -> FastPolicy {
        self.policy
    }

    pub fn effective_fast(self) -> bool {
        self.policy.resolve(self.requested_fast)
    }
}

#[derive(Clone, Debug, Default)]
pub struct FastPolicyHandle {
    value: Arc<AtomicU8>,
}

impl FastPolicyHandle {
    pub fn get(&self) -> FastPolicy {
        FastPolicy::from_raw(self.value.load(Ordering::Acquire))
    }

    pub fn set(&self, policy: FastPolicy) {
        self.value.store(policy as u8, Ordering::Release);
    }

    pub fn cycle(&self) -> FastPolicy {
        let previous = self
            .value
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                Some(FastPolicy::from_raw(value).next() as u8)
            })
            .unwrap_or_else(|value| value);
        FastPolicy::from_raw(previous).next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_resolves_and_cycles_all_states() {
        for (policy, requested, expected) in [
            (FastPolicy::Passthrough, false, false),
            (FastPolicy::Passthrough, true, true),
            (FastPolicy::ForceFast, false, true),
            (FastPolicy::ForceFast, true, true),
            (FastPolicy::ForceOff, false, false),
            (FastPolicy::ForceOff, true, false),
        ] {
            assert_eq!(policy.resolve(requested), expected);
        }

        assert_eq!(FastPolicy::Passthrough.next(), FastPolicy::ForceFast);
        assert_eq!(FastPolicy::ForceFast.next(), FastPolicy::ForceOff);
        assert_eq!(FastPolicy::ForceOff.next(), FastPolicy::Passthrough);
    }

    #[test]
    fn cloned_handles_share_updates_and_cycle() {
        let handle = FastPolicyHandle::default();
        let clone = handle.clone();

        assert_eq!(handle.get(), FastPolicy::Passthrough);
        assert_eq!(clone.cycle(), FastPolicy::ForceFast);
        assert_eq!(handle.get(), FastPolicy::ForceFast);
        handle.set(FastPolicy::ForceOff);
        assert_eq!(clone.get(), FastPolicy::ForceOff);
    }

    #[test]
    fn invalid_encoding_falls_back_to_passthrough() {
        let handle = FastPolicyHandle {
            value: Arc::new(AtomicU8::new(u8::MAX)),
        };

        assert_eq!(handle.get(), FastPolicy::Passthrough);
        assert_eq!(handle.cycle(), FastPolicy::ForceFast);
    }
}
