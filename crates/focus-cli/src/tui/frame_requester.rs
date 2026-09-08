//! Coalesced frame requests for the terminal projection.
//!
//! The state machine follows the scheduling model in Codex CLI's
//! `codex-rs/tui/src/tui/frame_requester.rs` (Apache-2.0), adapted for
//! Focus's synchronous event loop. It deliberately owns no transcript state.

use std::time::{Duration, Instant};

/// Schedules at most one pending frame and limits presentation rate.
#[derive(Debug)]
pub(super) struct FrameRequester {
    min_interval: Duration,
    last_drawn_at: Option<Instant>,
    requested: bool,
    scheduled: Option<Instant>,
}

impl FrameRequester {
    pub(super) fn new(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_drawn_at: None,
            requested: false,
            scheduled: None,
        }
    }

    /// Coalesce repeated requests until the next eligible frame is consumed.
    pub(super) fn request(&mut self, _now: Instant) {
        self.requested = true;
    }

    /// Schedule one future frame, coalescing it with an earlier deadline.
    pub(super) fn request_in(&mut self, now: Instant, delay: Duration) {
        let deadline = now + delay;
        if self.scheduled.is_none_or(|current| deadline < current) {
            self.scheduled = Some(deadline);
        }
    }

    /// Drop a future-only frame after the active status has ended.
    pub(super) fn cancel_scheduled(&mut self) {
        self.scheduled = None;
    }

    /// Consume the pending frame once it is outside the rate-limit window.
    pub(super) fn take_due(&mut self, now: Instant) -> bool {
        let scheduled_due = self.scheduled.is_some_and(|deadline| deadline <= now);
        if !self.requested && !scheduled_due {
            return false;
        }
        if let Some(last) = self.last_drawn_at
            && now.duration_since(last) < self.min_interval
        {
            return false;
        }
        self.requested = false;
        if scheduled_due {
            self.scheduled = None;
        }
        self.last_drawn_at = Some(now);
        true
    }

    /// Bound terminal event polling while keeping scheduled status frames dormant.
    pub(super) fn poll_timeout(&self, now: Instant, maximum: Duration) -> Duration {
        let rate_deadline = self.last_drawn_at.map(|last| last + self.min_interval);
        let next_deadline = match (self.requested, rate_deadline, self.scheduled) {
            (true, None, _) => Some(now),
            (true, Some(rate), Some(scheduled)) => Some(rate.min(scheduled)),
            (true, Some(rate), None) => Some(rate),
            (false, _, Some(scheduled)) => Some(scheduled),
            (false, _, None) => None,
        };
        next_deadline.map_or(maximum, |deadline| {
            deadline.saturating_duration_since(now).min(maximum)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::FrameRequester;

    #[test]
    fn coalesces_bursty_requests_into_one_due_frame() {
        let now = Instant::now();
        let mut requester = FrameRequester::new(Duration::from_millis(8));

        requester.request(now);
        requester.request(now);
        requester.request(now);

        assert!(requester.take_due(now));
        assert!(!requester.take_due(now));
    }

    #[test]
    fn delays_only_an_already_requested_frame() {
        let now = Instant::now();
        let mut requester = FrameRequester::new(Duration::from_millis(8));
        requester.request(now);
        assert!(requester.take_due(now));
        requester.request(now + Duration::from_millis(1));

        assert_eq!(
            requester.poll_timeout(now + Duration::from_millis(1), Duration::from_millis(50)),
            Duration::from_millis(7)
        );
    }

    #[test]
    fn schedules_one_delayed_frame_and_can_cancel_it() {
        let now = Instant::now();
        let mut requester = FrameRequester::new(Duration::from_millis(8));
        requester.request_in(now, Duration::from_secs(1));

        assert_eq!(
            requester.poll_timeout(now, Duration::from_secs(5)),
            Duration::from_secs(1)
        );
        assert!(!requester.take_due(now));
        assert!(requester.take_due(now + Duration::from_secs(1)));

        requester.request_in(now, Duration::from_secs(1));
        requester.cancel_scheduled();
        assert_eq!(
            requester.poll_timeout(now, Duration::from_secs(5)),
            Duration::from_secs(5)
        );
    }
}
