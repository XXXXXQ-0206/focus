//! Shared cancellation wait primitive for Runtime async operations.

use std::time::Duration;

use focus_kernel::CancellationSignal;

pub(crate) async fn wait_for_cancellation(cancellation: &dyn CancellationSignal) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
