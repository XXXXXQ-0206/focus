use std::path::Path;
use std::sync::Arc;

use crate::{
    policy::{ApprovalHandler, FixedApproval},
    runtime::FocusRuntime,
    sandbox,
};
use focus_kernel::{KernelError, ModelEvent, ModelEventStream, ModelProvider, ModelResponse};
use serde_json::json;

/// A deterministic provider useful for integration tests and `doctor` checks.
#[derive(Debug, Clone)]
pub struct StaticProvider {
    response: String,
}

impl StaticProvider {
    /// Build a provider that returns a final text response without tool calls.
    #[must_use]
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for StaticProvider {
    async fn stream(
        &self,
        _request: focus_kernel::ModelRequest,
        _cancellation: &dyn focus_kernel::CancellationSignal,
    ) -> Result<ModelEventStream, KernelError> {
        let response = ModelResponse {
            content: self.response.clone(),
            tool_calls: Vec::new(),
        };
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ModelEvent::RequestStarted {
                provider: "static".into(),
                model: "static".into(),
                endpoint: "static://provider".into(),
            }),
            Ok(ModelEvent::TextDelta {
                text: response.content.clone(),
            }),
            Ok(ModelEvent::Completed { response }),
        ])))
    }
}

/// Default noninteractive policy host used only in tests and diagnostics.
#[must_use]
pub fn deny_approvals() -> Arc<dyn ApprovalHandler> {
    Arc::new(FixedApproval(false))
}

/// Default full-automation policy host for explicitly configured CI flows.
#[must_use]
pub fn approve_all() -> Arc<dyn ApprovalHandler> {
    Arc::new(FixedApproval(true))
}

/// Render a compact machine-readable health snapshot for integrations.
#[must_use]
pub fn doctor(runtime: &FocusRuntime) -> serde_json::Value {
    let capabilities = sandbox::probe_container_capabilities();
    json!({
        "workspace_root": runtime.config.workspace_root,
        "data_root": runtime.config.data_root,
        "context_budget": runtime.config.context_budget,
        "mcp_servers": runtime.mcp_clients.len(),
        "network": runtime.config.network,
        "sandbox_backend": runtime.config.sandbox_backend,
        "container_capabilities": capabilities,
        "kernel": "focus-kernel",
        "runtime": "focus-runtime",
    })
}

/// Return the current workspace root only after opening the Runtime boundary.
pub fn workspace_root(runtime: &FocusRuntime) -> &Path {
    runtime.sandbox.root()
}

#[cfg(test)]
mod tests {
    use focus_kernel::{Message, ModelEvent, ModelRequest, NoCancellation, Role};
    use futures::TryStreamExt;

    use super::StaticProvider;
    use focus_kernel::ModelProvider;

    #[tokio::test]
    async fn static_provider_emits_its_own_normalized_stream() {
        let events = StaticProvider::new("done")
            .stream(
                ModelRequest {
                    messages: vec![Message::text(Role::User, "hello")],
                    tools: Vec::new(),
                },
                &NoCancellation,
            )
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert!(matches!(
            events.first(),
            Some(ModelEvent::RequestStarted { provider, .. }) if provider == "static"
        ));
        assert!(
            matches!(events.last(), Some(ModelEvent::Completed { response }) if response.content == "done")
        );
    }
}
