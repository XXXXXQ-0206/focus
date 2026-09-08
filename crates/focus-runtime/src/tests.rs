mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::mcp::{
        canonical_tool_name,
        test_support::{FakeMcpServer, wait_for_logged_method, wait_for_process_exit},
    };
    use crate::policy::{ApprovalHandler, ToolOperation};
    use crate::session::SessionPhase;
    use focus_kernel::{
        AgentStatus, CancellationSignal, EventKind, KernelError, ModelEvent, ModelEventStream,
        ModelProvider, ModelRequest, ModelResponse, Role, ToolCall,
    };
    use futures::stream;
    use serde_json::json;
    use uuid::Uuid;

    struct SearchThenFinishProvider;

    trait ResponseProvider: Send + Sync {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError>;
    }

    macro_rules! impl_fixture_provider {
        ($provider:ty) => {
            #[async_trait::async_trait]
            impl ModelProvider for $provider {
                async fn stream(
                    &self,
                    request: ModelRequest,
                    cancellation: &dyn CancellationSignal,
                ) -> Result<ModelEventStream, KernelError> {
                    if cancellation.is_cancelled() {
                        return Ok(Box::pin(stream::iter(vec![Ok(ModelEvent::Cancelled)])));
                    }
                    let response = self.response(request)?;
                    let mut events = vec![Ok(ModelEvent::RequestStarted {
                        provider: "fixture".into(),
                        model: "fixture".into(),
                        endpoint: "fixture://response".into(),
                    })];
                    if !response.content.is_empty() {
                        events.push(Ok(ModelEvent::TextDelta {
                            text: response.content.clone(),
                        }));
                    }
                    for call in &response.tool_calls {
                        events.push(Ok(ModelEvent::ToolCallStarted {
                            id: call.id.clone(),
                            name: call.name.clone(),
                        }));
                        events.push(Ok(ModelEvent::ToolCallReady { call: call.clone() }));
                    }
                    events.push(Ok(ModelEvent::Completed { response }));
                    Ok(Box::pin(stream::iter(events)))
                }
            }
        };
    }

    impl ResponseProvider for SearchThenFinishProvider {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError> {
            if request
                .messages
                .iter()
                .any(|message| message.role == Role::Tool)
            {
                return Ok(ModelResponse {
                    content: "Search result reviewed.".into(),
                    tool_calls: Vec::new(),
                });
            }
            Ok(ModelResponse {
                content: "I will inspect the workspace.".into(),
                tool_calls: vec![ToolCall {
                    id: "search-1".into(),
                    name: "search".into(),
                    arguments: json!({"query":"[workspace]"}),
                }],
            })
        }
    }
    impl_fixture_provider!(SearchThenFinishProvider);

    struct WebFetchThenFinishProvider {
        url: String,
    }

    impl ResponseProvider for WebFetchThenFinishProvider {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError> {
            if request
                .messages
                .iter()
                .any(|message| message.role == Role::Tool)
            {
                return Ok(ModelResponse {
                    content: "Network result reviewed.".into(),
                    tool_calls: Vec::new(),
                });
            }
            assert!(
                request.tools.iter().any(|tool| tool.name == "web_fetch"),
                "enabled network must be model-visible"
            );
            Ok(ModelResponse {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "web-fetch-1".into(),
                    name: "web_fetch".into(),
                    arguments: json!({"url": self.url}),
                }],
            })
        }
    }
    impl_fixture_provider!(WebFetchThenFinishProvider);

    struct RuntimeFixture {
        directory: PathBuf,
        config: RuntimeConfig,
    }

    impl RuntimeFixture {
        fn new(name: &str) -> Self {
            let directory = std::env::temp_dir().join(format!("{name}-{}", Uuid::new_v4()));
            std::fs::create_dir_all(&directory).unwrap();
            let mut config = RuntimeConfig::for_workspace(&directory);
            config.data_root = directory.join("state");
            Self { directory, config }
        }

        fn open(&self) -> FocusRuntime {
            FocusRuntime::open(self.config.clone()).unwrap()
        }
    }

    impl Drop for RuntimeFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    #[test]
    fn runtime_fixture_creates_and_removes_its_state_root() {
        let directory = {
            let fixture = RuntimeFixture::new("runtime-fixture");
            assert!(fixture.directory.is_dir());
            assert_eq!(fixture.config.data_root, fixture.directory.join("state"));
            fixture.directory.clone()
        };

        assert!(!directory.exists());
    }

    #[test]
    fn opening_runtime_does_not_start_configured_mcp_without_explicit_enablement() {
        let fixture = RuntimeFixture::new("runtime-mcp-disabled");
        let mut config = fixture.config.clone();
        config.mcp_servers.push(crate::mcp::McpServerConfig::new(
            "untrusted",
            "focus-mcp-command-that-does-not-exist",
        ));

        let runtime = FocusRuntime::open(config).unwrap();

        assert_eq!(crate::diagnostics::doctor(&runtime)["mcp_servers"], 0);
    }

    #[test]
    fn run_persists_and_resumes_a_single_runtime_session() {
        let fixture = RuntimeFixture::new("runtime-e2e-test");
        let runtime = fixture.open();

        let result = runtime
            .run_with_options(
                "test",
                "inspect",
                Arc::new(StaticProvider::new("done")),
                RunOptions::core(deny_approvals()),
            )
            .unwrap();
        let restored = runtime.resume(result.session_id).unwrap();

        assert_eq!(result.final_response, "done");
        assert_eq!(restored.status, AgentStatus::Complete);
        assert!(runtime.replay(result.session_id).unwrap().len() >= 5);
    }

    struct PausingProvider {
        started: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl ModelProvider for PausingProvider {
        async fn stream(
            &self,
            _request: ModelRequest,
            _cancellation: &dyn CancellationSignal,
        ) -> Result<ModelEventStream, KernelError> {
            let started = self.started.clone();
            Ok(Box::pin(stream::once(async move {
                started.store(true, Ordering::Release);
                tokio::time::sleep(Duration::from_millis(250)).await;
                Ok(ModelEvent::Completed {
                    response: ModelResponse {
                        content: "first run".into(),
                        tool_calls: Vec::new(),
                    },
                })
            })))
        }
    }

    #[test]
    fn rejects_a_second_active_run_for_the_same_session() {
        let fixture = RuntimeFixture::new("runtime-session-lease");
        let runtime = fixture.open();
        let session = runtime.create_session("shared session").unwrap();
        let started = Arc::new(AtomicBool::new(false));
        let worker_runtime = runtime.clone();
        let worker_started = started.clone();
        let worker = std::thread::spawn(move || {
            worker_runtime.run_in_session_with_options(
                session.id,
                "long running task",
                Arc::new(PausingProvider {
                    started: worker_started,
                }),
                RunOptions::core(deny_approvals()),
            )
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while !started.load(Ordering::Acquire) {
            assert!(std::time::Instant::now() < deadline, "first run did not start");
            std::thread::yield_now();
        }

        let error = runtime
            .run_in_session_with_options(
                session.id,
                "must not overlap",
                Arc::new(StaticProvider::new("second run")),
                RunOptions::core(deny_approvals()),
            )
            .unwrap_err();

        assert!(error.to_string().contains("already has an active run"));
        assert_eq!(worker.join().unwrap().unwrap().final_response, "first run");
    }

    #[test]
    fn core_turn_omits_the_engineering_workflow_contract() {
        let fixture = RuntimeFixture::new("runtime-core-instruction");
        let runtime = fixture.open();

        let result = runtime
            .run_with_options(
                "core",
                "reply briefly",
                Arc::new(StaticProvider::new("done")),
                RunOptions::core(deny_approvals()),
            )
            .unwrap();
        let transcript = runtime.resume(result.session_id).unwrap().transcript;
        let instruction = transcript
            .iter()
            .find(|message| message.role == Role::System)
            .expect("a system instruction must be persisted");

        assert!(!instruction.content.contains("workflow_checkpoint"));
    }

    struct CoreCapabilityProbe;

    impl ResponseProvider for CoreCapabilityProbe {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError> {
            assert!(!request.tools.iter().any(|tool| tool.name == "workflow_checkpoint"));
            assert!(!request.tools.iter().any(|tool| tool.name == "delegate"));
            Ok(ModelResponse {
                content: "core complete".into(),
                tool_calls: Vec::new(),
            })
        }
    }
    impl_fixture_provider!(CoreCapabilityProbe);

    #[test]
    fn core_turn_excludes_workflow_and_delegation_tools() {
        let fixture = RuntimeFixture::new("runtime-core-capabilities");
        let runtime = fixture.open();

        let result = runtime
            .run_with_options(
                "core",
                "answer directly",
                Arc::new(CoreCapabilityProbe),
                RunOptions::core(deny_approvals()),
            )
            .unwrap();

        assert_eq!(result.final_response, "core complete");
    }

    #[test]
    fn preparation_error_marks_the_session_failed() {
        let fixture = RuntimeFixture::new("runtime-preparation-failure");
        let config = fixture.config.clone();
        let runtime = fixture.open();
        let session = runtime.create_session("broken event stream").unwrap();
        let event_path = config
            .data_root
            .join("events")
            .join(format!("{}.jsonl", session.id));
        std::fs::write(event_path, "not-json\nstill-not-json\n").unwrap();

        let result = runtime.run_in_session_with_options(
            session.id,
            "resume corrupt state",
            Arc::new(StaticProvider::new("unused")),
            RunOptions::core(deny_approvals()),
        );

        assert!(matches!(
            result,
            Err(RuntimeError::Kernel(KernelError::Store(_)))
        ));
        assert_eq!(
            runtime
                .list_sessions()
                .unwrap()
                .into_iter()
                .find(|metadata| metadata.id == session.id)
            .unwrap()
            .phase,
            SessionPhase::Failed
        );
    }

    #[test]
    fn model_tool_calls_share_runtime_policy_and_event_persistence() {
        let fixture = RuntimeFixture::new("runtime-tool-loop-test");
        std::fs::write(fixture.directory.join("marker.txt"), "[workspace]").unwrap();
        let runtime = fixture.open();

        let result = runtime
            .run_with_options(
                "tool loop",
                "find marker",
                Arc::new(SearchThenFinishProvider),
                RunOptions::core(deny_approvals()),
            )
            .unwrap();
        let events = runtime.replay(result.session_id).unwrap();

        assert_eq!(result.final_response, "Search result reviewed.");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.kind, EventKind::ToolResultReceived { .. }))
        );
    }

    #[test]
    fn web_fetch_shares_runtime_policy_events_and_replay() {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            thread,
        };

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 7\r\nConnection: close\r\n\r\nnetwork",
                )
                .unwrap();
        });
        let fixture = RuntimeFixture::new("runtime-web-fetch-e2e");
        let mut config = fixture.config.clone();
        config.network = crate::network::NetworkConfig::enabled();
        config.network.allow_local = true;
        config.network.allow_port(address.port());
        config.policy.network = policy::PolicyDecision::RequireApproval;
        let runtime = FocusRuntime::open(config).unwrap();
        let approval = Arc::new(RecordingApproval::default());

        let result = runtime
            .run_with_options(
                "web fetch loop",
                "fetch fixture",
                Arc::new(WebFetchThenFinishProvider {
                    url: format!("http://{address}/fixture"),
                }),
                RunOptions::core(approval.clone()),
            )
            .unwrap();
        server.join().unwrap();
        let events = runtime.replay(result.session_id).unwrap();

        assert_eq!(result.final_response, "Network result reviewed.");
        assert!(approval.requests.lock().unwrap().iter().any(|request| {
            request.tool == "web_fetch" && request.operation == ToolOperation::Network
        }));
        assert!(events.iter().any(|event| matches!(
            &event.kind,
            EventKind::ToolCallRequested { call } if call.name == "web_fetch"
        )));
        assert!(events.iter().any(|event| matches!(
            &event.kind,
            EventKind::ToolResultReceived { result }
                if result.name == "web_fetch" && !result.is_error && result.content.contains("network")
        )));
    }

    struct McpThenFinishProvider;

    impl ResponseProvider for McpThenFinishProvider {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError> {
            if request
                .messages
                .iter()
                .any(|message| message.role == Role::Tool)
            {
                return Ok(ModelResponse {
                    content: "MCP result reviewed.".into(),
                    tool_calls: Vec::new(),
                });
            }
            let tool = request
                .tools
                .iter()
                .find(|tool| tool.description == "echo")
                .expect("MCP tool must share the model-visible registry");
            Ok(ModelResponse {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "mcp-1".into(),
                    name: tool.name.clone(),
                    arguments: json!({"value": 7}),
                }],
            })
        }
    }
    impl_fixture_provider!(McpThenFinishProvider);

    #[derive(Default)]
    struct RecordingApproval {
        requests: Mutex<Vec<policy::ApprovalRequest>>,
    }

    impl ApprovalHandler for RecordingApproval {
        fn approve(&self, request: &policy::ApprovalRequest) -> bool {
            self.requests.lock().unwrap().push(request.clone());
            true
        }
    }

    #[test]
    fn mcp_tools_share_policy_agent_loop_and_persisted_events() {
        let runtime_fixture = RuntimeFixture::new("runtime-mcp-e2e");
        let mcp_fixture = FakeMcpServer::new("ok");
        let mut config = runtime_fixture.config.clone();
        config.policy.network = policy::PolicyDecision::RequireApproval;
        config.enable_mcp = true;
        config
            .mcp_servers
            .push(mcp_fixture.config().with_operation(ToolOperation::Network));
        let runtime = FocusRuntime::open(config).unwrap();
        let approval = Arc::new(RecordingApproval::default());

        let result = runtime
            .run_with_options(
                "MCP tool loop",
                "call remote tool",
                Arc::new(McpThenFinishProvider),
                RunOptions::core(approval.clone()),
            )
            .unwrap();
        let events = runtime.replay(result.session_id).unwrap();
        let canonical = canonical_tool_name("Fixture Server", "Echo Tool");

        assert_eq!(result.final_response, "MCP result reviewed.");
        assert_eq!(approval.requests.lock().unwrap().len(), 1);
        assert_eq!(
            approval.requests.lock().unwrap()[0].operation,
            ToolOperation::Network
        );
        assert!(mcp_fixture.methods().contains(&"tools/call".to_owned()));
        assert!(events.iter().any(|event| matches!(
            &event.kind,
            EventKind::ToolCallRequested { call } if call.name == canonical
        )));
        assert!(events.iter().any(|event| matches!(
            &event.kind,
            EventKind::ToolResultReceived { result } if result.name == canonical && result.content == "tool-result" && !result.is_error
        )));
    }

    #[test]
    fn mcp_tools_default_to_other_and_require_approval_in_the_agent_loop() {
        let runtime_fixture = RuntimeFixture::new("runtime-mcp-other-e2e");
        let mcp_fixture = FakeMcpServer::new("ok");
        let mut config = runtime_fixture.config.clone();
        config.enable_mcp = true;
        config.mcp_servers.push(mcp_fixture.config());
        let runtime = FocusRuntime::open(config).unwrap();
        let approval = Arc::new(RecordingApproval::default());

        let result = runtime
            .run_with_options(
                "MCP default policy",
                "call remote tool",
                Arc::new(McpThenFinishProvider),
                RunOptions::core(approval.clone()),
            )
            .unwrap();

        assert_eq!(result.final_response, "MCP result reviewed.");
        assert_eq!(approval.requests.lock().unwrap().len(), 1);
        assert_eq!(
            approval.requests.lock().unwrap()[0].operation,
            ToolOperation::Other
        );
        assert!(mcp_fixture.methods().contains(&"tools/call".to_owned()));
    }

    #[test]
    fn cancelling_an_in_flight_mcp_tool_reaps_the_server_and_returns_kernel_cancelled() {
        let runtime_fixture = RuntimeFixture::new("runtime-mcp-cancel-e2e");
        let mcp_fixture = FakeMcpServer::new("slow_call");
        let mut config = runtime_fixture.config.clone();
        config.enable_mcp = true;
        config.mcp_servers.push(mcp_fixture.config());
        let runtime = FocusRuntime::open(config).unwrap();
        let process_id = mcp_fixture.process_id();
        let cancellation = subagent::CancellationToken::default();
        let cancellation_trigger = cancellation.clone();
        let method_log = mcp_fixture.log.clone();
        let trigger = std::thread::spawn(move || {
            assert!(wait_for_logged_method(
                &method_log,
                "tools/call",
                std::time::Duration::from_secs(1)
            ));
            cancellation_trigger.cancel();
        });
        let mut options = RunOptions::core(approve_all());
        options.cancellation = Arc::new(cancellation);
        let started = std::time::Instant::now();

        let error = runtime
            .run_with_options(
                "MCP cancellation",
                "call slow remote tool",
                Arc::new(McpThenFinishProvider),
                options,
            )
            .unwrap_err();
        trigger.join().unwrap();

        assert!(matches!(error, RuntimeError::Cancelled));
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
        assert!(wait_for_process_exit(
            process_id,
            std::time::Duration::from_secs(1)
        ));
    }

    struct PrematureProvider(AtomicUsize);

    impl ResponseProvider for PrematureProvider {
        fn response(&self, _request: ModelRequest) -> Result<ModelResponse, KernelError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(ModelResponse {
                content: "done without evidence".into(),
                tool_calls: Vec::new(),
            })
        }
    }
    impl_fixture_provider!(PrematureProvider);

    #[test]
    fn workflow_gate_retries_then_rejects_premature_completion() {
        let fixture = RuntimeFixture::new("runtime-workflow-gate");
        let mut config = fixture.config.clone();
        config.max_workflow_gate_retries = 1;
        let runtime = FocusRuntime::open(config).unwrap();
        let provider = Arc::new(PrematureProvider(AtomicUsize::new(0)));

        let error = runtime
            .run_with_options(
                "workflow gate",
                "finish early",
                provider.clone(),
                RunOptions::engineering(approve_all()),
            )
            .unwrap_err();

        assert!(matches!(error, RuntimeError::Workflow(_)));
        assert_eq!(provider.0.load(Ordering::SeqCst), 2);
        let session = runtime.list_sessions().unwrap().remove(0);
        let events = runtime.replay(session.id).unwrap();
        assert!(events.iter().any(|event| matches!(
            &event.kind,
            EventKind::Runtime { name, .. } if name == "workflow_gate_failed"
        )));
    }

    struct CompleteWorkflowProvider;

    impl ResponseProvider for CompleteWorkflowProvider {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError> {
            let completed = request
                .messages
                .iter()
                .filter(|message| message.role == Role::Tool)
                .filter_map(|message| message.tool_name.as_deref())
                .collect::<Vec<_>>();
            let call = if !completed.contains(&"search") {
                ToolCall {
                    id: "explore".into(),
                    name: "search".into(),
                    arguments: json!({"query":"workflow-marker"}),
                }
            } else if completed
                .iter()
                .filter(|name| **name == "workflow_checkpoint")
                .count()
                == 0
            {
                ToolCall {
                    id: "plan".into(),
                    name: "workflow_checkpoint".into(),
                    arguments: json!({"kind":"plan","summary":"inspect and verify"}),
                }
            } else if completed
                .iter()
                .filter(|name| **name == "workflow_checkpoint")
                .count()
                == 1
            {
                ToolCall {
                    id: "no-change".into(),
                    name: "workflow_checkpoint".into(),
                    arguments: json!({"kind":"no_change","summary":"fixture requires no edit"}),
                }
            } else if !completed.contains(&"shell") {
                ToolCall {
                    id: "verify".into(),
                    name: "shell".into(),
                    arguments: json!({
                        "command": if cfg!(windows) { "exit /b 0" } else { "true" },
                        "purpose": "verify"
                    }),
                }
            } else if completed
                .iter()
                .filter(|name| **name == "workflow_checkpoint")
                .count()
                == 2
            {
                ToolCall {
                    id: "review".into(),
                    name: "workflow_checkpoint".into(),
                    arguments: json!({"kind":"review","summary":"review complete"}),
                }
            } else {
                return Ok(ModelResponse {
                    content: "workflow complete".into(),
                    tool_calls: Vec::new(),
                });
            };
            Ok(ModelResponse {
                content: String::new(),
                tool_calls: vec![call],
            })
        }
    }
    impl_fixture_provider!(CompleteWorkflowProvider);

    #[test]
    fn executable_workflow_records_ordered_stages_and_passes_real_verification() {
        let fixture = RuntimeFixture::new("runtime-workflow-pass");
        std::fs::write(fixture.directory.join("marker.txt"), "workflow-marker").unwrap();
        let runtime = fixture.open();

        let result = runtime
            .run_with_options(
                "workflow pass",
                "complete the fixture workflow",
                Arc::new(CompleteWorkflowProvider),
                RunOptions::engineering(approve_all()),
            )
            .unwrap();
        let stages = runtime
            .replay(result.session_id)
            .unwrap()
            .into_iter()
            .filter_map(|event| match event.kind {
                EventKind::Runtime { name, data } if name == "workflow_stage_entered" => {
                    serde_json::from_value::<workflow::WorkflowStage>(data["stage"].clone()).ok()
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            stages,
            vec![
                workflow::WorkflowStage::Explore,
                workflow::WorkflowStage::Plan,
                workflow::WorkflowStage::Implement,
                workflow::WorkflowStage::Verify,
                workflow::WorkflowStage::Review,
                workflow::WorkflowStage::Complete,
            ]
        );
        assert_eq!(result.final_response, "workflow complete");
    }

    struct LongShellProvider;

    impl ResponseProvider for LongShellProvider {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError> {
            if request
                .messages
                .iter()
                .any(|message| message.role == Role::Tool)
            {
                return Ok(ModelResponse {
                    content: "unexpected continuation".into(),
                    tool_calls: Vec::new(),
                });
            }
            Ok(ModelResponse {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "long-shell".into(),
                    name: "shell".into(),
                    arguments: json!({
                        "command": if cfg!(windows) {
                            "powershell -NoProfile -Command \"Start-Sleep -Seconds 5\""
                        } else {
                            "sleep 5"
                        }
                    }),
                }],
            })
        }
    }
    impl_fixture_provider!(LongShellProvider);

    #[test]
    fn runtime_cancellation_reaches_an_in_flight_shell_tool() {
        let fixture = RuntimeFixture::new("runtime-shell-cancel");
        let runtime = fixture.open();
        let cancellation = subagent::CancellationToken::default();
        let trigger = cancellation.clone();
        let worker = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            trigger.cancel();
        });
        let options = RunOptions {
            cancellation: Arc::new(cancellation),
            ..RunOptions::core(approve_all())
        };
        let started = std::time::Instant::now();

        let error = runtime
            .run_with_options(
                "cancel shell",
                "run a long shell",
                Arc::new(LongShellProvider),
                options,
            )
            .unwrap_err();
        worker.join().unwrap();

        assert!(matches!(error, RuntimeError::Cancelled));
        assert!(started.elapsed() < std::time::Duration::from_secs(3));
    }

    struct DelegatingProvider;

    impl ResponseProvider for DelegatingProvider {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError> {
            if request.messages.iter().any(|message| {
                message.role == Role::User && message.content.contains("Act as the `")
            }) {
                return Ok(ModelResponse {
                    content: "child evidence".into(),
                    tool_calls: Vec::new(),
                });
            }
            if request.messages.iter().any(|message| {
                message.role == Role::Tool && message.tool_name.as_deref() == Some("delegate")
            }) {
                return Ok(ModelResponse {
                    content: "parent consumed child results".into(),
                    tool_calls: Vec::new(),
                });
            }
            assert!(request.tools.iter().any(|tool| tool.name == "delegate"));
            Ok(ModelResponse {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "delegate-1".into(),
                    name: "delegate".into(),
                    arguments: json!({
                        "mode":"parallel",
                        "tasks":[
                            {"role":"explorer","objective":"inspect alpha","focus_paths":["src"]},
                            {"role":"reviewer","objective":"review beta","focus_paths":[]}
                        ]
                    }),
                }],
            })
        }
    }
    impl_fixture_provider!(DelegatingProvider);

    #[test]
    fn delegate_tool_runs_isolated_children_and_injects_ordered_results() {
        let fixture = RuntimeFixture::new("runtime-delegate-e2e");
        let runtime = fixture.open();

        let result = runtime
            .run_with_options(
                "delegate",
                "use specialists",
                Arc::new(DelegatingProvider),
                RunOptions::core(approve_all())
                    .with_delegation(subagent::SubagentLimits::default()),
            )
            .unwrap();
        let state = runtime.resume(result.session_id).unwrap();
        let delegate_result = state
            .transcript
            .iter()
            .find(|message| {
                message.role == Role::Tool && message.tool_name.as_deref() == Some("delegate")
            })
            .unwrap();
        let aggregated: serde_json::Value = serde_json::from_str(&delegate_result.content).unwrap();
        let child_ids = aggregated["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| Uuid::parse_str(item["child_session_id"].as_str().unwrap()).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(result.final_response, "parent consumed child results");
        assert_eq!(aggregated["results"][0]["role"], "explorer");
        assert_eq!(aggregated["results"][1]["role"], "reviewer");
        for child_id in child_ids {
            let child = runtime
                .list_sessions()
                .unwrap()
                .into_iter()
                .find(|session| session.id == child_id)
                .unwrap();
            assert_eq!(child.parent_id, Some(result.session_id));
            assert_eq!(
                child.transcript_inheritance,
                session::TranscriptInheritance::None
            );
            assert!(
                !runtime
                    .resume(child_id)
                    .unwrap()
                    .transcript
                    .iter()
                    .any(|message| message.content == "use specialists")
            );
        }
        let events = runtime.replay(result.session_id).unwrap();
        assert!(events.iter().any(|event| matches!(
            &event.kind,
            EventKind::Runtime { name, .. } if name == "subagent_batch_completed"
        )));
    }

    struct ProductionDelegatingProvider;

    impl ResponseProvider for ProductionDelegatingProvider {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError> {
            if request.messages.iter().any(|message| {
                message.role == Role::User && message.content.contains("Act as the `")
            }) {
                assert!(
                    request
                        .messages
                        .iter()
                        .any(|message| { message.content.contains("parent-observation") })
                );
                return Ok(ModelResponse {
                    content: "child inspected inherited parent evidence".into(),
                    tool_calls: Vec::new(),
                });
            }
            let has_tool = |name: &str| {
                request.messages.iter().any(|message| {
                    message.role == Role::Tool && message.tool_name.as_deref() == Some(name)
                })
            };
            let has_checkpoint = |kind: &str| {
                request.messages.iter().any(|message| {
                    message.role == Role::Assistant
                        && message.tool_calls.iter().any(|call| {
                            call.name == "workflow_checkpoint" && call.arguments["kind"] == kind
                        })
                })
            };
            if !has_tool("search") {
                return Ok(ModelResponse {
                    content: String::new(),
                    tool_calls: vec![ToolCall {
                        id: "parent-search".into(),
                        name: "search".into(),
                        arguments: json!({"query":"parent-observation"}),
                    }],
                });
            }
            if !has_tool("delegate") {
                return Ok(ModelResponse {
                    content: String::new(),
                    tool_calls: vec![
                        ToolCall {
                            id: "parent-plan".into(),
                            name: "workflow_checkpoint".into(),
                            arguments: json!({"kind":"plan","summary":"delegate focused inspection"}),
                        },
                        ToolCall {
                            id: "parent-delegate".into(),
                            name: "delegate".into(),
                            arguments: json!({"tasks":[{"role":"explorer","objective":"inspect inherited evidence","focus_paths":["marker.txt"]}]}),
                        },
                    ],
                });
            }
            if !has_checkpoint("review") {
                return Ok(ModelResponse {
                    content: String::new(),
                    tool_calls: vec![
                        ToolCall {
                            id: "parent-no-change".into(),
                            name: "workflow_checkpoint".into(),
                            arguments: json!({"kind":"no_change","summary":"inspection-only task"}),
                        },
                        ToolCall {
                            id: "parent-verify".into(),
                            name: "shell".into(),
                            arguments: json!({"command": if cfg!(windows) { "cmd /C exit 0" } else { "true" }, "purpose":"verify"}),
                        },
                        ToolCall {
                            id: "parent-review".into(),
                            name: "workflow_checkpoint".into(),
                            arguments: json!({"kind":"review","summary":"child evidence consumed"}),
                        },
                    ],
                });
            }
            Ok(ModelResponse {
                content: "production delegation complete".into(),
                tool_calls: Vec::new(),
            })
        }
    }
    impl_fixture_provider!(ProductionDelegatingProvider);

    #[test]
    fn production_delegate_inherits_bounded_parent_transcript() {
        let fixture = RuntimeFixture::new("runtime-production-delegate");
        std::fs::write(fixture.directory.join("marker.txt"), "parent-observation").unwrap();
        let runtime = fixture.open();

        let result = runtime
            .run_with_options(
                "production delegate",
                "inspect with a specialist",
                Arc::new(ProductionDelegatingProvider),
                RunOptions::engineering(approve_all())
                    .with_delegation(subagent::SubagentLimits::default()),
            )
            .unwrap();

        assert_eq!(result.final_response, "production delegation complete");
        let events = runtime.replay(result.session_id).unwrap();
        assert!(events.iter().any(|event| matches!(
            &event.kind,
            EventKind::Runtime { name, .. } if name == "subagent_completed"
        )));
        assert!(events.iter().any(|event| matches!(
            &event.kind,
            EventKind::Runtime { name, .. } if name == "workflow_completed"
        )));
    }

    struct SearchThenFailProvider;

    impl ResponseProvider for SearchThenFailProvider {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError> {
            if request
                .messages
                .iter()
                .any(|message| message.role == Role::Tool)
            {
                return Err(KernelError::Model("interrupted after search".into()));
            }
            Ok(ModelResponse {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "recovery-search".into(),
                    name: "search".into(),
                    arguments: json!({"query":"recovery-marker"}),
                }],
            })
        }
    }
    impl_fixture_provider!(SearchThenFailProvider);

    struct RecoveryCompleter;

    impl ResponseProvider for RecoveryCompleter {
        fn response(&self, request: ModelRequest) -> Result<ModelResponse, KernelError> {
            let has_checkpoint = |kind: &str| {
                request.messages.iter().any(|message| {
                    message.role == Role::Assistant
                        && message.tool_calls.iter().any(|call| {
                            call.name == "workflow_checkpoint" && call.arguments["kind"] == kind
                        })
                })
            };
            if !has_checkpoint("review") {
                return Ok(ModelResponse {
                    content: String::new(),
                    tool_calls: vec![
                        ToolCall {
                            id: "recovery-plan".into(),
                            name: "workflow_checkpoint".into(),
                            arguments: json!({"kind":"plan","summary":"resume from persisted search"}),
                        },
                        ToolCall {
                            id: "recovery-no-change".into(),
                            name: "workflow_checkpoint".into(),
                            arguments: json!({"kind":"no_change","summary":"no edit required"}),
                        },
                        ToolCall {
                            id: "recovery-verify".into(),
                            name: "shell".into(),
                            arguments: json!({"command": if cfg!(windows) { "cmd /C exit 0" } else { "true" }, "purpose":"verify"}),
                        },
                        ToolCall {
                            id: "recovery-review".into(),
                            name: "workflow_checkpoint".into(),
                            arguments: json!({"kind":"review","summary":"recovered evidence reviewed"}),
                        },
                    ],
                });
            }
            Ok(ModelResponse {
                content: "recovered workflow complete".into(),
                tool_calls: Vec::new(),
            })
        }
    }
    impl_fixture_provider!(RecoveryCompleter);

    #[test]
    fn runtime_recovery_reconciles_persisted_tool_evidence() {
        let fixture = RuntimeFixture::new("runtime-workflow-recovery");
        std::fs::write(fixture.directory.join("marker.txt"), "recovery-marker").unwrap();
        let runtime = fixture.open();
        let session = runtime.create_session("recovery").unwrap();

        assert!(
            runtime
                .run_in_session_with_options(
                    session.id,
                    "inspect before interruption",
                    Arc::new(SearchThenFailProvider),
                    RunOptions::engineering(approve_all()),
                )
                .is_err()
        );
        let interrupted = runtime.replay(session.id).unwrap();
        let started_run = interrupted
            .iter()
            .find_map(|event| match &event.kind {
                EventKind::Runtime { name, data } if name == "workflow_started" => {
                    data["run_id"].as_str().map(str::to_owned)
                }
                _ => None,
            })
            .unwrap();
        assert!(interrupted.iter().any(|event| matches!(
            &event.kind,
            EventKind::Runtime { name, data }
                if name == "workflow_evidence_recorded"
                    && data["evidence"]["kind"] == "repository_inspected"
                    && data["evidence"]["tool"] == "search"
        )));

        let result = runtime
            .run_in_session_with_options(
                session.id,
                "resume interrupted workflow",
                Arc::new(RecoveryCompleter),
                RunOptions::engineering(approve_all()),
            )
            .unwrap();
        let replayed = runtime.replay(session.id).unwrap();

        assert_eq!(result.final_response, "recovered workflow complete");
        assert_eq!(
            replayed
                .iter()
                .filter(|event| matches!(
                    &event.kind,
                    EventKind::Runtime { name, .. } if name == "workflow_started"
                ))
                .count(),
            1
        );
        assert!(replayed.iter().any(|event| matches!(
            &event.kind,
            EventKind::Runtime { name, data }
                if name == "workflow_recovered" && data["run_id"] == started_run
        )));
    }

    #[test]
    fn openai_mcp_policy_workflow_and_subagent_share_one_runtime_event_model() {
        let runtime_fixture = RuntimeFixture::new("runtime-combined-e2e");
        std::fs::write(runtime_fixture.directory.join("marker.txt"), "combined-marker").unwrap();
        let mcp_fixture = FakeMcpServer::new("ok");
        let canonical_mcp = canonical_tool_name("Fixture Server", "Echo Tool");
        let (base_url, request_count, server) = openai_combined_fixture(canonical_mcp.clone());
        let provider = Arc::new(
            provider::OpenAiCompatibleProvider::new(provider::OpenAiConfig::new(
                base_url,
                "fixture-model",
            ))
            .unwrap(),
        );
        let mut config = runtime_fixture.config.clone();
        config.policy.network = policy::PolicyDecision::RequireApproval;
        config.enable_mcp = true;
        config
            .mcp_servers
            .push(mcp_fixture.config().with_operation(ToolOperation::Network));
        let runtime = FocusRuntime::open(config).unwrap();
        let approval = Arc::new(RecordingApproval::default());

        let result = runtime
            .run_with_options(
                "combined acceptance",
                "exercise the complete runtime",
                provider,
                RunOptions::engineering(approval.clone())
                    .with_delegation(subagent::SubagentLimits::default()),
            )
            .unwrap();
        server.join().unwrap();
        let events = runtime.replay(result.session_id).unwrap();

        assert_eq!(result.final_response, "combined runtime complete");
        assert_eq!(request_count.load(Ordering::SeqCst), 6);
        assert!(mcp_fixture.methods().contains(&"tools/call".to_owned()));
        assert!(
            approval
                .requests
                .lock()
                .unwrap()
                .iter()
                .any(|request| request.tool == canonical_mcp
                    && request.operation == ToolOperation::Network)
        );
        for event_name in [
            "workflow_started",
            "subagent_queued",
            "subagent_started",
            "subagent_completed",
            "subagent_batch_completed",
            "workflow_gate_passed",
            "workflow_completed",
        ] {
            assert!(events.iter().any(|event| matches!(
                &event.kind,
                EventKind::Runtime { name, .. } if name == event_name
            )));
        }
        assert!(events.iter().any(|event| matches!(
            &event.kind,
            EventKind::ToolResultReceived { result }
                if result.name == canonical_mcp && result.content == "tool-result"
        )));
        assert!(runtime.list_sessions().unwrap().iter().any(|session| {
            session.parent_id == Some(result.session_id)
                && session.transcript_inheritance == session::TranscriptInheritance::None
        }));
    }

    fn openai_combined_fixture(
        canonical_mcp: String,
    ) -> (String, Arc<AtomicUsize>, std::thread::JoinHandle<()>) {
        use std::io::Write;
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let server_count = request_count.clone();
        let server = std::thread::spawn(move || {
            for _ in 0..6 {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_http_json_request(&mut stream);
                server_count.fetch_add(1, Ordering::SeqCst);
                let messages = request["messages"].as_array().unwrap();
                let is_child = messages.iter().any(|message| {
                    message["content"]
                        .as_str()
                        .is_some_and(|content| content.contains("Act as the `"))
                });
                let has_tool = |name: &str| {
                    messages
                        .iter()
                        .any(|message| message["role"] == "tool" && message["name"] == name)
                };
                let has_review = messages.iter().any(|message| {
                    message["role"] == "tool"
                        && message["content"]
                            .as_str()
                            .is_some_and(|content| content.contains("checkpoint `review`"))
                });
                let response = if is_child {
                    openai_text_response("child inspected combined context")
                } else if has_review {
                    openai_text_response("combined runtime complete")
                } else if has_tool("delegate") {
                    openai_tool_response(vec![
                        (
                            "combined-no-change",
                            "workflow_checkpoint",
                            json!({"kind":"no_change","summary":"fixture requires no edit"}),
                        ),
                        (
                            "combined-verify",
                            "shell",
                            json!({"command": if cfg!(windows) { "cmd /C exit 0" } else { "true" }, "purpose":"verify"}),
                        ),
                        (
                            "combined-review",
                            "workflow_checkpoint",
                            json!({"kind":"review","summary":"combined evidence reviewed"}),
                        ),
                    ])
                } else if has_tool(&canonical_mcp) {
                    openai_tool_response(vec![
                        (
                            "combined-plan",
                            "workflow_checkpoint",
                            json!({"kind":"plan","summary":"exercise MCP and delegation"}),
                        ),
                        (
                            "combined-delegate",
                            "delegate",
                            json!({"tasks":[{"role":"reviewer","objective":"review MCP evidence"}]}),
                        ),
                    ])
                } else if has_tool("search") {
                    openai_tool_response(vec![(
                        "combined-mcp",
                        canonical_mcp.as_str(),
                        json!({"value":7}),
                    )])
                } else {
                    openai_tool_response(vec![(
                        "combined-search",
                        "search",
                        json!({"query":"combined-marker"}),
                    )])
                };
                let body = response.to_string();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .unwrap();
            }
        });
        (format!("http://{address}/v1"), request_count, server)
    }

    fn read_http_json_request(stream: &mut impl std::io::Read) -> serde_json::Value {
        let mut bytes = Vec::new();
        let mut chunk = [0_u8; 4096];
        let header_end = loop {
            let read = stream.read(&mut chunk).unwrap();
            assert!(read > 0);
            bytes.extend_from_slice(&chunk[..read]);
            if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                break index + 4;
            }
        };
        let headers = String::from_utf8_lossy(&bytes[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap_or_default();
        while bytes.len() < header_end + content_length {
            let read = stream.read(&mut chunk).unwrap();
            assert!(read > 0);
            bytes.extend_from_slice(&chunk[..read]);
        }
        serde_json::from_slice(&bytes[header_end..header_end + content_length]).unwrap()
    }

    fn openai_text_response(content: &str) -> serde_json::Value {
        json!({"choices":[{"message":{"role":"assistant","content":content}}]})
    }

    fn openai_tool_response(calls: Vec<(&str, &str, serde_json::Value)>) -> serde_json::Value {
        json!({"choices":[{"message":{
            "role":"assistant",
            "content":null,
            "tool_calls":calls.into_iter().map(|(id, name, arguments)| json!({
                "id":id,
                "type":"function",
                "function":{"name":name,"arguments":arguments}
            })).collect::<Vec<_>>()
        }}]})
    }

    #[test]
    fn live_subscribers_observe_the_same_persisted_event_order() {
        let fixture = RuntimeFixture::new("runtime-event-hub");
        let runtime = fixture.open();
        let receiver = runtime.subscribe();

        let result = runtime
            .run_with_options(
                "events",
                "emit events",
                Arc::new(StaticProvider::new("done")),
                RunOptions::core(deny_approvals()),
            )
            .unwrap();
        let live = receiver.try_iter().collect::<Vec<_>>();
        let replayed = runtime.replay(result.session_id).unwrap();

        assert_eq!(
            live.iter().map(|event| event.id).collect::<Vec<_>>(),
            replayed.iter().map(|event| event.id).collect::<Vec<_>>()
        );
    }
}
