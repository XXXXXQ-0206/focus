//! Thin CLI adapter over `focus-runtime`; it contains no agent logic.

mod cli_args;
mod supervisor;
mod tui;

use std::{
    collections::BTreeMap,
    env,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use focus_kernel::{Event, EventKind, ModelEvent, ModelProvider, redact_sensitive_text};
use focus_runtime::{
    FocusRuntime, RunOptions, RunResult, RuntimeConfig, StaticProvider, approve_all,
    benchmark::run_deterministic,
    deny_approvals, doctor,
    goal::GoalPhase,
    host::{HostFrameV1, HostRequestV1},
    interaction::{ApprovalBroker, ApprovalEnvelope, ApprovalInbox},
    mcp::McpServerConfig,
    memory::MemoryScope,
    network::{DomainAccess, NetworkConfig},
    policy::{ApprovalDecision, PolicyDecision, ToolOperation},
    provider::{CommandModelProvider, OpenAiCompatibleProvider, OpenAiConfig, OpenAiWireApi},
    sandbox::SandboxBackend,
    subagent::{CancellationToken, SubagentLimits},
    update::{UpdateManifest, VersionedArtifactStore, default_update_root},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::cli_args::Arguments;

pub(crate) const REVIEW_TASK: &str = "Review the current Focus repository as a senior engineer. Inspect the worktree and recent changes, identify concrete correctness, reliability, security, or test gaps, implement the smallest justified fixes when a real issue exists, run focused verification, and report observed evidence. Do not stop at listing findings; preserve existing functionality, performance, and ownership boundaries.";
pub(crate) const SIMPLIFY_TASK: &str = "Simplify the current Focus repository as a senior engineer. Inspect the affected code paths, preserve functionality, behavior, and performance, make only measured low-risk simplifications, add or update focused tests, run verification, and report before-and-after evidence. Do not rewrite distinct state machines without demonstrated equivalence or benefit.";

/// Runs the Focus command-line interface and returns its process exit status.
pub fn entrypoint() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if error == supervisor::RESTART_REQUESTED => {
            ExitCode::from(supervisor::RESTART_EXIT_CODE as u8)
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Run the primary `focus` entrypoint under a stable process supervisor.
pub fn supervised_entrypoint() -> ExitCode {
    if env::var_os(supervisor::APP_CHILD_ENV).is_some() {
        entrypoint()
    } else {
        match supervisor::supervise(env::args().skip(1).collect()) {
            Ok(status) => status,
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

fn run(arguments: Vec<String>) -> Result<(), String> {
    let (command, arguments) = normalize_command(arguments);
    let parser = Arguments::new(arguments);
    match command.as_str() {
        "run" => run_agent(parser),
        "chat" => run_chat(parser),
        "demo" => run_demo(parser),
        "host" => run_host(parser),
        "doctor" => run_doctor(parser),
        "benchmark" => run_benchmark(parser),
        "self-review" => run_self_review(parser),
        "self-update" => run_self_update(parser),
        "session" => run_session(parser),
        "memory" => run_memory(parser),
        "subagent" => run_subagent(parser),
        "goal" => run_goal(parser),
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        other => Err(format!("unknown command `{other}`\n{}", usage())),
    }
}

fn normalize_command(mut arguments: Vec<String>) -> (String, Vec<String>) {
    match arguments.first() {
        None => ("chat".into(), arguments),
        Some(argument) if argument == "--help" => (arguments.remove(0), arguments),
        Some(argument) if argument.starts_with("--") => ("chat".into(), arguments),
        Some(_) => (arguments.remove(0), arguments),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BenchmarkOptions {
    runs: usize,
    output: Option<PathBuf>,
}

impl BenchmarkOptions {
    fn parse(parser: &mut Arguments) -> Result<Self, String> {
        let runs = parser
            .take_option("--runs")?
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --runs: {error}"))
            })
            .transpose()?
            .unwrap_or(3);
        if runs == 0 || runs > 100 {
            return Err("--runs must be between 1 and 100".into());
        }
        Ok(Self {
            runs,
            output: parser.take_option("--output")?.map(PathBuf::from),
        })
    }
}

fn run_benchmark(mut parser: Arguments) -> Result<(), String> {
    let options = BenchmarkOptions::parse(&mut parser)?;
    parser.ensure_empty()?;
    let receipt = run_deterministic(options.runs).map_err(|error| error.to_string())?;
    let encoded = serde_json::to_string_pretty(&receipt).map_err(|error| error.to_string())?;
    if let Some(path) = options.output {
        write_receipt(&path, &encoded)?;
        let readback = std::fs::read_to_string(&path).map_err(|error| {
            format!(
                "failed to read benchmark receipt {}: {error}",
                path.display()
            )
        })?;
        serde_json::from_str::<focus_runtime::benchmark::BenchmarkReceipt>(&readback)
            .map_err(|error| format!("invalid benchmark receipt {}: {error}", path.display()))?;
    }
    println!("{encoded}");
    if !benchmark_passed(&receipt) {
        return Err("one or more benchmark modes failed; inspect the JSON receipt".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReviewCheck {
    name: String,
    command: String,
    passed: bool,
    output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SelfReviewReceipt {
    schema_version: u32,
    workspace: String,
    commit: Option<String>,
    checks: Vec<ReviewCheck>,
    artifact_path: Option<String>,
    artifact_sha256: Option<String>,
    artifact_size: Option<u64>,
    passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UpdateOptions {
    workspace: PathBuf,
    version: Option<String>,
    commit: Option<String>,
    output: Option<PathBuf>,
}

fn parse_update_options(parser: &mut Arguments) -> Result<UpdateOptions, String> {
    let workspace = parser
        .take_option("--workspace")?
        .map(PathBuf::from)
        .unwrap_or(env::current_dir().map_err(|error| error.to_string())?);
    let version = parser.take_option("--version")?;
    let commit = parser.take_option("--commit")?;
    let output = parser.take_option("--output")?.map(PathBuf::from);
    parser.ensure_empty()?;
    if !workspace.is_dir() {
        return Err(format!("workspace does not exist: {}", workspace.display()));
    }
    Ok(UpdateOptions {
        workspace,
        version,
        commit,
        output,
    })
}

fn run_self_review(mut parser: Arguments) -> Result<(), String> {
    let options = parse_update_options(&mut parser)?;
    let workspace = options
        .workspace
        .canonicalize()
        .map_err(|error| format!("failed to resolve workspace: {error}"))?;
    let commit = git_commit(&workspace);
    let target_dir = self_update_target_dir(&workspace);
    let checks = [
        ("fmt", vec!["fmt", "--all", "--", "--check"]),
        (
            "workspace-tests",
            vec!["test", "--workspace", "--locked", "--offline"],
        ),
        (
            "clippy",
            vec![
                "clippy",
                "--workspace",
                "--all-targets",
                "--locked",
                "--offline",
                "--",
                "-D",
                "warnings",
            ],
        ),
        (
            "release-build",
            vec![
                "build",
                "--release",
                "--locked",
                "--offline",
                "--bin",
                "focus",
            ],
        ),
    ]
    .into_iter()
    .map(|(name, args)| {
        run_review_check_with_target(&workspace, name, "cargo", &args, Some(&target_dir))
    })
    .collect::<Vec<_>>();
    let artifact_path = release_executable_at(&target_dir);
    let (artifact_sha256, artifact_size) = if artifact_path.is_file() {
        focus_runtime::update::hash_file(&artifact_path)
            .ok()
            .map_or((None, None), |(hash, size)| (Some(hash), Some(size)))
    } else {
        (None, None)
    };
    let receipt = SelfReviewReceipt {
        schema_version: 1,
        workspace: workspace.display().to_string(),
        commit,
        checks: checks.clone(),
        artifact_path: artifact_path
            .is_file()
            .then(|| artifact_path.display().to_string()),
        artifact_sha256,
        artifact_size,
        passed: checks.iter().all(|check| check.passed),
    };
    let encoded = serde_json::to_string_pretty(&receipt).map_err(|error| error.to_string())?;
    if let Some(path) = options.output {
        write_receipt(&path, &encoded)?;
    }
    println!("{encoded}");
    if receipt.passed {
        Ok(())
    } else {
        Err("self-review failed; inspect the JSON receipt".into())
    }
}

fn run_self_update(mut parser: Arguments) -> Result<(), String> {
    let operation = parser
        .next()
        .ok_or_else(|| "self-update requires stage, apply, or rollback".to_owned())?;
    match operation.as_str() {
        "stage" => self_update_stage(parser),
        "apply" => self_update_apply(parser),
        "rollback" => self_update_rollback(parser),
        _ => Err(format!("unknown self-update operation `{operation}`")),
    }
}

fn self_update_stage(mut parser: Arguments) -> Result<(), String> {
    let options = parse_update_options(&mut parser)?;
    let workspace = options
        .workspace
        .canonicalize()
        .map_err(|error| format!("failed to resolve workspace: {error}"))?;
    let target_dir = self_update_target_dir(&workspace);
    let build = run_review_check_with_target(
        &workspace,
        "release-build",
        "cargo",
        &[
            "build",
            "--release",
            "--locked",
            "--offline",
            "--bin",
            "focus",
        ],
        Some(&target_dir),
    );
    if !build.passed {
        return Err(format!("release build failed:\n{}", build.output));
    }
    let commit = options
        .commit
        .or_else(|| git_commit(&workspace))
        .unwrap_or_else(|| "unknown".into());
    let version = options.version.unwrap_or_else(|| {
        let short_commit = commit.chars().take(12).collect::<String>();
        format!("v{}-{}", env!("CARGO_PKG_VERSION"), short_commit)
    });
    let artifact = VersionedArtifactStore::new(default_update_root())
        .stage_file(&release_executable_at(&target_dir), version, commit)
        .map_err(|error| error.to_string())?;
    let encoded = serde_json::to_string_pretty(&artifact).map_err(|error| error.to_string())?;
    if let Some(path) = options.output {
        write_receipt(&path, &encoded)?;
    }
    println!("{encoded}");
    Ok(())
}

fn self_update_apply(mut parser: Arguments) -> Result<(), String> {
    let output = parser.take_option("--output")?.map(PathBuf::from);
    parser.ensure_empty()?;
    let store = VersionedArtifactStore::new(default_update_root());
    let artifact = store
        .load_staged()
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "no staged Focus artifact; run self-update stage first".to_owned())?;
    let manifest = store
        .activate(&artifact)
        .map_err(|error| error.to_string())?;
    write_update_manifest(output, &manifest)
}

fn self_update_rollback(mut parser: Arguments) -> Result<(), String> {
    let output = parser.take_option("--output")?.map(PathBuf::from);
    parser.ensure_empty()?;
    let manifest = VersionedArtifactStore::new(default_update_root())
        .rollback()
        .map_err(|error| error.to_string())?;
    write_update_manifest(output, &manifest)
}

fn write_update_manifest(output: Option<PathBuf>, manifest: &UpdateManifest) -> Result<(), String> {
    let encoded = serde_json::to_string_pretty(manifest).map_err(|error| error.to_string())?;
    if let Some(path) = output {
        write_receipt(&path, &encoded)?;
    }
    println!("{encoded}");
    Ok(())
}

fn run_review_check_with_target(
    workspace: &Path,
    name: &str,
    program: &str,
    args: &[&str],
    target_dir: Option<&Path>,
) -> ReviewCheck {
    let command = format!("{} {}", program, args.join(" "));
    let mut command_builder = Command::new(program);
    command_builder.args(args).current_dir(workspace);
    if let Some(target_dir) = target_dir {
        command_builder.env("CARGO_TARGET_DIR", target_dir);
    }
    let result = command_builder.output();
    match result {
        Ok(output) => ReviewCheck {
            name: name.into(),
            command,
            passed: output.status.success(),
            output: bounded_command_output(&output),
        },
        Err(error) => ReviewCheck {
            name: name.into(),
            command,
            passed: false,
            output: error.to_string(),
        },
    }
}

fn bounded_command_output(output: &std::process::Output) -> String {
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    bounded_output(text, 64 * 1024)
}

fn bounded_output(mut text: String, max_bytes: usize) -> String {
    if text.len() > max_bytes {
        let mut end = max_bytes;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str("\n[output truncated]");
    }
    text
}

fn git_commit(workspace: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(workspace)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn release_executable_at(target: &Path) -> PathBuf {
    target
        .join("release")
        .join(format!("focus{}", std::env::consts::EXE_SUFFIX))
}

fn self_update_target_dir(workspace: &Path) -> PathBuf {
    workspace.join("target").join(".focus-self-update")
}

fn benchmark_passed(receipt: &focus_runtime::benchmark::BenchmarkReceipt) -> bool {
    receipt.records.iter().all(|record| record.metrics.success)
        && receipt.failure_probes.iter().all(|probe| probe.passed)
        && receipt.cancellation_probes.iter().all(|probe| probe.passed)
}

fn write_receipt(path: &Path, content: &str) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let temporary = parent.join(format!(".focus-benchmark-{}.tmp", Uuid::new_v4()));
    std::fs::write(&temporary, content)
        .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
    let backup = parent.join(format!(".focus-benchmark-{}.bak", Uuid::new_v4()));
    publish_receipt(&temporary, path, &backup)
}

fn publish_receipt(temporary: &Path, path: &Path, backup: &Path) -> Result<(), String> {
    let had_previous = path.is_file();
    if had_previous {
        std::fs::rename(path, backup).map_err(|error| {
            format!(
                "failed to stage {} for replacement: {error}",
                path.display()
            )
        })?;
    }
    if let Err(publish_error) = std::fs::rename(temporary, path) {
        let rollback = if had_previous {
            std::fs::rename(backup, path).map_err(|error| error.to_string())
        } else {
            Ok(())
        };
        let _ = std::fs::remove_file(temporary);
        return match rollback {
            Ok(()) => Err(format!(
                "failed to publish {}: {publish_error}; previous receipt restored",
                path.display()
            )),
            Err(rollback_error) => Err(format!(
                "failed to publish {}: {publish_error}; rollback failed: {rollback_error}",
                path.display()
            )),
        };
    }
    if had_previous {
        std::fs::remove_file(backup).map_err(|error| {
            format!(
                "published {} but failed to remove backup {}: {error}",
                path.display(),
                backup.display()
            )
        })?;
    }
    Ok(())
}

fn run_agent(mut parser: Arguments) -> Result<(), String> {
    let options = CommonOptions::parse(&mut parser)?;
    let provider = ProviderSelection::parse(&mut parser)?.into_provider()?;
    let approval_mode = ApprovalMode::parse(&mut parser)?;
    let capabilities = RunCapabilities::production();
    let session = RunSession::parse(&mut parser)?;
    let goal = RunGoal::parse(&mut parser)?;
    let title = parser
        .take_option("--title")?
        .unwrap_or_else(|| "Code agent task".into());
    let task = parser.remaining_task()?;
    let runtime = open_runtime(options)?;
    let session = match (session, goal) {
        (Some(session_id), Some(goal_id)) => {
            runtime
                .attach_goal_session(goal_id, session_id)
                .map_err(|error| error.to_string())?;
            Some(session_id)
        }
        (None, Some(goal_id)) => {
            let created = runtime
                .create_session(title.clone())
                .map_err(|error| error.to_string())?;
            runtime
                .attach_goal_session(goal_id, created.id)
                .map_err(|error| error.to_string())?;
            Some(created.id)
        }
        (session, None) => session,
    };
    execute(
        &runtime,
        session,
        title,
        task,
        provider,
        approval_mode,
        capabilities,
    )
}

fn run_chat(mut parser: Arguments) -> Result<(), String> {
    let options = CommonOptions::parse(&mut parser)?;
    let selection = ProviderSelection::parse(&mut parser)?;
    let model_label = selection.label();
    let provider = selection.into_provider()?;
    let approval_mode = ApprovalMode::parse(&mut parser)?;
    let capabilities = RunCapabilities::production();
    let session_target = ChatSessionTarget::parse(&mut parser)?;
    let plain = parser.take_flag("--plain");
    let title = parser
        .take_option("--title")?
        .unwrap_or_else(|| "Focus chat".into());
    parser.ensure_empty()?;
    let runtime = open_runtime(options)?;
    let session = resolve_chat_session(
        &runtime,
        session_target,
        &title,
        env::var_os(supervisor::HANDOFF_PATH_ENV).is_some(),
    )?;
    if !plain && tui::is_supported() {
        return tui::run(
            runtime,
            session,
            title,
            model_label,
            provider,
            capabilities,
            approval_mode,
        );
    }
    let approval = match approval_mode {
        ApprovalMode::ApproveAll => approve_all(),
        ApprovalMode::DenyAll => deny_approvals(),
        ApprovalMode::Interactive => {
            return run_chat_interactive(runtime, session, title, provider, capabilities);
        }
    };
    run_chat_with_approval(runtime, session, title, provider, capabilities, approval)
}

fn resolve_chat_session(
    runtime: &FocusRuntime,
    target: ChatSessionTarget,
    title: &str,
    handoff_present: bool,
) -> Result<Option<Uuid>, String> {
    if handoff_present {
        return Ok(target.session);
    }
    match target.new_session {
        Some(id) => Ok(Some(
            runtime
                .create_session_with_id(id, title.to_owned())
                .map_err(|error| error.to_string())?
                .id,
        )),
        None => Ok(target.session),
    }
}

fn run_chat_with_approval(
    runtime: FocusRuntime,
    session: Option<Uuid>,
    title: String,
    provider: Arc<dyn ModelProvider>,
    capabilities: RunCapabilities,
    approval: Arc<dyn focus_runtime::policy::ApprovalHandler>,
) -> Result<(), String> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    run_chat_with_approval_reader(
        runtime,
        session,
        title,
        provider,
        capabilities,
        approval,
        &mut reader,
    )
}

fn run_chat_with_approval_reader<R: BufRead>(
    runtime: FocusRuntime,
    session: Option<Uuid>,
    title: String,
    provider: Arc<dyn ModelProvider>,
    capabilities: RunCapabilities,
    approval: Arc<dyn focus_runtime::policy::ApprovalHandler>,
    reader: &mut R,
) -> Result<(), String> {
    let mut chat = ChatSession { session };
    println!("{}", chat_welcome());
    loop {
        print_chat_prompt()?;
        let mut task = String::new();
        if reader
            .read_line(&mut task)
            .map_err(|error| error.to_string())?
            == 0
        {
            return Ok(());
        }
        let task = match classify_chat_input(&task) {
            ChatInput::Help => {
                println!("{}", chat_help());
                continue;
            }
            ChatInput::Exit => return Ok(()),
            ChatInput::Task(task) if task.is_empty() => continue,
            ChatInput::Task(task) => task,
        };
        let renderer = spawn_stderr_event_renderer(runtime.subscribe());
        let result = chat.run_turn(
            &runtime,
            &title,
            &task,
            provider.clone(),
            capabilities,
            approval.clone(),
        );
        let streamed = renderer.finish()?;
        print_chat_turn(result, &streamed);
    }
}

fn run_chat_interactive(
    runtime: FocusRuntime,
    session: Option<Uuid>,
    title: String,
    provider: Arc<dyn ModelProvider>,
    capabilities: RunCapabilities,
) -> Result<(), String> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut chat = ChatSession { session };
    println!("{}", chat_welcome());
    loop {
        print_chat_prompt()?;
        let mut task = String::new();
        if reader
            .read_line(&mut task)
            .map_err(|error| error.to_string())?
            == 0
        {
            return Ok(());
        }
        let task = match classify_chat_input(&task) {
            ChatInput::Help => {
                println!("{}", chat_help());
                continue;
            }
            ChatInput::Exit => return Ok(()),
            ChatInput::Task(task) if task.is_empty() => continue,
            ChatInput::Task(task) => task,
        };
        let cancellation = new_chat_cancellation();
        let (broker, inbox) = ApprovalBroker::new();
        let approval = Arc::new(broker);
        let mut options = capabilities.options(approval);
        options.cancellation = Arc::new(cancellation.clone());
        let renderer = spawn_stderr_event_renderer(runtime.subscribe());
        let runtime_for_worker = runtime.clone();
        let title_for_worker = title.clone();
        let task_for_worker = task;
        let provider_for_worker = provider.clone();
        let mut worker_chat = chat;
        let worker_options = options.clone();
        let worker = thread::spawn(move || {
            let result = worker_chat.run_turn_with_options(
                &runtime_for_worker,
                &title_for_worker,
                &task_for_worker,
                provider_for_worker,
                worker_options,
            );
            (worker_chat, result)
        });
        let interaction = drive_approval_inbox(&inbox, &worker, &mut reader);
        let (next_chat, result) =
            cancel_and_join_chat(worker, inbox, cancellation.clone(), interaction)?;
        chat = next_chat;
        let streamed = renderer.finish()?;
        print_chat_turn(result, &streamed);
    }
}

fn print_chat_turn(result: Result<RunResult, String>, streamed: &str) {
    match result {
        Ok(result) => print_result(&result, streamed),
        Err(error) => eprintln!("\nerror: {error}"),
    }
}

fn new_chat_cancellation() -> CancellationToken {
    CancellationToken::default()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatCommand {
    Help,
    Review,
    Simplify,
    Exit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ChatInput {
    Task(String),
    Help,
    Exit,
}

fn chat_welcome() -> &'static str {
    "Focus\nType a task to start. /help for commands."
}

fn chat_prompt() -> &'static str {
    "focus> "
}

fn chat_help() -> &'static str {
    "Commands: /help, /review, /simplify, /exit, /quit"
}

fn print_chat_prompt() -> Result<(), String> {
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    write!(writer, "{}", chat_prompt()).map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

fn parse_chat_command(input: &str) -> Option<ChatCommand> {
    match input.trim() {
        "/help" => Some(ChatCommand::Help),
        "/review" => Some(ChatCommand::Review),
        "/simplify" => Some(ChatCommand::Simplify),
        "/exit" | "/quit" => Some(ChatCommand::Exit),
        _ => None,
    }
}

fn classify_chat_input(input: &str) -> ChatInput {
    match parse_chat_command(input) {
        Some(ChatCommand::Help) => ChatInput::Help,
        Some(ChatCommand::Review) => ChatInput::Task(REVIEW_TASK.into()),
        Some(ChatCommand::Simplify) => ChatInput::Task(SIMPLIFY_TASK.into()),
        Some(ChatCommand::Exit) => ChatInput::Exit,
        None => ChatInput::Task(input.trim().to_owned()),
    }
}

fn run_demo(mut parser: Arguments) -> Result<(), String> {
    let options = CommonOptions::parse(&mut parser)?;
    let title = parser
        .take_option("--title")?
        .unwrap_or_else(|| "Runtime demo".into());
    let task = parser.remaining_task()?;
    let runtime = open_runtime(options)?;
    let provider = StaticProvider::new(
        "Runtime demo completed. Configure `run --adapter` for a model-backed coding turn.",
    );
    execute(
        &runtime,
        None,
        title,
        task,
        Arc::new(provider),
        ApprovalMode::DenyAll,
        RunCapabilities::core(),
    )
}

fn run_host(mut parser: Arguments) -> Result<(), String> {
    if !parser.take_flag("--stdio") {
        return Err("`host` requires --stdio".into());
    }
    let options = CommonOptions::parse(&mut parser)?;
    parser.ensure_empty()?;
    run_host_stdio(open_runtime(options)?)
}

fn parse_host_request(line: &str) -> Result<HostRequestV1, String> {
    let request: HostRequestV1 =
        serde_json::from_str(line).map_err(|error| format!("invalid host request: {error}"))?;
    request.validate().map_err(|error| error.to_string())?;
    Ok(request)
}

fn run_host_stdio(runtime: FocusRuntime) -> Result<(), String> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();
    if reader
        .read_line(&mut line)
        .map_err(|error| error.to_string())?
        == 0
    {
        return Ok(());
    }
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    let request = match parse_host_request(line.trim()) {
        Ok(request) => request,
        Err(error) => {
            write_host_frame(&mut writer, &HostFrameV1::error(error))?;
            return Ok(());
        }
    };
    stream_host_attachment(&runtime, request, &mut writer)
}

fn stream_host_attachment(
    runtime: &FocusRuntime,
    request: HostRequestV1,
    writer: &mut impl Write,
) -> Result<(), String> {
    let session_id = request.session_id();
    // Subscribe first: events committed while replay is loading remain queued and are deduplicated below.
    let receiver = runtime.subscribe();
    // Record the append-only file position before the baseline. Any concurrent
    // append is either in the replay or appears in this tail and is deduplicated.
    let mut tail = runtime
        .host_event_tail(session_id)
        .map_err(|error| error.to_string())?;
    let replay = runtime
        .host_replay(session_id, request.after_event_id())
        .map_err(|error| error.to_string())?;
    if replay.reset {
        write_host_frame(writer, &HostFrameV1::reset(session_id))?;
    }
    for frame in &replay.events {
        write_host_frame(writer, &HostFrameV1::Event(frame.clone()))?;
    }
    if replay
        .events
        .last()
        .is_some_and(|frame| host_attachment_should_close(&request, &frame.event))
    {
        return Ok(());
    }
    let mut deduplicator = replay.deduplicator();
    loop {
        match receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(event) if event.session_id == session_id => {
                if let Some(frame) = deduplicator.accept(event) {
                    let terminal = host_attachment_should_close(&request, &frame.event);
                    write_host_frame(writer, &HostFrameV1::Event(frame))?;
                    if terminal {
                        return Ok(());
                    }
                }
            }
            Ok(_) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                for event in tail.read_new().map_err(|error| error.to_string())? {
                    if event.session_id == session_id
                        && let Some(frame) = deduplicator.accept(event)
                    {
                        let terminal = host_attachment_should_close(&request, &frame.event);
                        write_host_frame(writer, &HostFrameV1::Event(frame))?;
                        if terminal {
                            return Ok(());
                        }
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
        }
    }
}

fn host_event_is_terminal(session_id: Uuid, event: &Event) -> bool {
    event.session_id == session_id
        && matches!(
            event.kind,
            EventKind::StateChanged {
                status: focus_kernel::AgentStatus::Complete
                    | focus_kernel::AgentStatus::Failed
                    | focus_kernel::AgentStatus::Cancelled,
                ..
            }
        )
}

fn host_attachment_should_close(request: &HostRequestV1, event: &Event) -> bool {
    !request.follow() && host_event_is_terminal(request.session_id(), event)
}

fn write_host_frame(writer: &mut impl Write, frame: &HostFrameV1) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, frame).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApprovalMode {
    Interactive,
    ApproveAll,
    DenyAll,
}

/// The default production capability profile exposed by the CLI.
///
/// The Runtime still exposes individual `RunOptions` constructors for
/// benchmarks and embedded hosts, but the user-facing entry points always
/// run with the strongest built-in profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RunCapabilities {
    workflow: bool,
    delegation: bool,
}

impl Default for RunCapabilities {
    fn default() -> Self {
        Self {
            workflow: true,
            delegation: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunSession;

impl RunSession {
    fn parse(parser: &mut Arguments) -> Result<Option<Uuid>, String> {
        parser
            .take_option("--session")?
            .map(|value| {
                Uuid::parse_str(&value).map_err(|error| format!("invalid --session: {error}"))
            })
            .transpose()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChatSessionTarget {
    session: Option<Uuid>,
    new_session: Option<Uuid>,
}

impl ChatSessionTarget {
    fn parse(parser: &mut Arguments) -> Result<Self, String> {
        let session = RunSession::parse(parser)?;
        let new_session = parser
            .take_option("--new-session")?
            .map(|value| {
                Uuid::parse_str(&value).map_err(|error| format!("invalid --new-session: {error}"))
            })
            .transpose()?;
        if session.is_some() && new_session.is_some() {
            return Err("--session and --new-session are mutually exclusive".into());
        }
        Ok(Self {
            session,
            new_session,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunGoal;

impl RunGoal {
    fn parse(parser: &mut Arguments) -> Result<Option<Uuid>, String> {
        parser
            .take_option("--goal")?
            .map(|value| {
                Uuid::parse_str(&value).map_err(|error| format!("invalid --goal: {error}"))
            })
            .transpose()
    }
}

#[derive(Debug, Default)]
struct ChatSession {
    session: Option<Uuid>,
}

impl ChatSession {
    fn run_turn(
        &mut self,
        runtime: &FocusRuntime,
        title: &str,
        task: &str,
        provider: Arc<dyn ModelProvider>,
        capabilities: RunCapabilities,
        approval: Arc<dyn focus_runtime::policy::ApprovalHandler>,
    ) -> Result<RunResult, String> {
        self.run_turn_with_options(
            runtime,
            title,
            task,
            provider,
            capabilities.options(approval),
        )
    }

    fn run_turn_with_options(
        &mut self,
        runtime: &FocusRuntime,
        title: &str,
        task: &str,
        provider: Arc<dyn ModelProvider>,
        options: RunOptions,
    ) -> Result<RunResult, String> {
        run_chat_turn(runtime, &mut self.session, title, task, provider, options)
    }
}

pub(crate) fn run_chat_turn(
    runtime: &FocusRuntime,
    session: &mut Option<Uuid>,
    title: &str,
    task: &str,
    provider: Arc<dyn ModelProvider>,
    options: RunOptions,
) -> Result<RunResult, String> {
    let session_id = match session {
        Some(session_id) => *session_id,
        None => {
            let created = runtime
                .create_session(title.to_owned())
                .map_err(|error| error.to_string())?;
            *session = Some(created.id);
            created.id
        }
    };
    execute_runtime(
        runtime,
        Some(session_id),
        title.to_owned(),
        task.to_owned(),
        provider,
        options,
    )
}

impl RunCapabilities {
    fn core() -> Self {
        Self {
            workflow: false,
            delegation: false,
        }
    }

    fn production() -> Self {
        Self {
            workflow: true,
            delegation: true,
        }
    }

    pub(crate) fn options(
        self,
        approval: Arc<dyn focus_runtime::policy::ApprovalHandler>,
    ) -> RunOptions {
        let options = if self.workflow {
            RunOptions::engineering(approval)
        } else {
            RunOptions::core(approval)
        };
        if self.delegation {
            options.with_delegation(SubagentLimits::default())
        } else {
            options
        }
    }
}

impl ApprovalMode {
    fn parse(parser: &mut Arguments) -> Result<Self, String> {
        let approve_flag = parser.take_flag("--approve");
        let yolo_flag = parser.take_flag("--yolo");
        let interactive = parser.take_flag("--interactive");
        let deny = parser.take_flag("--deny");
        let approve = approve_flag || yolo_flag;
        match (approve, interactive, deny) {
            (true, false, false) => Ok(Self::ApproveAll),
            (false, true, false) => Ok(Self::Interactive),
            (false, false, true) => Ok(Self::DenyAll),
            (false, false, false) => Ok(Self::ApproveAll),
            _ => Err("--yolo/--approve, --interactive, and --deny are mutually exclusive".into()),
        }
    }
}

fn execute(
    runtime: &FocusRuntime,
    session: Option<Uuid>,
    title: String,
    task: String,
    provider: Arc<dyn ModelProvider>,
    approval_mode: ApprovalMode,
    capabilities: RunCapabilities,
) -> Result<(), String> {
    if approval_mode == ApprovalMode::Interactive {
        return execute_interactive(runtime, session, title, task, provider, capabilities);
    }
    let approval = match approval_mode {
        ApprovalMode::ApproveAll => approve_all(),
        ApprovalMode::DenyAll => deny_approvals(),
        ApprovalMode::Interactive => unreachable!(),
    };
    let options = capabilities.options(approval);
    let renderer = spawn_stderr_event_renderer(runtime.subscribe());
    let result = execute_runtime(runtime, session, title, task, provider, options);
    let streamed = renderer.finish()?;
    let result = result?;
    print_result(&result, &streamed);
    Ok(())
}

fn execute_interactive(
    runtime: &FocusRuntime,
    session: Option<Uuid>,
    title: String,
    task: String,
    provider: Arc<dyn ModelProvider>,
    capabilities: RunCapabilities,
) -> Result<(), String> {
    let (broker, inbox) = ApprovalBroker::new();
    let approval = Arc::new(broker);
    let cancellation = CancellationToken::default();
    let mut options = capabilities.options(approval);
    options.cancellation = Arc::new(cancellation.clone());
    let renderer = spawn_stderr_event_renderer(runtime.subscribe());
    let runtime = runtime.clone();
    let worker =
        thread::spawn(move || execute_runtime(&runtime, session, title, task, provider, options));

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let interaction = drive_approval_inbox(&inbox, &worker, &mut reader);
    let result = cancel_and_join(worker, inbox, cancellation, interaction);
    let streamed = renderer.finish()?;
    let result = result?;
    print_result(&result, &streamed);
    Ok(())
}

fn execute_runtime(
    runtime: &FocusRuntime,
    session: Option<Uuid>,
    title: String,
    task: String,
    provider: Arc<dyn ModelProvider>,
    options: RunOptions,
) -> Result<RunResult, String> {
    match session {
        Some(session_id) => runtime
            .run_in_session_with_options(session_id, task, provider, options)
            .map_err(|error| error.to_string()),
        None => runtime
            .run_with_options(title, task, provider, options)
            .map_err(|error| error.to_string()),
    }
}

fn drive_approval_inbox<T, R: BufRead>(
    inbox: &ApprovalInbox,
    worker: &thread::JoinHandle<T>,
    reader: &mut R,
) -> Result<(), String> {
    while !worker.is_finished() {
        if let Some(pending) = inbox
            .recv_timeout(Duration::from_millis(100))
            .map_err(|error| error.to_string())?
        {
            let stderr = io::stderr();
            let mut writer = stderr.lock();
            let decision = prompt_for_approval(&pending, reader, &mut writer)?;
            inbox
                .resolve(pending.id, decision)
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn cancel_and_join<T>(
    worker: thread::JoinHandle<Result<T, String>>,
    inbox: ApprovalInbox,
    cancellation: CancellationToken,
    interaction: Result<(), String>,
) -> Result<T, String> {
    if interaction.is_err() {
        cancellation.cancel();
    }
    drop(inbox);
    let worker_result = worker
        .join()
        .map_err(|_| "Runtime worker panicked".to_owned())?;
    match interaction {
        Ok(()) => worker_result,
        Err(error) => Err(error),
    }
}

fn cancel_and_join_chat(
    worker: thread::JoinHandle<(ChatSession, Result<RunResult, String>)>,
    inbox: ApprovalInbox,
    cancellation: CancellationToken,
    interaction: Result<(), String>,
) -> Result<(ChatSession, Result<RunResult, String>), String> {
    if interaction.is_err() {
        cancellation.cancel();
    }
    drop(inbox);
    let worker_result = worker
        .join()
        .map_err(|_| "Runtime worker panicked".to_owned())?;
    match interaction {
        Ok(()) => Ok(worker_result),
        Err(error) => Err(error),
    }
}

fn prompt_for_approval<R: BufRead, W: Write>(
    pending: &ApprovalEnvelope,
    reader: &mut R,
    writer: &mut W,
) -> Result<ApprovalDecision, String> {
    writeln!(
        writer,
        "\nPermission required\n{} ({})\n{}{}",
        pending.request.tool,
        approval_operation_label(pending.request.operation),
        pending.request.rationale,
        approval_argument_preview(&pending.request.arguments)
    )
    .map_err(|error| error.to_string())?;
    loop {
        write!(writer, "Allow? [y] once / [s] session / [n] deny: ")
            .map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())?;
        let mut input = String::new();
        if reader
            .read_line(&mut input)
            .map_err(|error| error.to_string())?
            == 0
        {
            return Ok(ApprovalDecision::Deny);
        }
        if let Some(decision) = parse_approval_decision(&input) {
            return Ok(decision);
        }
        writeln!(writer, "Enter y, s, or n.").map_err(|error| error.to_string())?;
    }
}

fn parse_approval_decision(input: &str) -> Option<ApprovalDecision> {
    match input.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" | "once" => Some(ApprovalDecision::ApproveOnce),
        "s" | "session" => Some(ApprovalDecision::ApproveSession),
        "n" | "no" | "deny" | "" => Some(ApprovalDecision::Deny),
        _ => None,
    }
}

fn approval_operation_label(operation: ToolOperation) -> &'static str {
    match operation {
        ToolOperation::Read => "read",
        ToolOperation::Write => "write",
        ToolOperation::Execute => "execute",
        ToolOperation::Network => "network",
        ToolOperation::Other => "extension",
    }
}

fn approval_argument_preview(arguments: &serde_json::Value) -> String {
    let Some(arguments) = arguments.as_object() else {
        return String::new();
    };
    for key in ["command", "path", "url"] {
        if let Some(value) = arguments.get(key).and_then(serde_json::Value::as_str) {
            return format!("\n{key}: {}", focus_kernel::redact_sensitive_text(value));
        }
    }
    String::new()
}

fn print_result(result: &RunResult, streamed: &str) {
    let remainder = format_final_response(&result.final_response, streamed);
    if !remainder.is_empty() {
        println!("{remainder}");
    } else if !streamed.ends_with('\n') {
        println!();
    }
}

fn format_final_response(final_response: &str, streamed: &str) -> String {
    let final_response = redact_sensitive_text(final_response);
    final_response
        .strip_prefix(streamed)
        .unwrap_or(&final_response)
        .to_owned()
}

#[derive(Debug)]
struct EventRenderer {
    complete: Arc<AtomicBool>,
    visible_text: Arc<Mutex<String>>,
    worker: thread::JoinHandle<io::Result<()>>,
}

impl EventRenderer {
    fn finish(self) -> Result<String, String> {
        self.complete.store(true, Ordering::Release);
        self.worker
            .join()
            .map_err(|_| "event renderer panicked".to_owned())?
            .map_err(|error| error.to_string())?;
        self.visible_text
            .lock()
            .map(|text| text.clone())
            .map_err(|_| "event renderer state lock poisoned".to_owned())
    }
}

fn render_model_event(
    event: &Event,
    writer: &mut impl Write,
    visible_text: &Mutex<String>,
) -> io::Result<()> {
    match &event.kind {
        EventKind::Model {
            event: ModelEvent::RequestStarted { .. },
        } => {
            visible_text
                .lock()
                .map_err(|_| io::Error::other("event renderer state lock poisoned"))?
                .clear();
            Ok(())
        }
        EventKind::Model {
            event: ModelEvent::TextDelta { text },
        } => {
            let text = redact_sensitive_text(text);
            visible_text
                .lock()
                .map_err(|_| io::Error::other("event renderer state lock poisoned"))?
                .push_str(&text);
            write!(writer, "{text}")?;
            writer.flush()
        }
        EventKind::ToolCallRequested { call } => {
            writeln!(
                writer,
                "\n- Using {}{}",
                call.name,
                tool_argument_preview(&call.arguments)
            )
        }
        EventKind::ToolResultReceived { result } if result.is_error => {
            writeln!(writer, "- {} failed", result.name)
        }
        EventKind::ToolResultReceived { result } => writeln!(writer, "- {} finished", result.name),
        _ => Ok(()),
    }
}

fn tool_argument_preview(arguments: &serde_json::Value) -> String {
    let Some(arguments) = arguments.as_object() else {
        return String::new();
    };
    for key in ["command", "path", "url"] {
        if let Some(value) = arguments.get(key).and_then(serde_json::Value::as_str) {
            return format!(": {}", focus_kernel::redact_sensitive_text(value));
        }
    }
    String::new()
}

fn spawn_stderr_event_renderer(receiver: std::sync::mpsc::Receiver<Event>) -> EventRenderer {
    spawn_event_renderer(receiver, io::stderr())
}

fn spawn_event_renderer<W: Write + Send + 'static>(
    receiver: std::sync::mpsc::Receiver<Event>,
    mut writer: W,
) -> EventRenderer {
    let complete = Arc::new(AtomicBool::new(false));
    let visible_text = Arc::new(Mutex::new(String::new()));
    let renderer_complete = complete.clone();
    let renderer_text = visible_text.clone();
    let worker = thread::spawn(move || {
        loop {
            match receiver.recv_timeout(Duration::from_millis(20)) {
                Ok(event) => render_model_event(&event, &mut writer, &renderer_text)?,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                    if renderer_complete.load(Ordering::Acquire) =>
                {
                    return Ok(());
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
            }
        }
    });
    EventRenderer {
        complete,
        visible_text,
        worker,
    }
}

fn run_doctor(mut parser: Arguments) -> Result<(), String> {
    let live_containers = parser.take_flag("--live-containers");
    let options = CommonOptions::parse(&mut parser)?;
    parser.ensure_empty()?;
    let runtime = open_runtime(options)?;
    if live_containers {
        let evidence = runtime
            .config()
            .sandbox_backend
            .clone()
            .live_acceptance(&runtime.config().workspace_root)
            .map_err(|error| error.to_string())?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "doctor": doctor(&runtime),
                "live_container_acceptance": evidence,
            }))
            .map_err(|error| error.to_string())?
        );
        if !matches!(
            evidence.termination,
            focus_runtime::sandbox::CommandTermination::Exited(0)
        ) || !evidence.workspace_mount
            || !evidence.network_isolated
            || !evidence.resource_limits
            || !evidence.cleanup
        {
            return Err("live container acceptance failed; inspect the JSON evidence".into());
        }
        return Ok(());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&doctor(&runtime)).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn run_session(mut parser: Arguments) -> Result<(), String> {
    let operation = parser
        .next()
        .ok_or_else(|| "`session` requires list, show, replay, or fork".to_owned())?;
    let options = CommonOptions::parse(&mut parser)?;
    let runtime = open_runtime(options)?;
    match operation.as_str() {
        "list" => {
            parser.ensure_empty()?;
            for session in runtime.list_sessions().map_err(|error| error.to_string())? {
                println!("{}\t{:?}\t{}", session.id, session.phase, session.title);
            }
            Ok(())
        }
        "show" => {
            let id = parse_session_id(&mut parser, "`session show` requires SESSION_ID")?;
            parser.ensure_empty()?;
            print_json(&runtime.session(id).map_err(|error| error.to_string())?)
        }
        "replay" => {
            let id = parser
                .next()
                .ok_or_else(|| "`session replay` requires SESSION_ID".to_owned())?;
            parser.ensure_empty()?;
            let id =
                Uuid::parse_str(&id).map_err(|error| format!("invalid session id: {error}"))?;
            for event in runtime.replay(id).map_err(|error| error.to_string())? {
                println!(
                    "{}\t{}",
                    event.timestamp_ms,
                    serde_json::to_string(&event.kind).map_err(|error| error.to_string())?
                );
            }
            Ok(())
        }
        "fork" => {
            let parent = parser
                .next()
                .ok_or_else(|| "`session fork` requires PARENT_SESSION_ID".to_owned())?;
            let title = parser.remaining_task()?;
            let parent = Uuid::parse_str(&parent)
                .map_err(|error| format!("invalid parent session id: {error}"))?;
            let child = runtime
                .fork_session(parent, title)
                .map_err(|error| error.to_string())?;
            println!("{}", child.id);
            Ok(())
        }
        other => Err(format!("unknown session operation `{other}`")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MemorySelection {
    scope: MemoryScope,
    session_id: Option<Uuid>,
}

impl MemorySelection {
    fn parse(parser: &mut Arguments) -> Result<Self, String> {
        let scope = match parser.take_option("--scope")?.as_deref() {
            None | Some("project") => MemoryScope::Project,
            Some("session") => MemoryScope::Session,
            Some(value) => {
                return Err(format!(
                    "invalid --scope `{value}`; expected project or session"
                ));
            }
        };
        let session_id = parser
            .take_option("--session")?
            .map(|value| {
                Uuid::parse_str(&value).map_err(|error| format!("invalid --session: {error}"))
            })
            .transpose()?;
        match (scope, session_id) {
            (MemoryScope::Project, Some(_)) => {
                Err("--session is only valid with --scope session".into())
            }
            (MemoryScope::Session, None) => {
                Err("--scope session requires --session SESSION_ID".into())
            }
            _ => Ok(Self { scope, session_id }),
        }
    }
}

fn run_memory(mut parser: Arguments) -> Result<(), String> {
    let operation = parser
        .next()
        .ok_or_else(|| "`memory` requires add or list".to_owned())?;
    let options = CommonOptions::parse(&mut parser)?;
    let selection = MemorySelection::parse(&mut parser)?;
    let runtime = open_runtime(options)?;
    match operation.as_str() {
        "add" => {
            let source = parser
                .take_option("--source")?
                .unwrap_or_else(|| "operator".into());
            let content = parser.remaining_task()?;
            let entry = runtime
                .remember(selection.scope, selection.session_id, content, source)
                .map_err(|error| error.to_string())?;
            print_json(&entry)
        }
        "list" => {
            parser.ensure_empty()?;
            let entries = runtime
                .list_memory(selection.scope, selection.session_id)
                .map_err(|error| error.to_string())?;
            print_json(&entries)
        }
        other => Err(format!("unknown memory operation `{other}`")),
    }
}

struct SubagentLifecycleEvent<'a> {
    timestamp_ms: u128,
    kind: &'a str,
    data: &'a serde_json::Value,
}

fn subagent_lifecycle_events(events: &[Event]) -> impl Iterator<Item = SubagentLifecycleEvent<'_>> {
    events.iter().filter_map(|event| match &event.kind {
        EventKind::Runtime { name, data }
            if matches!(
                name.as_str(),
                "subagent_queued"
                    | "subagent_started"
                    | "subagent_completed"
                    | "subagent_failed"
                    | "subagent_cancelled"
                    | "subagent_batch_completed"
            ) =>
        {
            Some(SubagentLifecycleEvent {
                timestamp_ms: event.timestamp_ms,
                kind: name,
                data,
            })
        }
        _ => None,
    })
}

fn run_subagent(mut parser: Arguments) -> Result<(), String> {
    let operation = parser
        .next()
        .ok_or_else(|| "`subagent` requires list".to_owned())?;
    let options = CommonOptions::parse(&mut parser)?;
    let runtime = open_runtime(options)?;
    match operation.as_str() {
        "list" => {
            let session_id = parser
                .take_option("--session")?
                .ok_or_else(|| "`subagent list` requires --session SESSION_ID".to_owned())?;
            parser.ensure_empty()?;
            let session_id = Uuid::parse_str(&session_id)
                .map_err(|error| format!("invalid --session: {error}"))?;
            for event in subagent_lifecycle_events(
                &runtime
                    .replay(session_id)
                    .map_err(|error| error.to_string())?,
            ) {
                println!(
                    "{}\t{}\t{}",
                    event.timestamp_ms,
                    event.kind,
                    serde_json::to_string(event.data).map_err(|error| error.to_string())?
                );
            }
            Ok(())
        }
        other => Err(format!("unknown subagent operation `{other}`")),
    }
}

fn run_goal(mut parser: Arguments) -> Result<(), String> {
    let operation = parser.next().ok_or_else(|| {
        "`goal` requires create, list, show, attach, complete, or cancel".to_owned()
    })?;
    let options = CommonOptions::parse(&mut parser)?;
    let runtime = open_runtime(options)?;
    match operation.as_str() {
        "create" => {
            let title = parser
                .take_option("--title")?
                .ok_or_else(|| "`goal create` requires --title TITLE".to_owned())?;
            let goal = runtime
                .create_goal(title, parser.remaining_task()?)
                .map_err(|error| error.to_string())?;
            print_json(&goal)
        }
        "list" => {
            parser.ensure_empty()?;
            print_json(&runtime.list_goals().map_err(|error| error.to_string())?)
        }
        "show" => {
            let id = parse_goal_id(&mut parser, "`goal show` requires GOAL_ID")?;
            parser.ensure_empty()?;
            print_json(&runtime.goal(id).map_err(|error| error.to_string())?)
        }
        "attach" => {
            let goal_id = parse_goal_id(&mut parser, "`goal attach` requires GOAL_ID SESSION_ID")?;
            let session_id =
                parse_session_id(&mut parser, "`goal attach` requires GOAL_ID SESSION_ID")?;
            parser.ensure_empty()?;
            print_json(
                &runtime
                    .attach_goal_session(goal_id, session_id)
                    .map_err(|error| error.to_string())?,
            )
        }
        "complete" | "cancel" => {
            let id = parse_goal_id(&mut parser, "`goal complete|cancel` requires GOAL_ID")?;
            parser.ensure_empty()?;
            let phase = if operation == "complete" {
                GoalPhase::Complete
            } else {
                GoalPhase::Cancelled
            };
            print_json(
                &runtime
                    .set_goal_phase(id, phase)
                    .map_err(|error| error.to_string())?,
            )
        }
        other => Err(format!("unknown goal operation `{other}`")),
    }
}

fn parse_session_id(parser: &mut Arguments, error: &str) -> Result<Uuid, String> {
    let id = parser.next().ok_or_else(|| error.to_owned())?;
    Uuid::parse_str(&id).map_err(|parse_error| format!("invalid session id: {parse_error}"))
}

fn parse_goal_id(parser: &mut Arguments, error: &str) -> Result<Uuid, String> {
    let id = parser.next().ok_or_else(|| error.to_owned())?;
    Uuid::parse_str(&id).map_err(|parse_error| format!("invalid goal id: {parse_error}"))
}

fn print_json(value: &impl serde::Serialize) -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string_pretty(value).map_err(|error| error.to_string())?
    );
    Ok(())
}

#[derive(Debug)]
struct CommonOptions {
    workspace: PathBuf,
    state: Option<PathBuf>,
    mcp_config: Option<PathBuf>,
    enable_mcp: bool,
    sandbox_backend: SandboxBackend,
    network: NetworkConfig,
}

impl CommonOptions {
    fn parse(parser: &mut Arguments) -> Result<Self, String> {
        let workspace = parser
            .take_option("--workspace")?
            .map(PathBuf::from)
            .unwrap_or(env::current_dir().map_err(|error| error.to_string())?);
        let state = parser.take_option("--state")?.map(PathBuf::from);
        let mcp_config = parser.take_option("--mcp-config")?.map(PathBuf::from);
        let enable_mcp = parser.take_flag("--enable-mcp");
        if enable_mcp && mcp_config.is_none() {
            return Err("--enable-mcp requires --mcp-config PATH".into());
        }
        let allow_network = parser.take_flag("--allow-network");
        let allow_local_network = parser.take_flag("--allow-local-network");
        let network_ports = parser.take_repeated_option("--network-port")?;
        let mut network = if allow_network {
            NetworkConfig::enabled()
        } else {
            NetworkConfig::default()
        };
        let network_domain_rules = parser.take_repeated_option("--network-domain")?;
        if (!network_domain_rules.is_empty() || !network_ports.is_empty() || allow_local_network)
            && !allow_network
        {
            return Err(
                "--network-domain, --network-port, and --allow-local-network require --allow-network"
                    .into(),
            );
        }
        network.allow_local = allow_local_network;
        for rule in network_domain_rules {
            let (pattern, access) = rule
                .rsplit_once('=')
                .ok_or_else(|| "--network-domain requires PATTERN=allow|deny".to_owned())?;
            let access = match access.to_ascii_lowercase().as_str() {
                "allow" => DomainAccess::Allow,
                "deny" => DomainAccess::Deny,
                _ => return Err("--network-domain access must be allow or deny".into()),
            };
            network
                .insert_domain_rule(pattern, access)
                .map_err(|error| error.to_string())?;
        }
        for raw_port in network_ports {
            let port = raw_port
                .parse::<u16>()
                .map_err(|error| format!("invalid --network-port `{raw_port}`: {error}"))?;
            if port == 0 {
                return Err("--network-port must be between 1 and 65535".into());
            }
            network.allow_port(port);
        }
        let sandbox_backend = match parser.take_option("--sandbox")?.as_deref() {
            None | Some("native") => SandboxBackend::Native,
            Some(value) if value.starts_with("docker:") => SandboxBackend::Docker {
                image: non_empty_suffix(value, "docker:")?,
            },
            Some(value) if value.starts_with("podman:") => SandboxBackend::Podman {
                image: non_empty_suffix(value, "podman:")?,
            },
            Some(value) => return Err(format!("invalid --sandbox `{value}`")),
        };
        Ok(Self {
            workspace,
            state,
            mcp_config,
            enable_mcp,
            sandbox_backend,
            network,
        })
    }
}

fn open_runtime(options: CommonOptions) -> Result<FocusRuntime, String> {
    let mut config = RuntimeConfig::for_workspace(options.workspace);
    config.sandbox_backend = options.sandbox_backend;
    if options.network.enabled {
        config.policy.network = PolicyDecision::RequireApproval;
    }
    config.network = options.network;
    if let Some(state) = options.state {
        config.data_root = state;
    }
    config.enable_mcp = options.enable_mcp;
    if let Some(path) = options.mcp_config {
        config.mcp_servers = load_mcp_config(&path)?;
    }
    FocusRuntime::open(config).map_err(|error| error.to_string())
}

fn non_empty_suffix(value: &str, prefix: &str) -> Result<String, String> {
    value
        .strip_prefix(prefix)
        .filter(|suffix| !suffix.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("{prefix} requires an image name"))
}

#[derive(Debug)]
enum ProviderSelection {
    Command {
        program: PathBuf,
        arguments: Vec<String>,
        timeout: Duration,
    },
    OpenAi(OpenAiOptions),
}

#[derive(Debug)]
struct OpenAiOptions {
    base_url: String,
    model: String,
    wire_api: OpenAiWireApi,
    api_key_env: String,
    headers: BTreeMap<String, String>,
    timeout: Duration,
}

impl ProviderSelection {
    fn label(&self) -> String {
        match self {
            Self::OpenAi(options) => options.model.clone(),
            Self::Command { program, .. } => format!("adapter: {}", program.display()),
        }
    }

    fn parse(parser: &mut Arguments) -> Result<Self, String> {
        let timeout = parser
            .take_option("--timeout-seconds")?
            .map(|value| -> Result<Duration, String> {
                let seconds = value
                    .parse::<u64>()
                    .map_err(|error| format!("invalid --timeout-seconds: {error}"))?;
                if seconds == 0 {
                    return Err("--timeout-seconds must be positive".into());
                }
                Ok(Duration::from_secs(seconds))
            })
            .transpose()?
            .unwrap_or(Duration::from_secs(120));
        let provider = parser.take_option("--provider")?;
        let adapter = parser.take_option("--adapter")?;
        if let Some(adapter) = adapter.as_deref()
            && provider.as_deref().is_none_or(|value| value == "command")
        {
            return Ok(Self::Command {
                program: PathBuf::from(adapter),
                arguments: parser.take_repeated_option("--adapter-arg")?,
                timeout,
            });
        }
        if let Some(provider) = provider.as_deref()
            && provider != "openai"
        {
            return Err(format!("unsupported provider `{provider}`"));
        }
        if adapter.is_some() {
            return Err("--adapter cannot be combined with --provider openai".into());
        }
        let base_url = parser
            .take_option("--base-url")?
            .or_else(|| env::var("FOCUS_BASE_URL").ok())
            .or_else(|| env::var("OPENAI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.openai.com/v1".into());
        let model = parser
            .take_option("--model")?
            .or_else(|| env::var("FOCUS_MODEL").ok())
            .or_else(|| env::var("OPENAI_MODEL").ok())
            .ok_or_else(|| "OpenAI provider requires --model MODEL".to_owned())?;
        let wire_api = parser
            .take_option("--wire-api")?
            .or_else(|| env::var("FOCUS_WIRE_API").ok())
            .as_deref()
            .map(|value| match value {
                "chat-completions" => Ok(OpenAiWireApi::ChatCompletions),
                "responses" => Ok(OpenAiWireApi::Responses),
                _ => Err("--wire-api must be chat-completions or responses".to_owned()),
            })
            .transpose()?
            .unwrap_or(OpenAiWireApi::ChatCompletions);
        let api_key_env = parser
            .take_option("--api-key-env")?
            .unwrap_or_else(|| "OPENAI_API_KEY".into());
        let mut headers = BTreeMap::new();
        for header in parser.take_repeated_option("--header")? {
            let (name, value) = header
                .split_once('=')
                .ok_or_else(|| "--header requires NAME=VALUE".to_owned())?;
            if name.is_empty() || value.is_empty() {
                return Err("--header requires non-empty NAME=VALUE".into());
            }
            headers.insert(name.into(), value.into());
        }
        Ok(Self::OpenAi(OpenAiOptions {
            base_url,
            model,
            wire_api,
            api_key_env,
            headers,
            timeout,
        }))
    }

    fn into_provider(self) -> Result<Arc<dyn ModelProvider>, String> {
        match self {
            Self::Command {
                program,
                arguments,
                timeout,
            } => Ok(Arc::new(
                CommandModelProvider::new(program, arguments).with_timeout(timeout),
            )),
            Self::OpenAi(options) => {
                let mut config = OpenAiConfig::new(options.base_url, options.model)
                    .with_wire_api(options.wire_api)
                    .with_timeout(options.timeout);
                if let Ok(api_key) = env::var(&options.api_key_env) {
                    config = config.with_api_key(api_key);
                }
                for (name, value) in options.headers {
                    config = config.with_header(name, value);
                }
                OpenAiCompatibleProvider::new(config)
                    .map(|provider| Arc::new(provider) as Arc<dyn ModelProvider>)
                    .map_err(|error| error.to_string())
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct McpConfigDocument {
    servers: Vec<McpServerDocument>,
}

#[derive(Debug, Deserialize)]
struct McpServerDocument {
    name: String,
    command: PathBuf,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    current_dir: Option<PathBuf>,
    timeout_ms: Option<u64>,
    stderr_limit: Option<usize>,
    response_limit: Option<usize>,
    operation: Option<ToolOperation>,
}

fn load_mcp_config(path: &PathBuf) -> Result<Vec<McpServerConfig>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read MCP config {}: {error}", path.display()))?;
    let document: McpConfigDocument = serde_json::from_str(&content)
        .map_err(|error| format!("invalid MCP config {}: {error}", path.display()))?;
    document
        .servers
        .into_iter()
        .map(|server| {
            let mut config = McpServerConfig::new(server.name, server.command)
                .with_args(server.args)
                .with_env(server.env)
                .with_timeout(Duration::from_millis(server.timeout_ms.unwrap_or(30_000)));
            if let Some(operation) = server.operation {
                config = config.with_operation(operation);
            }
            if let Some(current_dir) = server.current_dir {
                config = config.with_current_dir(current_dir);
            }
            if let Some(stderr_limit) = server.stderr_limit {
                config.stderr_limit = stderr_limit;
            }
            if let Some(response_limit) = server.response_limit {
                config.response_limit = response_limit;
            }
            Ok(config)
        })
        .collect()
}

fn usage() -> String {
    "Usage:\n  focus [chat options]                         Start Focus in the current directory\n  focus chat [chat options]                    Explicit chat command\n  focus-harness <command> [options]             Compatibility alias\n  focus-harness run [--workspace PATH] [--state PATH] [--mcp-config PATH --enable-mcp] [--sandbox native|docker:IMAGE|podman:IMAGE] [--allow-network [--network-domain PATTERN=allow|deny]... [--network-port PORT]... [--allow-local-network]] [--provider openai --base-url URL --model MODEL --wire-api chat-completions|responses --api-key-env NAME | --adapter PROGRAM [--adapter-arg ARG]...] [--yolo|--interactive|--deny] [--session SESSION_ID] [--goal GOAL_ID] [--title TITLE] -- TASK\n  focus-harness chat [--workspace PATH] [--state PATH] [--mcp-config PATH --enable-mcp] [--sandbox BACKEND] [--allow-network [--network-domain PATTERN=allow|deny]... [--network-port PORT]... [--allow-local-network]] [--provider openai --base-url URL --model MODEL --wire-api chat-completions|responses --api-key-env NAME | --adapter PROGRAM [--adapter-arg ARG]...] [--yolo|--interactive|--deny] [--session SESSION_ID | --new-session SESSION_ID] [--title TITLE] [--plain]\n  focus-harness demo [--workspace PATH] [--state PATH] [--mcp-config PATH --enable-mcp] [--sandbox native|docker:IMAGE|podman:IMAGE] [--allow-network [--network-domain PATTERN=allow|deny]... [--network-port PORT]... [--allow-local-network]] [--title TITLE] -- TASK\n  focus-harness host --stdio [--workspace PATH] [--state PATH]\n  focus-harness doctor [--workspace PATH] [--state PATH] [--mcp-config PATH --enable-mcp] [--allow-network [--network-domain PATTERN=allow|deny]... [--network-port PORT]... [--allow-local-network]] [--live-containers]\n  focus-harness benchmark [--runs N] [--output PATH]\n  focus-harness session list|show|replay|fork [--workspace PATH] [--state PATH] ...\n  focus-harness memory add|list [--workspace PATH] [--state PATH] [--scope project|session --session SESSION_ID] [--source SOURCE] -- CONTENT\n  focus-harness subagent list [--workspace PATH] [--state PATH] --session SESSION_ID\n  focus-harness goal create|list|show|attach|complete|cancel [--workspace PATH] [--state PATH] ...".into()
}

fn print_usage() {
    println!(
        "{}\n  focus self-review [--workspace PATH] [--output PATH]\n  focus self-update stage [--workspace PATH] [--version VERSION] [--commit COMMIT] [--output PATH]\n  focus self-update apply|rollback [--output PATH]",
        usage()
    );
}

#[cfg(test)]
mod tests {
    use std::io;

    use focus_runtime::{
        interaction::ApprovalBroker,
        policy::{ApprovalHandler, ApprovalRequest, ToolOperation},
        subagent::CancellationToken,
    };
    use serde_json::json;

    use super::*;

    #[test]
    fn bare_focus_defaults_to_chat() {
        assert_eq!(normalize_command(Vec::new()), ("chat".into(), Vec::new()));
    }

    #[test]
    fn focus_options_are_forwarded_to_chat() {
        let arguments = vec!["--model".into(), "MODEL".into(), "--plain".into()];
        assert_eq!(
            normalize_command(arguments.clone()),
            ("chat".into(), arguments)
        );
    }

    #[test]
    fn explicit_commands_and_help_are_preserved() {
        for command in ["chat", "run", "doctor", "help", "--help", "-h"] {
            let arguments = vec![command.into(), "--plain".into()];
            assert_eq!(
                normalize_command(arguments.clone()),
                (command.into(), vec!["--plain".into()])
            );
        }
    }

    #[test]
    fn host_request_parser_accepts_v1_and_rejects_other_protocols() {
        let session_id = Uuid::new_v4();
        let request = parse_host_request(&format!(
            r#"{{"protocol":"focus-host-v1","operation":"attach","session_id":"{session_id}"}}"#
        ))
        .unwrap();

        assert_eq!(request.session_id(), session_id);
        assert_eq!(request.after_event_id(), None);
        assert!(parse_host_request(r#"{"protocol":"focus-host-v2","operation":"attach","session_id":"00000000-0000-0000-0000-000000000000"}"#)
            .unwrap_err()
            .contains("unsupported host protocol"));
        assert!(
            parse_host_request(
                r#"{"operation":"attach","session_id":"00000000-0000-0000-0000-000000000000"}"#
            )
            .unwrap_err()
            .contains("missing field `protocol`")
        );
    }

    #[test]
    fn host_stream_continues_after_a_terminal_turn_when_following() {
        let session_id = Uuid::new_v4();
        let request = parse_host_request(&format!(
            r#"{{"protocol":"focus-host-v1","operation":"attach","session_id":"{session_id}","follow":true}}"#
        ))
        .unwrap();
        let terminal = Event::now(
            session_id,
            EventKind::StateChanged {
                status: focus_kernel::AgentStatus::Complete,
                turn: 1,
            },
        );

        assert!(request.follow());
        assert!(!host_attachment_should_close(&request, &terminal));
    }

    #[test]
    fn host_attachment_emits_the_canonical_replay_as_jsonl() {
        let directory = std::env::temp_dir().join(format!("focus-cli-host-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let runtime = FocusRuntime::open(RuntimeConfig::for_workspace(&directory)).unwrap();
        let result = runtime
            .run(
                "host output",
                "return fixture output",
                Arc::new(StaticProvider::new("done")),
                deny_approvals(),
            )
            .unwrap();
        let request = parse_host_request(&format!(
            r#"{{"protocol":"focus-host-v1","operation":"attach","session_id":"{}"}}"#,
            result.session_id
        ))
        .unwrap();
        let mut output = Vec::new();

        stream_host_attachment(&runtime, request, &mut output).unwrap();

        let lines = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        let replay = runtime.replay(result.session_id).unwrap();
        assert_eq!(lines.len(), replay.len());
        assert!(
            lines
                .iter()
                .all(|frame| frame["protocol"] == "focus-host-v1")
        );
        assert!(lines.iter().all(|frame| frame["type"] == "event"));
        assert_eq!(
            lines
                .iter()
                .map(|frame| frame["event"]["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            replay
                .iter()
                .map(|event| event.id.to_string())
                .collect::<Vec<_>>()
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn parses_options_out_of_order() {
        let mut arguments = Arguments::new(vec![
            "task".into(),
            "--state".into(),
            "data".into(),
            "--workspace".into(),
            "project".into(),
        ]);
        let options = CommonOptions::parse(&mut arguments).unwrap();
        assert_eq!(options.workspace, PathBuf::from("project"));
        assert_eq!(options.state, Some(PathBuf::from("data")));
        assert_eq!(arguments.remaining_task().unwrap(), "task");
    }

    #[test]
    fn parser_treats_options_after_double_dash_as_task_text() {
        let mut arguments =
            Arguments::new(vec!["--".into(), "--title".into(), "literal task".into()]);

        assert_eq!(arguments.take_option("--title").unwrap(), None);
        assert_eq!(arguments.remaining_task().unwrap(), "--title literal task");
    }

    #[test]
    fn parser_rejects_unparsed_options_before_double_dash() {
        let arguments = Arguments::new(vec!["--workflow".into(), "task".into()]);

        assert_eq!(
            arguments.remaining_task().unwrap_err(),
            "unexpected arguments: --workflow"
        );
    }

    #[test]
    fn cli_renderer_outputs_visible_model_deltas_without_terminal_payloads() {
        let event = Event::now(
            Uuid::nil(),
            EventKind::Model {
                event: ModelEvent::TextDelta {
                    text: "first visible delta".into(),
                },
            },
        );
        let terminal = Event::now(
            Uuid::nil(),
            EventKind::Model {
                event: ModelEvent::Completed {
                    response: focus_kernel::ModelResponse {
                        content: "final payload".into(),
                        tool_calls: Vec::new(),
                    },
                },
            },
        );
        let mut output = Vec::new();
        let visible_text = Mutex::new(String::new());

        render_model_event(&event, &mut output, &visible_text).unwrap();
        render_model_event(&terminal, &mut output, &visible_text).unwrap();

        assert_eq!(String::from_utf8(output).unwrap(), "first visible delta");
        assert_eq!(visible_text.into_inner().unwrap(), "first visible delta");
    }

    #[test]
    fn cli_renderer_redacts_model_delta_secrets() {
        let event = Event::now(
            Uuid::nil(),
            EventKind::Model {
                event: ModelEvent::TextDelta {
                    text: "token=cli-renderer-secret".into(),
                },
            },
        );
        let mut output = Vec::new();
        let visible_text = Mutex::new(String::new());

        render_model_event(&event, &mut output, &visible_text).unwrap();

        let output = String::from_utf8(output).unwrap();
        assert!(!output.contains("cli-renderer-secret"));
        assert!(
            !visible_text
                .into_inner()
                .unwrap()
                .contains("cli-renderer-secret")
        );
        assert_eq!(output, "token=<redacted>");
    }

    #[test]
    fn final_response_is_not_repeated_after_streaming() {
        assert_eq!(
            format_final_response("first visible delta", "first visible delta"),
            ""
        );
        assert_eq!(
            format_final_response("first visible delta and more", "first visible delta"),
            " and more"
        );
        assert_eq!(format_final_response("final answer", ""), "final answer");
    }

    #[test]
    fn approval_prompt_is_compact_and_does_not_dump_raw_arguments() {
        let pending = ApprovalEnvelope {
            id: Uuid::nil(),
            request: ApprovalRequest {
                tool: "shell".into(),
                operation: ToolOperation::Execute,
                arguments: json!({"command": "dir", "secret": "redacted"}),
                rationale: "Build, test, inspect, or format the current project.".into(),
            },
        };
        let mut reader = io::Cursor::new(b"n\n".to_vec());
        let mut output = Vec::new();

        let decision = prompt_for_approval(&pending, &mut reader, &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();

        assert_eq!(decision, ApprovalDecision::Deny);
        assert!(output.contains("Permission required"));
        assert!(output.contains("shell (execute)"));
        assert!(output.contains("command: dir"));
        assert!(!output.contains("arguments:"));
        assert!(!output.contains("secret"));
    }

    #[test]
    fn background_renderer_consumes_live_model_deltas() {
        #[derive(Clone)]
        struct SharedWriter(Arc<std::sync::Mutex<Vec<u8>>>);

        impl Write for SharedWriter {
            fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
                self.0
                    .lock()
                    .map_err(|_| io::Error::other("test writer lock poisoned"))?
                    .extend_from_slice(buffer);
                Ok(buffer.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let output = Arc::new(std::sync::Mutex::new(Vec::new()));
        let renderer = spawn_event_renderer(receiver, SharedWriter(output.clone()));
        sender
            .send(Event::now(
                Uuid::nil(),
                EventKind::Model {
                    event: ModelEvent::TextDelta {
                        text: "streamed before completion".into(),
                    },
                },
            ))
            .unwrap();
        drop(sender);

        let streamed = renderer.finish().unwrap();

        assert_eq!(
            String::from_utf8(output.lock().unwrap().clone()).unwrap(),
            "streamed before completion"
        );
        assert_eq!(streamed, "streamed before completion");
    }

    #[test]
    fn renderer_keeps_only_the_current_model_turn_for_final_deduplication() {
        #[derive(Clone)]
        struct SharedWriter(Arc<std::sync::Mutex<Vec<u8>>>);

        impl Write for SharedWriter {
            fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
                self.0
                    .lock()
                    .map_err(|_| io::Error::other("test writer lock poisoned"))?
                    .extend_from_slice(buffer);
                Ok(buffer.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let (sender, receiver) = std::sync::mpsc::sync_channel(4);
        let output = Arc::new(std::sync::Mutex::new(Vec::new()));
        let renderer = spawn_event_renderer(receiver, SharedWriter(output));
        for event in [
            ModelEvent::RequestStarted {
                provider: "fixture".into(),
                model: "fixture".into(),
                endpoint: "fixture://provider".into(),
            },
            ModelEvent::TextDelta {
                text: "I will inspect the project.\n".into(),
            },
            ModelEvent::RequestStarted {
                provider: "fixture".into(),
                model: "fixture".into(),
                endpoint: "fixture://provider".into(),
            },
            ModelEvent::TextDelta {
                text: "The project root contains Agent.md.".into(),
            },
        ] {
            sender
                .send(Event::now(Uuid::nil(), EventKind::Model { event }))
                .unwrap();
        }
        drop(sender);

        let streamed = renderer.finish().unwrap();

        assert_eq!(streamed, "The project root contains Agent.md.");
    }

    #[test]
    fn renderer_summarizes_tool_activity_without_exposing_payloads() {
        let call = focus_kernel::ToolCall {
            id: "call-1".into(),
            name: "shell".into(),
            arguments: json!({"command": "dir", "resource_limits": {"memory": 10485760}}),
        };
        let result = focus_kernel::ToolResult {
            tool_call_id: call.id.clone(),
            name: call.name.clone(),
            content: "full command output must stay in the event log".into(),
            is_error: false,
        };
        let mut output = Vec::new();
        let visible_text = Mutex::new(String::new());

        render_model_event(
            &Event::now(
                Uuid::nil(),
                EventKind::Model {
                    event: ModelEvent::ToolCallStarted {
                        id: call.id.clone(),
                        name: call.name.clone(),
                    },
                },
            ),
            &mut output,
            &visible_text,
        )
        .unwrap();
        render_model_event(
            &Event::now(Uuid::nil(), EventKind::ToolCallRequested { call }),
            &mut output,
            &visible_text,
        )
        .unwrap();
        render_model_event(
            &Event::now(Uuid::nil(), EventKind::ToolResultReceived { result }),
            &mut output,
            &visible_text,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();

        assert!(output.contains("- Using shell: dir"));
        assert!(output.contains("- shell finished"));
        assert!(!output.contains("Running shell"));
        assert!(!output.contains("resource_limits"));
        assert!(!output.contains("full command output"));
    }

    #[test]
    fn tool_argument_preview_redacts_credentials() {
        let preview = tool_argument_preview(&json!({
            "command": "curl -H 'Authorization: Bearer top-secret' https://example.com/?token=top-secret"
        }));

        assert!(preview.contains("<redacted>"));
        assert!(!preview.contains("top-secret"));
    }

    #[test]
    fn chat_surface_exposes_a_prompt_and_minimal_commands() {
        assert!(chat_welcome().contains("/help"));
        assert!(chat_help().contains("/quit"));
        assert_eq!(chat_prompt(), "focus> ");
        assert_eq!(parse_chat_command("/help"), Some(ChatCommand::Help));
        assert_eq!(parse_chat_command("/review"), Some(ChatCommand::Review));
        assert_eq!(parse_chat_command("/simplify"), Some(ChatCommand::Simplify));
        assert_eq!(parse_chat_command("/exit"), Some(ChatCommand::Exit));
        assert_eq!(parse_chat_command("/quit"), Some(ChatCommand::Exit));
        assert_eq!(parse_chat_command("implement the change"), None);
    }

    #[test]
    fn chat_input_routes_commands_without_submitting_them_to_the_agent() {
        assert_eq!(classify_chat_input("/help"), ChatInput::Help);
        assert_eq!(
            classify_chat_input("/review"),
            ChatInput::Task(REVIEW_TASK.into())
        );
        assert_eq!(
            classify_chat_input("/simplify"),
            ChatInput::Task(SIMPLIFY_TASK.into())
        );
        assert_eq!(classify_chat_input("/exit"), ChatInput::Exit);
        assert_eq!(
            classify_chat_input("implement the change"),
            ChatInput::Task("implement the change".into())
        );
    }

    #[test]
    fn usage_documents_the_default_production_profile() {
        let usage = usage();
        assert!(!usage.contains("[--workflow]"));
        assert!(!usage.contains("[--delegate]"));
        assert!(usage.contains("host --stdio"));
        assert!(usage.contains("benchmark [--runs N] [--output PATH]"));
        assert!(!usage.contains("--diagnostic"));
    }

    #[test]
    fn self_update_options_are_order_independent_and_bounded() {
        let workspace = std::env::current_dir().unwrap();
        let mut arguments = Arguments::new(vec![
            "--version".into(),
            "v-test".into(),
            "--output".into(),
            "receipt.json".into(),
            "--workspace".into(),
            workspace.display().to_string(),
            "--commit".into(),
            "abc123".into(),
        ]);

        let options = parse_update_options(&mut arguments).unwrap();

        assert_eq!(options.version.as_deref(), Some("v-test"));
        assert_eq!(options.commit.as_deref(), Some("abc123"));
        assert_eq!(options.output, Some(PathBuf::from("receipt.json")));
        assert_eq!(options.workspace, workspace);
    }

    #[test]
    fn self_update_builds_outside_the_running_target_directory() {
        let workspace = std::env::current_dir().unwrap();
        let target = self_update_target_dir(&workspace);
        assert!(target.starts_with(workspace.join("target")));
        assert_ne!(target, workspace.join("target"));
        assert_eq!(
            release_executable_at(&target),
            target
                .join("release")
                .join(format!("focus{}", std::env::consts::EXE_SUFFIX))
        );
    }

    #[test]
    fn self_review_receipt_round_trips_bounded_check_evidence() {
        let receipt = SelfReviewReceipt {
            schema_version: 1,
            workspace: "workspace".into(),
            commit: Some("abc".into()),
            checks: vec![ReviewCheck {
                name: "fmt".into(),
                command: "cargo fmt --all -- --check".into(),
                passed: true,
                output: "ok".into(),
            }],
            artifact_path: Some("target/release/focus.exe".into()),
            artifact_sha256: Some("0".repeat(64)),
            artifact_size: Some(42),
            passed: true,
        };

        let encoded = serde_json::to_string(&receipt).unwrap();
        let decoded: SelfReviewReceipt = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded.checks[0].command, receipt.checks[0].command);
        assert_eq!(decoded.artifact_size, Some(42));
        assert!(decoded.passed);
    }

    #[test]
    fn benchmark_options_parse_positive_runs_and_output() {
        let mut arguments = Arguments::new(vec![
            "--output".into(),
            "receipt.json".into(),
            "--runs".into(),
            "3".into(),
        ]);

        let options = BenchmarkOptions::parse(&mut arguments).unwrap();

        assert_eq!(options.runs, 3);
        assert_eq!(options.output, Some(PathBuf::from("receipt.json")));
        arguments.ensure_empty().unwrap();
    }

    #[test]
    fn benchmark_options_reject_zero_runs() {
        let mut arguments = Arguments::new(vec!["--runs".into(), "0".into()]);

        assert!(BenchmarkOptions::parse(&mut arguments).is_err());
    }

    #[test]
    fn benchmark_gate_includes_failure_probes() {
        let mut receipt = run_deterministic(1).unwrap();
        assert!(benchmark_passed(&receipt));
        receipt.failure_probes[0].passed = false;
        assert!(!benchmark_passed(&receipt));
    }

    #[test]
    fn benchmark_gate_includes_cancellation_probes() {
        let mut receipt = run_deterministic(1).unwrap();
        assert!(benchmark_passed(&receipt));
        receipt.cancellation_probes[0].passed = false;
        assert!(!benchmark_passed(&receipt));
    }

    #[test]
    fn receipt_replacement_is_atomic_and_cleans_backup() {
        let directory = std::env::temp_dir().join(format!("focus-receipt-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let receipt = directory.join("receipt.json");
        std::fs::write(&receipt, "old").unwrap();

        write_receipt(&receipt, "new").unwrap();

        assert_eq!(std::fs::read_to_string(&receipt).unwrap(), "new");
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn receipt_publish_failure_restores_previous_receipt() {
        let directory =
            std::env::temp_dir().join(format!("focus-receipt-rollback-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let receipt = directory.join("receipt.json");
        let missing_temporary = directory.join("missing.tmp");
        let backup = directory.join("receipt.bak");
        std::fs::write(&receipt, "old").unwrap();

        let error = publish_receipt(&missing_temporary, &receipt, &backup).unwrap_err();

        assert!(error.contains("previous receipt restored"));
        assert_eq!(std::fs::read_to_string(&receipt).unwrap(), "old");
        assert!(!backup.exists());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn benchmark_command_rejects_unknown_options_before_execution() {
        let error = run(vec!["benchmark".into(), "--unknown".into()]).unwrap_err();

        assert!(error.contains("unexpected arguments: --unknown"));
    }

    #[test]
    fn cli_defaults_to_the_combined_production_profile() {
        let arguments = Arguments::new(vec!["plain text".into()]);

        let capabilities = RunCapabilities::production();
        let options = capabilities.options(approve_all());
        assert!(options.workflow.is_some());
        assert!(options.delegation.is_some());
        assert_eq!(arguments.remaining_task().unwrap(), "plain text");
    }

    #[test]
    fn run_session_target_accepts_an_existing_session_id() {
        let id = Uuid::new_v4();
        let mut arguments = Arguments::new(vec!["--session".into(), id.to_string(), "task".into()]);

        assert_eq!(RunSession::parse(&mut arguments).unwrap(), Some(id));
        assert_eq!(arguments.remaining_task().unwrap(), "task");
    }

    #[test]
    fn chat_session_target_accepts_a_host_allocated_id_and_rejects_ambiguity() {
        let id = Uuid::new_v4();
        let mut arguments = Arguments::new(vec!["--new-session".into(), id.to_string()]);

        assert_eq!(
            ChatSessionTarget::parse(&mut arguments).unwrap(),
            ChatSessionTarget {
                session: None,
                new_session: Some(id),
            }
        );

        let mut ambiguous = Arguments::new(vec![
            "--session".into(),
            Uuid::new_v4().to_string(),
            "--new-session".into(),
            id.to_string(),
        ]);
        assert!(
            ChatSessionTarget::parse(&mut ambiguous)
                .unwrap_err()
                .contains("mutually exclusive")
        );
    }

    #[test]
    fn handoff_does_not_recreate_a_new_session_id() {
        let directory =
            std::env::temp_dir().join(format!("focus-chat-handoff-session-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let runtime = FocusRuntime::open(RuntimeConfig::for_workspace(&directory)).unwrap();
        let session_id = Uuid::new_v4();
        let target = ChatSessionTarget {
            session: None,
            new_session: Some(session_id),
        };

        assert_eq!(
            resolve_chat_session(&runtime, target, "chat", true).unwrap(),
            None
        );
        assert!(runtime.replay(session_id).is_err());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn chat_session_reuses_the_first_runtime_session_for_followup_turns() {
        let directory = std::env::temp_dir().join(format!("focus-chat-session-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut config = RuntimeConfig::for_workspace(&directory);
        config.data_root = directory.join("state");
        let runtime = FocusRuntime::open(config).unwrap();
        let mut chat = ChatSession::default();
        let provider = Arc::new(StaticProvider::new("done"));

        let first = chat
            .run_turn(
                &runtime,
                "chat",
                "first turn",
                provider.clone(),
                RunCapabilities::core(),
                deny_approvals(),
            )
            .unwrap();
        let second = chat
            .run_turn(
                &runtime,
                "chat",
                "follow up",
                provider,
                RunCapabilities::core(),
                deny_approvals(),
            )
            .unwrap();

        assert_eq!(first.session_id, second.session_id);
        assert_eq!(runtime.list_sessions().unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn chat_continues_after_a_stream_failure_in_the_same_session() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Default)]
        struct FlakyProvider {
            calls: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl ModelProvider for FlakyProvider {
            async fn stream(
                &self,
                _request: focus_kernel::ModelRequest,
                _cancellation: &dyn focus_kernel::CancellationSignal,
            ) -> Result<focus_kernel::ModelEventStream, focus_kernel::KernelError> {
                let events = if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    vec![
                        ModelEvent::RequestStarted {
                            provider: "fixture".into(),
                            model: "fixture".into(),
                            endpoint: "fixture://stream".into(),
                        },
                        ModelEvent::TextDelta {
                            text: "partial response".into(),
                        },
                        ModelEvent::Failed {
                            error: "response read failed".into(),
                        },
                    ]
                } else {
                    vec![
                        ModelEvent::RequestStarted {
                            provider: "fixture".into(),
                            model: "fixture".into(),
                            endpoint: "fixture://stream".into(),
                        },
                        ModelEvent::TextDelta {
                            text: "recovered response".into(),
                        },
                        ModelEvent::Completed {
                            response: focus_kernel::ModelResponse {
                                content: "recovered response".into(),
                                tool_calls: Vec::new(),
                            },
                        },
                    ]
                };
                Ok(Box::pin(futures::stream::iter(events.into_iter().map(Ok))))
            }
        }

        let directory =
            std::env::temp_dir().join(format!("focus-chat-recovery-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut config = RuntimeConfig::for_workspace(&directory);
        config.data_root = directory.join("state");
        let runtime = FocusRuntime::open(config).unwrap();
        let provider = Arc::new(FlakyProvider::default());
        let mut reader = io::Cursor::new(b"first task\nsecond task\n/exit\n".to_vec());

        run_chat_with_approval_reader(
            runtime.clone(),
            None,
            "chat".into(),
            provider.clone(),
            RunCapabilities::core(),
            deny_approvals(),
            &mut reader,
        )
        .unwrap();

        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        let sessions = runtime.list_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(
            sessions[0].phase,
            focus_runtime::session::SessionPhase::Complete
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn interactive_chat_cancellation_is_scoped_to_one_turn() {
        let cancelled_turn = new_chat_cancellation();
        cancelled_turn.cancel();
        let next_turn = new_chat_cancellation();

        assert!(cancelled_turn.is_cancelled());
        assert!(!next_turn.is_cancelled());
    }

    #[test]
    fn run_goal_target_accepts_an_existing_goal_id() {
        let id = Uuid::new_v4();
        let mut arguments = Arguments::new(vec!["--goal".into(), id.to_string(), "task".into()]);

        assert_eq!(RunGoal::parse(&mut arguments).unwrap(), Some(id));
        assert_eq!(arguments.remaining_task().unwrap(), "task");
    }

    #[test]
    fn memory_scope_requires_a_session_id_for_session_memory() {
        let mut arguments = Arguments::new(vec!["--scope".into(), "session".into()]);

        assert!(MemorySelection::parse(&mut arguments).is_err());
    }

    #[test]
    fn subagent_event_filter_uses_the_canonical_runtime_lifecycle() {
        let session_id = Uuid::new_v4();
        let events = vec![
            Event::now(
                session_id,
                EventKind::Runtime {
                    name: "subagent_started".into(),
                    data: json!({"child_session_id": Uuid::new_v4()}),
                },
            ),
            Event::now(
                session_id,
                EventKind::Runtime {
                    name: "workflow_started".into(),
                    data: json!({}),
                },
            ),
        ];

        let lifecycle = subagent_lifecycle_events(&events).collect::<Vec<_>>();

        assert_eq!(lifecycle.len(), 1);
        assert_eq!(lifecycle[0].kind, "subagent_started");
    }

    #[test]
    fn parses_direct_openai_provider_options() {
        let mut arguments = Arguments::new(vec![
            "--provider".into(),
            "openai".into(),
            "--base-url".into(),
            "http://localhost:1234/v1".into(),
            "--wire-api".into(),
            "responses".into(),
            "--model".into(),
            "gpt-test".into(),
            "--api-key-env".into(),
            "TEST_API_KEY".into(),
            "--header".into(),
            "x-project=demo".into(),
        ]);

        let selection = ProviderSelection::parse(&mut arguments).unwrap();

        match selection {
            ProviderSelection::OpenAi(options) => {
                assert_eq!(options.base_url, "http://localhost:1234/v1");
                assert_eq!(options.model, "gpt-test");
                assert_eq!(options.wire_api, OpenAiWireApi::Responses);
                assert_eq!(options.api_key_env, "TEST_API_KEY");
                assert_eq!(options.headers["x-project"], "demo");
            }
            ProviderSelection::Command { .. } => panic!("expected OpenAI provider"),
        }
    }

    #[test]
    fn command_provider_uses_the_shared_timeout_option() {
        let mut arguments = Arguments::new(vec![
            "--adapter".into(),
            "fixture-adapter".into(),
            "--timeout-seconds".into(),
            "7".into(),
        ]);

        let selection = ProviderSelection::parse(&mut arguments).unwrap();

        assert!(matches!(
            selection,
            ProviderSelection::Command { timeout, .. } if timeout == Duration::from_secs(7)
        ));
    }

    #[test]
    fn common_options_accept_an_mcp_config_path() {
        let mut arguments = Arguments::new(vec![
            "--mcp-config".into(),
            "mcp.json".into(),
            "task".into(),
        ]);

        let options = CommonOptions::parse(&mut arguments).unwrap();

        assert_eq!(options.mcp_config, Some(PathBuf::from("mcp.json")));
        assert!(!options.enable_mcp);
    }

    #[test]
    fn common_options_require_explicit_mcp_enablement_with_a_config() {
        let mut enabled = Arguments::new(vec![
            "--mcp-config".into(),
            "mcp.json".into(),
            "--enable-mcp".into(),
            "task".into(),
        ]);
        let mut missing_config = Arguments::new(vec!["--enable-mcp".into(), "task".into()]);

        assert!(CommonOptions::parse(&mut enabled).unwrap().enable_mcp);
        assert!(CommonOptions::parse(&mut missing_config).is_err());
    }

    #[test]
    fn common_options_enable_bounded_network_and_domain_rules() {
        let mut arguments = Arguments::new(vec![
            "--allow-network".into(),
            "--allow-local-network".into(),
            "--network-domain".into(),
            "*.example.com=allow".into(),
            "--network-domain".into(),
            "api.example.com=deny".into(),
            "--network-port".into(),
            "8443".into(),
            "--network-port".into(),
            "8443".into(),
            "task".into(),
        ]);

        let options = CommonOptions::parse(&mut arguments).unwrap();

        assert!(options.network.enabled);
        assert!(options.network.allow_local);
        assert_eq!(
            options.network.domains.get("*.example.com"),
            Some(&DomainAccess::Allow)
        );
        assert_eq!(
            options.network.domains.get("api.example.com"),
            Some(&DomainAccess::Deny)
        );
        assert_eq!(options.network.allowed_ports, vec![8443]);
        assert_eq!(arguments.remaining_task().unwrap(), "task");
    }

    #[test]
    fn local_network_flag_requires_network_flag() {
        let mut arguments = Arguments::new(vec!["--allow-local-network".into(), "task".into()]);

        assert!(CommonOptions::parse(&mut arguments).is_err());
    }

    #[test]
    fn network_domain_rules_require_network_flag() {
        let mut arguments = Arguments::new(vec![
            "--network-domain".into(),
            "example.com=allow".into(),
            "task".into(),
        ]);

        assert!(CommonOptions::parse(&mut arguments).is_err());
    }

    #[test]
    fn network_port_requires_network_and_must_be_valid() {
        let mut without_network =
            Arguments::new(vec!["--network-port".into(), "8443".into(), "task".into()]);
        assert!(CommonOptions::parse(&mut without_network).is_err());

        let mut zero = Arguments::new(vec![
            "--allow-network".into(),
            "--network-port".into(),
            "0".into(),
            "task".into(),
        ]);
        assert!(CommonOptions::parse(&mut zero).is_err());
    }

    #[test]
    fn loads_every_mcp_server_field_including_operation() {
        let directory = std::env::temp_dir().join(format!("cli-mcp-config-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("mcp.json");
        std::fs::write(
            &path,
            r#"{
                "servers": [{
                    "name": "fixture",
                    "command": "fixture-command",
                    "args": ["--stdio"],
                    "env": {"KEY": "value"},
                    "current_dir": "fixture-workdir",
                    "timeout_ms": 1234,
                    "stderr_limit": 2345,
                    "response_limit": 3456,
                    "operation": "write"
                }]
            }"#,
        )
        .unwrap();

        let configs = load_mcp_config(&path).unwrap();
        let config = &configs[0];

        assert_eq!(config.name, "fixture");
        assert_eq!(config.command, PathBuf::from("fixture-command"));
        assert_eq!(config.args, vec!["--stdio"]);
        assert_eq!(config.env["KEY"], "value");
        assert_eq!(config.current_dir, Some(PathBuf::from("fixture-workdir")));
        assert_eq!(config.request_timeout, Duration::from_millis(1234));
        assert_eq!(config.stderr_limit, 2345);
        assert_eq!(config.response_limit, 3456);
        assert_eq!(config.operation, ToolOperation::Write);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn rejects_invalid_mcp_json_and_operation() {
        let directory = std::env::temp_dir().join(format!("cli-mcp-invalid-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let invalid_json = directory.join("invalid-json.json");
        let invalid_operation = directory.join("invalid-operation.json");
        std::fs::write(&invalid_json, "{not-json").unwrap();
        std::fs::write(
            &invalid_operation,
            r#"{"servers":[{"name":"fixture","command":"fixture","operation":"delete_everything"}]}"#,
        )
        .unwrap();

        assert!(load_mcp_config(&invalid_json).is_err());
        assert!(load_mcp_config(&invalid_operation).is_err());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn common_options_select_a_container_sandbox() {
        let mut arguments = Arguments::new(vec![
            "--sandbox".into(),
            "docker:rust:latest".into(),
            "task".into(),
        ]);

        let options = CommonOptions::parse(&mut arguments).unwrap();

        assert!(matches!(
            options.sandbox_backend,
            focus_runtime::sandbox::SandboxBackend::Docker { ref image }
                if image == "rust:latest"
        ));
    }

    #[test]
    fn approval_mode_defaults_to_yolo() {
        let mut arguments = Arguments::new(vec!["task".into()]);

        assert_eq!(
            ApprovalMode::parse(&mut arguments).unwrap(),
            ApprovalMode::ApproveAll
        );
    }

    #[test]
    fn approval_mode_supports_noninteractive_overrides() {
        let mut approve = Arguments::new(vec!["--approve".into(), "task".into()]);
        let mut yolo = Arguments::new(vec!["--yolo".into(), "task".into()]);
        let mut interactive = Arguments::new(vec!["--interactive".into(), "task".into()]);
        let mut deny = Arguments::new(vec!["--deny".into(), "task".into()]);
        let mut conflicting =
            Arguments::new(vec!["--approve".into(), "--deny".into(), "task".into()]);

        assert_eq!(
            ApprovalMode::parse(&mut approve).unwrap(),
            ApprovalMode::ApproveAll
        );
        assert_eq!(
            ApprovalMode::parse(&mut yolo).unwrap(),
            ApprovalMode::ApproveAll
        );
        assert_eq!(
            ApprovalMode::parse(&mut interactive).unwrap(),
            ApprovalMode::Interactive
        );
        assert_eq!(
            ApprovalMode::parse(&mut deny).unwrap(),
            ApprovalMode::DenyAll
        );
        assert!(ApprovalMode::parse(&mut conflicting).is_err());
    }

    #[test]
    fn parses_interactive_approval_decisions() {
        assert_eq!(
            parse_approval_decision("yes"),
            Some(ApprovalDecision::ApproveOnce)
        );
        assert_eq!(
            parse_approval_decision("S"),
            Some(ApprovalDecision::ApproveSession)
        );
        assert_eq!(parse_approval_decision(""), Some(ApprovalDecision::Deny));
        assert_eq!(parse_approval_decision("maybe"), None);
    }

    struct FailingReader;

    impl io::Read for FailingReader {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("input failed"))
        }
    }

    #[test]
    fn input_error_cancels_and_joins_the_approval_worker() {
        let (broker, inbox) = ApprovalBroker::new();
        let cancellation = CancellationToken::default();
        let worker_cancellation = cancellation.clone();
        let worker = thread::spawn(move || {
            broker
                .request(
                    &ApprovalRequest {
                        tool: "shell".into(),
                        operation: ToolOperation::Execute,
                        arguments: json!({"command":"cargo test"}),
                        rationale: "verify".into(),
                    },
                    &worker_cancellation,
                )
                .map_err(|error| error.to_string())
        });
        let mut reader = io::BufReader::new(FailingReader);

        let interaction = drive_approval_inbox(&inbox, &worker, &mut reader);
        let result = cancel_and_join(worker, inbox, cancellation.clone(), interaction);

        assert!(result.is_err());
        assert!(cancellation.is_cancelled());
    }
}
