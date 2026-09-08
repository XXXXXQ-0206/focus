//! Full-screen terminal host for interactive Focus chat.
//!
//! This is a projection over EventHub. Runtime remains the only owner of
//! transcript, tools, approvals, persistence, and cancellation semantics.

mod frame_requester;
mod markdown;
mod slash;

use std::{
    collections::{BTreeMap, HashSet, VecDeque, hash_map::DefaultHasher},
    env, fs,
    hash::{Hash, Hasher},
    io::{self, IsTerminal, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};

use crossterm::{
    cursor::{Hide, Show},
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event as TerminalEvent, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEventKind,
    },
    execute,
    style::Colored,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use focus_kernel::{
    AgentStatus, Event, EventKind, ModelEvent, ModelProvider, Role, redact_sensitive_text,
};
use focus_runtime::{
    FocusRuntime,
    interaction::{ApprovalBroker, ApprovalEnvelope, ApprovalInbox},
    policy::{ApprovalDecision, ApprovalHandler, ToolOperation},
    subagent::CancellationToken,
    update::{VersionedArtifactStore, default_update_root},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line as RatLine, Span, Text},
    widgets::{Block, Borders, Paragraph},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;
use uuid::Uuid;

use crate::supervisor::{self, HandoffState};
use crate::{ApprovalMode, REVIEW_TASK, RunCapabilities, SIMPLIFY_TASK, run_chat_turn};
use frame_requester::FrameRequester;
use slash::SlashMenu;

const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(16);
const MIN_FRAME_INTERVAL: Duration = Duration::from_millis(8);
const ACTIVITY_FRAME_INTERVAL: Duration = Duration::from_millis(100);
const MIN_WIDTH: u16 = 36;
const MIN_HEIGHT: u16 = 10;
const MAX_APPLIED_EVENT_IDS: usize = 8_192;
const MAX_RUNTIME_EVENTS_PER_DRAIN: usize = 256;
const MAX_TRANSCRIPT_ITEMS: usize = 4_096;
const MAX_TRANSCRIPT_TEXT_BYTES: usize = 256 * 1024;
const MAX_TRANSCRIPT_TOTAL_BYTES: usize = 8 * 1024 * 1024;
const MAX_COMPOSER_BYTES: usize = 256 * 1024;
const MAX_TOOL_VIEWS: usize = 4_096;
const MAX_CHILD_VIEWS: usize = 256;
const MAX_TOOL_NAME_BYTES: usize = 128;
const MAX_TOOL_PREVIEW_BYTES: usize = 4_096;
const MAX_TOOL_CALL_ID_PREFIX_BYTES: usize = 512;
const MAX_TOOL_TEXT_BYTES: usize = 4 * 1024 * 1024;
const MAX_APPROVAL_FIELD_BYTES: usize = 16 * 1024;
const MAX_CHILD_TEXT_BYTES: usize = 64 * 1024;
const MAX_PENDING_CHILD_EVENT_BYTES: usize = 64 * 1024;
const MAX_PENDING_CHILD_EVENT_TOTAL_BYTES: usize = 1024 * 1024;

/// Keep the line-oriented host for non-interactive invocation and CI.
pub(crate) fn is_supported() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

/// Run the full-screen interactive host.
pub(crate) fn run(
    runtime: FocusRuntime,
    initial_session: Option<Uuid>,
    title: String,
    model_label: String,
    provider: Arc<dyn ModelProvider>,
    capabilities: RunCapabilities,
    approval_mode: ApprovalMode,
) -> Result<(), String> {
    let handoff = supervisor::load_handoff_from_env()?;
    let initial_session = handoff
        .as_ref()
        .and_then(|state| state.session_id)
        .or(initial_session);
    let mut events = runtime.subscribe();
    let initial_replay = initial_session
        .map(|session_id| {
            runtime
                .replay(session_id)
                .map_err(|error| error.to_string())
        })
        .transpose()?;
    let mut terminal = TerminalSession::enter().map_err(|error| error.to_string())?;
    let mut view = ChatView::new(approval_mode, Language::load());
    if let (Some(session_id), Some(replay)) = (initial_session, initial_replay.as_deref()) {
        view.load_replay(session_id, replay);
        for child_session_id in view.child_order.clone() {
            match runtime.replay(child_session_id) {
                Ok(child_replay) => view.load_child_replay(child_session_id, &child_replay),
                Err(error) => view.push_notice(if view.language == Language::Chinese {
                    format!("恢复委派会话失败：{child_session_id}：{error}")
                } else {
                    format!("Could not restore delegate: {child_session_id}: {error}")
                }),
            }
        }
    }
    if let Some(state) = handoff.as_ref() {
        let replay_cursor = view.last_event_id;
        view.restore_handoff(state)?;
        if replay_cursor.is_some() {
            view.last_event_id = replay_cursor;
        }
    }
    if handoff.is_some() {
        supervisor::confirm_tui_ready()?;
        terminal.complete_handoff();
    }
    let mut session = initial_session;
    let mut active: Option<ActiveTurn> = None;
    let mut quitting = false;
    let mut approval_mode = approval_mode;
    let mut frames = FrameRequester::new(MIN_FRAME_INTERVAL);
    frames.request(Instant::now());
    let mut terminal_size = terminal::size().map_err(|error| error.to_string())?;

    while !quitting {
        let now = Instant::now();
        let event_session = active.as_ref().map(|turn| turn.session_id).or(session);
        let mut runtime_events_changed = false;
        if let Some(session_id) = event_session {
            let drained =
                drain_runtime_events_with_runtime(&events, &mut view, session_id, Some(&runtime));
            runtime_events_changed = drained.changed;
            if drained.disconnected {
                let replacement = runtime.subscribe();
                let started_at = view.turn_started_at;
                match rebuild_view_from_replay(&runtime, &mut view, session_id, started_at) {
                    Ok(()) => {
                        events = replacement;
                        runtime_events_changed = true;
                        view.push_notice(view.language.text(
                            "Live event stream recovered from persisted replay.",
                            "已从持久化回放恢复实时事件流。",
                        ));
                    }
                    Err(error) => {
                        events = replacement;
                        let message = format!(
                            "{}{}{}",
                            view.language
                                .text("Live event stream disconnected", "实时事件流已断开"),
                            view.language.text(": ", "："),
                            error
                        );
                        if view.turn_started_at.is_some() {
                            view.push_active_error(message);
                        } else {
                            view.push_error(message);
                        }
                    }
                }
            }
        }
        if runtime_events_changed
            || receive_approval(&mut active, &mut view)?
            || finish_turn(
                Some(&runtime),
                &mut active,
                &mut session,
                &mut view,
                &events,
            )?
        {
            frames.request(now);
        }
        if !view.should_refresh_status(active.is_some()) {
            frames.cancel_scheduled();
        }
        if frames.take_due(now) {
            if view.should_refresh_status(active.is_some()) {
                view.advance_activity();
            }
            render(&mut terminal, &view, &title, &model_label, active.is_some())
                .map_err(|error| error.to_string())?;
            let drawn_at = Instant::now();
            if let Some(delay) = view.status_refresh_delay(active.is_some(), drawn_at) {
                frames.request_in(drawn_at, delay);
            } else {
                frames.cancel_scheduled();
            }
        }

        if event::poll(frames.poll_timeout(Instant::now(), EVENT_POLL_INTERVAL))
            .map_err(|error| error.to_string())?
        {
            match event::read().map_err(|error| error.to_string())? {
                TerminalEvent::Key(key) if should_handle_key_event(key) => {
                    let action = view.handle_key_with_width(
                        key,
                        active.is_some(),
                        usize::from(terminal_size.0).saturating_sub(2).max(1),
                    );
                    if !matches!(action, UiAction::None) {
                        frames.request(Instant::now());
                    }
                    if active.is_some()
                        && let Some(message) = active_action_message(&action, view.language)
                    {
                        view.push_notice(message);
                        continue;
                    }
                    match action {
                        UiAction::None => {}
                        UiAction::Redraw => {}
                        UiAction::Approval(decision) => {
                            resolve_approval(&mut active, &mut view, decision)?;
                        }
                        UiAction::Cancel => {
                            if let Some(active) = active.as_ref() {
                                active.cancellation.cancel();
                                view.push_notice(
                                    view.language
                                        .text("Cancelling current turn...", "正在取消当前回合……"),
                                );
                            }
                        }
                        UiAction::Clear => view.clear_visible(),
                        UiAction::Copy => match view.copy_current_output() {
                            Ok(()) => view.push_notice(view.language.text(
                                "Copied the current response to the clipboard.",
                                "已复制本轮回答到剪贴板。",
                            )),
                            Err(error) => view.push_notice(format!(
                                "{}{}{}",
                                view.language.text(
                                    "Could not copy the current response",
                                    "复制本轮回答失败",
                                ),
                                view.language.text(": ", "："),
                                error
                            )),
                        },
                        UiAction::ToggleDetails => {
                            view.clear_selection();
                            view.details_visible = !view.details_visible;
                        }
                        UiAction::OpenPermissions => view.open_permissions(),
                        UiAction::OpenLanguage => view.open_language(),
                        UiAction::SetApprovalMode(mode) => {
                            approval_mode = mode;
                            view.set_approval_mode(mode);
                        }
                        UiAction::SetLanguage(language) => match view.set_language(language) {
                            Ok(()) => view.push_notice(
                                view.language.text("Language saved.", "语言设置已保存。"),
                            ),
                            Err(error) => view.push_notice(format!(
                                "{}{}{}",
                                view.language.text(
                                    "Language changed for this session, but could not be saved",
                                    "语言已在本次会话切换，但保存失败",
                                ),
                                view.language.text(": ", "："),
                                error
                            )),
                        },
                        UiAction::NewSession => {
                            session = None;
                            view.clear_visible();
                            view.push_notice(
                                view.language
                                    .text("Started a new session.", "已开始新会话。"),
                            );
                        }
                        UiAction::Status => view.push_status(session, capabilities),
                        UiAction::Update => {
                            let store = VersionedArtifactStore::new(default_update_root());
                            let manifest = match store.load_manifest() {
                                Ok(Some(manifest)) => manifest,
                                Ok(None) => {
                                    view.push_notice(view.language.text(
                                        "No verified Focus update is staged.",
                                        "没有已验证的 Focus 更新版本。",
                                    ));
                                    continue;
                                }
                                Err(error) => {
                                    view.push_notice(format!(
                                        "{}{}{}",
                                        view.language.text(
                                            "The staged Focus update is invalid",
                                            "已暂存的 Focus 更新无效",
                                        ),
                                        view.language.text(": ", "："),
                                        error,
                                    ));
                                    continue;
                                }
                            };
                            let handoff_path = default_update_root()
                                .join(format!("handoff-{}.json", std::process::id()));
                            let state = view.handoff_state(session);
                            match supervisor::request_handoff(manifest.active, state, &handoff_path)
                            {
                                Ok(()) => {
                                    terminal.preserve_for_handoff();
                                    return Err(supervisor::RESTART_REQUESTED.to_owned());
                                }
                                Err(error) => view.push_notice(format!(
                                    "{}{}{}",
                                    view.language
                                        .text("Focus update handoff failed", "Focus 更新交接失败",),
                                    view.language.text(": ", "："),
                                    error,
                                )),
                            }
                            let _ = fs::remove_file(&handoff_path);
                        }
                        UiAction::Exit => {
                            if let Some(active) = active.as_ref() {
                                active.cancellation.cancel();
                                view.push_notice(
                                    view.language
                                        .text("Cancelling current turn...", "正在取消当前回合……"),
                                );
                            } else {
                                quitting = true;
                            }
                        }
                        UiAction::Submit(task) if active.is_none() => {
                            let session_id = match session {
                                Some(id) => id,
                                None => match runtime.create_session(title.clone()) {
                                    Ok(created) => {
                                        session = Some(created.id);
                                        created.id
                                    }
                                    Err(error) => {
                                        view.push_error(error.to_string());
                                        continue;
                                    }
                                },
                            };
                            view.start_task(&task);
                            active = Some(start_turn(
                                runtime.clone(),
                                session_id,
                                title.clone(),
                                task,
                                provider.clone(),
                                capabilities,
                                approval_mode,
                            ));
                        }
                        UiAction::Submit(_) => {}
                    }
                }
                TerminalEvent::Mouse(mouse) => {
                    if view.handle_mouse(mouse, active.is_some(), terminal_size) {
                        frames.request(Instant::now());
                    }
                }
                TerminalEvent::Paste(text) => {
                    if view.insert_composer_text(&text) {
                        frames.request(Instant::now());
                    }
                }
                TerminalEvent::Resize(width, height)
                    if resize_changed(&mut terminal_size, (width, height)) =>
                {
                    view.clear_selection();
                    frames.request(Instant::now());
                }
                TerminalEvent::Resize(_, _) => {}
                _ => {}
            }
        }
    }

    if let Some(active) = active {
        stop_turn(active)?;
    }
    Ok(())
}

struct ActiveTurn {
    session_id: Uuid,
    worker: Option<TurnWorker>,
    cancellation: CancellationToken,
    inbox: Option<ApprovalInbox>,
    pending: Option<ApprovalEnvelope>,
}

type TurnWorker = thread::JoinHandle<(Option<Uuid>, Result<(), String>)>;

fn start_turn(
    runtime: FocusRuntime,
    session: Uuid,
    title: String,
    task: String,
    provider: Arc<dyn ModelProvider>,
    capabilities: RunCapabilities,
    approval_mode: ApprovalMode,
) -> ActiveTurn {
    let cancellation = CancellationToken::default();
    let (approval, inbox): (Arc<dyn ApprovalHandler>, Option<ApprovalInbox>) = match approval_mode {
        ApprovalMode::Interactive => {
            let (broker, inbox) = ApprovalBroker::new();
            (Arc::new(broker), Some(inbox))
        }
        ApprovalMode::ApproveAll => (focus_runtime::approve_all(), None),
        ApprovalMode::DenyAll => (focus_runtime::deny_approvals(), None),
    };
    let mut options = capabilities.options(approval);
    options.cancellation = Arc::new(cancellation.clone());
    let worker = thread::spawn(move || {
        let mut session = Some(session);
        let result =
            run_chat_turn(&runtime, &mut session, &title, &task, provider, options).map(|_| ());
        (session, result)
    });
    ActiveTurn {
        session_id: session,
        worker: Some(worker),
        cancellation,
        inbox,
        pending: None,
    }
}

impl ActiveTurn {
    fn is_finished(&self) -> bool {
        self.worker
            .as_ref()
            .is_none_or(thread::JoinHandle::is_finished)
    }

    fn join(mut self) -> Result<(Option<Uuid>, Result<(), String>), String> {
        self.worker
            .take()
            .ok_or_else(|| "Runtime worker was already joined".to_owned())?
            .join()
            .map_err(|_| "Runtime worker panicked".to_owned())
    }

    fn stop(mut self) -> Result<(), String> {
        self.cancellation.cancel();
        self.inbox.take();
        self.worker.take().map_or(Ok(()), |worker| {
            worker
                .join()
                .map(|_| ())
                .map_err(|_| "Runtime worker panicked".to_owned())
        })
    }
}

impl Drop for ActiveTurn {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.inbox.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn receive_approval(active: &mut Option<ActiveTurn>, view: &mut ChatView) -> Result<bool, String> {
    let Some(active) = active.as_mut() else {
        return Ok(false);
    };
    if active.pending.is_none()
        && let Some(inbox) = active.inbox.as_ref()
        && let Some(pending) = inbox
            .recv_timeout(Duration::ZERO)
            .map_err(|error| error.to_string())?
    {
        view.pending_approval = Some(ApprovalView::from_envelope(&pending, view.language));
        view.approval_selection = 0;
        active.pending = Some(pending);
        return Ok(true);
    }
    Ok(false)
}

fn resolve_approval(
    active: &mut Option<ActiveTurn>,
    view: &mut ChatView,
    decision: ApprovalDecision,
) -> Result<(), String> {
    let Some(active) = active.as_mut() else {
        return Ok(());
    };
    let Some(pending) = active.pending.take() else {
        return Ok(());
    };
    active
        .inbox
        .as_ref()
        .ok_or_else(|| "approval inbox was unavailable".to_owned())?
        .resolve(pending.id, decision)
        .map_err(|error| error.to_string())?;
    view.pending_approval = None;
    view.approval_selection = 0;
    Ok(())
}

fn finish_turn(
    runtime: Option<&FocusRuntime>,
    active: &mut Option<ActiveTurn>,
    session: &mut Option<Uuid>,
    view: &mut ChatView,
    events: &mpsc::Receiver<Event>,
) -> Result<bool, String> {
    if active.as_ref().is_none_or(|active| !active.is_finished()) {
        return Ok(false);
    }
    let active = active.take().expect("active turn was checked above");
    let session_id = active.session_id;
    let (next_session, result) = active.join()?;
    *session = next_session;
    // A Runtime replay is the canonical completion snapshot. It includes every
    // event persisted before the worker returned without consuming unrelated
    // process-wide EventHub traffic during shutdown.
    let started_at = view.turn_started_at;
    if let Some(runtime) = runtime {
        rebuild_view_from_replay(runtime, view, session_id, started_at)?;
    } else {
        // Unit fixtures have no persisted Runtime source, so consume their
        // finite channel before recording the terminal duration summary.
        let _ = drain_runtime_events_to_completion(events, view, session_id, None);
    }
    view.pending_approval = None;
    view.approval_selection = 0;
    match result {
        Ok(()) => view.turn_finished(),
        Err(error) => view.turn_failed(error),
    }
    Ok(true)
}

fn stop_turn(active: ActiveTurn) -> Result<(), String> {
    active.stop()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct DrainOutcome {
    changed: bool,
    disconnected: bool,
    processed: usize,
}

#[cfg(test)]
fn drain_runtime_events(
    receiver: &mpsc::Receiver<Event>,
    view: &mut ChatView,
    session_id: Uuid,
) -> DrainOutcome {
    drain_runtime_events_with_runtime(receiver, view, session_id, None)
}

fn drain_runtime_events_with_runtime(
    receiver: &mpsc::Receiver<Event>,
    view: &mut ChatView,
    session_id: Uuid,
    runtime: Option<&FocusRuntime>,
) -> DrainOutcome {
    let mut outcome = DrainOutcome::default();
    for _ in 0..MAX_RUNTIME_EVENTS_PER_DRAIN {
        let event = match receiver.try_recv() {
            Ok(event) => event,
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                outcome.disconnected = true;
                break;
            }
        };
        outcome.processed += 1;
        if event.session_id == session_id || view.has_child_session(event.session_id) {
            outcome.changed |= view.apply_stream_event(&event, session_id);
        } else {
            let belongs_to_root = runtime.is_none_or(|runtime| {
                view.is_descendant_session(runtime, session_id, event.session_id)
            });
            if !view.turn_started_at.is_some() || !view.accepting_child_events || !belongs_to_root {
                continue;
            }
            // A child can begin emitting before the root's registration event
            // wins the cross-thread fan-out race. Replay it once registration
            // arrives so the first visible child state is not lost.
            view.defer_child_event(&event);
        }
    }
    outcome
}

fn drain_runtime_events_to_completion(
    receiver: &mpsc::Receiver<Event>,
    view: &mut ChatView,
    session_id: Uuid,
    runtime: Option<&FocusRuntime>,
) -> DrainOutcome {
    let mut total = DrainOutcome::default();
    loop {
        let batch = drain_runtime_events_with_runtime(receiver, view, session_id, runtime);
        total.changed |= batch.changed;
        total.disconnected |= batch.disconnected;
        total.processed = total.processed.saturating_add(batch.processed);
        if batch.disconnected || batch.processed < MAX_RUNTIME_EVENTS_PER_DRAIN {
            return total;
        }
    }
}

/// EventHub is process-wide, so an event from another concurrent run can be
/// observed here before the root's child-registration event. Only retain an
/// unknown session when its persisted ancestry proves it belongs to this root.
fn is_descendant_session(runtime: &FocusRuntime, root: Uuid, candidate: Uuid) -> bool {
    if root == candidate {
        return false;
    }
    let mut current = candidate;
    for _ in 0..64 {
        let Ok(metadata) = runtime.session(current) else {
            return false;
        };
        let Some(parent) = metadata.parent_id else {
            return false;
        };
        if parent == root {
            return true;
        }
        current = parent;
    }
    false
}

fn rebuild_view_from_replay(
    runtime: &FocusRuntime,
    view: &mut ChatView,
    session_id: Uuid,
    preserve_started_at: Option<Instant>,
) -> Result<(), String> {
    let pending_approval = view.pending_approval.clone();
    let approval_selection = view.approval_selection;
    let scroll_from_tail = view.scroll_from_tail;
    let replay = runtime
        .replay(session_id)
        .map_err(|error| error.to_string())?;
    view.load_replay(session_id, &replay);
    for child_session_id in view.child_order.clone() {
        let child_replay = runtime
            .replay(child_session_id)
            .map_err(|error| error.to_string())?;
        view.load_child_replay(child_session_id, &child_replay);
    }
    if let Some(started_at) = preserve_started_at {
        view.turn_started_at = Some(started_at);
        view.completed_turn_elapsed = None;
        view.turn_summary_recorded = false;
    }
    // Replay rebuilds transcript state, but an interactive approval belongs to
    // the still-running worker and must remain visible while that worker waits.
    view.pending_approval = pending_approval;
    view.approval_selection = approval_selection;
    view.scroll_from_tail = scroll_from_tail;
    Ok(())
}

fn resize_changed(current: &mut (u16, u16), next: (u16, u16)) -> bool {
    if *current == next {
        return false;
    }
    *current = next;
    true
}

fn is_terminal_status(status: AgentStatus) -> bool {
    matches!(
        status,
        AgentStatus::Complete | AgentStatus::Failed | AgentStatus::Cancelled
    )
}

fn is_cancellation_error(error: &str) -> bool {
    error == "task was cancelled"
        || error == "agent loop was cancelled"
        || error.ends_with(": task was cancelled")
        || error.ends_with(": agent loop was cancelled")
}

fn should_handle_key_event(key: KeyEvent) -> bool {
    matches!(
        key.kind,
        event::KeyEventKind::Press | event::KeyEventKind::Repeat
    )
}

fn active_action_message(action: &UiAction, language: Language) -> Option<&'static str> {
    match action {
        UiAction::Clear => Some(language.text(
            "Finish or cancel the current turn before clearing.",
            "请先完成或取消当前回合，再清空界面。",
        )),
        UiAction::NewSession => Some(language.text(
            "Finish or cancel the current turn before starting a new session.",
            "请先完成或取消当前回合，再开始新会话。",
        )),
        UiAction::Update => Some(language.text(
            "Finish or cancel the current turn before updating Focus.",
            "请先完成或取消当前回合，再更新 Focus。",
        )),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolState {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
struct ToolView {
    call_key: Option<ToolCallKey>,
    name: String,
    preview: String,
    state: ToolState,
    detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolCallKey {
    prefix: String,
    length: usize,
    digest: u64,
}

impl ToolCallKey {
    fn from_call_id(call_id: &str) -> Self {
        let mut hasher = DefaultHasher::new();
        call_id.hash(&mut hasher);
        Self {
            prefix: bounded_text(call_id.to_owned(), MAX_TOOL_CALL_ID_PREFIX_BYTES),
            length: call_id.len(),
            digest: hasher.finish(),
        }
    }

    fn matches(&self, call_id: &str) -> bool {
        self.length == call_id.len()
            && self.prefix == bounded_text(call_id.to_owned(), MAX_TOOL_CALL_ID_PREFIX_BYTES)
            && self.digest == Self::from_call_id(call_id).digest
    }
}

fn tool_text_bytes(tool: &ToolView) -> usize {
    tool.name.len() + tool.preview.len() + tool.detail.as_ref().map_or(0, String::len)
}

#[derive(Debug, Clone, Default)]
struct ChildView {
    role: String,
    state: String,
    reasoning: String,
    output: String,
}

#[derive(Debug, Clone)]
enum TranscriptItem {
    User(String),
    Assistant(String),
    Reasoning(String),
    Activity(String),
    Notice(String),
    Error(String),
    Tool(usize),
    TurnSummary(Duration),
}

fn transcript_item_text_bytes(item: &TranscriptItem) -> usize {
    match item {
        TranscriptItem::User(text)
        | TranscriptItem::Assistant(text)
        | TranscriptItem::Reasoning(text)
        | TranscriptItem::Activity(text)
        | TranscriptItem::Notice(text)
        | TranscriptItem::Error(text) => text.len(),
        TranscriptItem::Tool(_) | TranscriptItem::TurnSummary(_) => 0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TextPoint {
    row: usize,
    column: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TextSelection {
    anchor: TextPoint,
    focus: TextPoint,
}

impl TextSelection {
    fn ordered(self) -> (TextPoint, TextPoint) {
        if self.anchor <= self.focus {
            (self.anchor, self.focus)
        } else {
            (self.focus, self.anchor)
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Language {
    #[default]
    English,
    Chinese,
}

impl Language {
    fn text(self, english: &'static str, chinese: &'static str) -> &'static str {
        match self {
            Self::English => english,
            Self::Chinese => chinese,
        }
    }

    fn code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::Chinese => "zh",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "en" | "english" => Some(Self::English),
            "zh" | "cn" | "中文" | "chinese" => Some(Self::Chinese),
            _ => None,
        }
    }

    fn load() -> Self {
        Self::config_path()
            .as_deref()
            .map_or(Self::English, Self::load_from)
    }

    fn save(self) -> Result<(), String> {
        let path =
            Self::config_path().ok_or_else(|| "user config directory is unavailable".to_owned())?;
        self.save_to(&path)
    }

    fn load_from(path: &std::path::Path) -> Self {
        fs::read_to_string(path)
            .ok()
            .and_then(|value| Self::parse(&value))
            .unwrap_or(Self::English)
    }

    fn save_to(self, path: &std::path::Path) -> Result<(), String> {
        let parent = path
            .parent()
            .ok_or_else(|| "language config path has no parent".to_owned())?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        let temporary = parent.join(format!(".language-{}.tmp", Uuid::new_v4()));
        fs::write(&temporary, format!("{}\n", self.code()))
            .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
        if let Err(error) = replace_file(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(format!("failed to save {}: {error}", path.display()));
        }
        Ok(())
    }

    fn config_path() -> Option<PathBuf> {
        if let Some(path) = env::var_os("FOCUS_CONFIG_HOME") {
            return Some(PathBuf::from(path).join("language"));
        }
        #[cfg(windows)]
        {
            env::var_os("APPDATA").map(|root| PathBuf::from(root).join("Focus").join("language"))
        }
        #[cfg(not(windows))]
        {
            env::var_os("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
                .map(|root| root.join("focus").join("language"))
        }
    }
}

fn replace_file(temporary: &std::path::Path, target: &std::path::Path) -> io::Result<()> {
    #[cfg(windows)]
    if target.exists() {
        let backup = target.with_extension(format!("bak-{}", Uuid::new_v4()));
        fs::rename(target, &backup)?;
        if let Err(error) = fs::rename(temporary, target) {
            let _ = fs::rename(&backup, target);
            return Err(error);
        }
        let _ = fs::remove_file(backup);
        return Ok(());
    }
    fs::rename(temporary, target)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum UiAction {
    None,
    Redraw,
    Submit(String),
    Approval(ApprovalDecision),
    Copy,
    OpenPermissions,
    OpenLanguage,
    SetApprovalMode(ApprovalMode),
    SetLanguage(Language),
    Cancel,
    Clear,
    ToggleDetails,
    NewSession,
    Status,
    Update,
    Exit,
}

#[derive(Debug, Clone)]
struct ApprovalView {
    tool: String,
    operation: String,
    preview: String,
    rationale: String,
}

impl ApprovalView {
    fn new(
        tool: impl Into<String>,
        operation: impl Into<String>,
        preview: impl Into<String>,
        rationale: impl Into<String>,
    ) -> Self {
        Self {
            tool: bounded_text(tool.into(), MAX_APPROVAL_FIELD_BYTES),
            operation: bounded_text(operation.into(), MAX_APPROVAL_FIELD_BYTES),
            preview: bounded_text(preview.into(), MAX_APPROVAL_FIELD_BYTES),
            rationale: bounded_text(rationale.into(), MAX_APPROVAL_FIELD_BYTES),
        }
    }

    fn from_envelope(envelope: &ApprovalEnvelope, language: Language) -> Self {
        Self::new(
            envelope.request.tool.clone(),
            approval_operation_label(envelope.request.operation),
            tool_preview_with_language(
                &envelope.request.tool,
                &envelope.request.arguments,
                language,
            ),
            envelope.request.rationale.clone(),
        )
    }

    #[cfg(test)]
    fn rendered_with_selection(&self, selected: usize) -> String {
        self.rendered_with_language(selected, Language::English)
    }

    fn rendered_with_language(&self, selected: usize, language: Language) -> String {
        let question = match self.operation.as_str() {
            "execute" => language.text(
                "Would you like to run the following command?",
                "要运行下面的命令吗？",
            ),
            "network" => language.text(
                "Would you like to approve this network request?",
                "要批准这个网络请求吗？",
            ),
            "write" => language.text(
                "Would you like to make the following edit?",
                "要应用下面的编辑吗？",
            ),
            _ => language.text(
                "Would you like to allow the following tool call?",
                "要允许下面的工具调用吗？",
            ),
        };
        let options = [
            language.text("Yes, just this once (y)", "是，仅本次 (y)"),
            language.text("Yes, allow for this session (s)", "是，本会话允许 (s)"),
            language.text(
                "No, continue without permission (n)",
                "否，继续但不允许 (n)",
            ),
        ];
        let choices = options
            .into_iter()
            .enumerate()
            .map(|(index, option)| {
                let cursor = if index == selected { '›' } else { ' ' };
                format!("{cursor} {}. {option}", index + 1)
            })
            .collect::<Vec<_>>()
            .join("\n");
        let operation = localized_operation_label(&self.operation, language);
        let target = language.text("Target", "目标");
        let target_separator = language.text(":", "：");
        let target_line = format!("{target}{target_separator}{}", self.preview);
        let reason_line = format!(
            "{}{}{}",
            language.text("Reason", "原因"),
            language.text(": ", "："),
            self.rationale
        );
        format!(
            "  {question}\n\n  {reason_line}\n\n  {} ({})\n  {target_line}\n\n{choices}\n\n  {}",
            self.tool,
            operation,
            language.text(
                "Press enter to confirm or esc to cancel",
                "回车确认，Esc 取消"
            ),
        )
    }

    fn decision_for_selection(selected: usize) -> ApprovalDecision {
        match selected {
            1 => ApprovalDecision::ApproveSession,
            2 => ApprovalDecision::Deny,
            _ => ApprovalDecision::ApproveOnce,
        }
    }
}

#[derive(Debug)]
struct ChatView {
    transcript: Vec<TranscriptItem>,
    transcript_text_bytes: usize,
    tools: Vec<ToolView>,
    tool_text_bytes: usize,
    composer: String,
    composer_cursor: usize,
    composer_cursor_set: bool,
    pending_approval: Option<ApprovalView>,
    approval_selection: usize,
    approval_mode: ApprovalMode,
    language: Language,
    permissions_visible: bool,
    permission_selection: usize,
    language_visible: bool,
    language_selection: usize,
    help_visible: bool,
    slash_menu: SlashMenu,
    details_visible: bool,
    error_seen: bool,
    assistant_index: Option<usize>,
    reasoning_buffer: String,
    child_views: BTreeMap<Uuid, ChildView>,
    child_order: Vec<Uuid>,
    pending_child_events: BTreeMap<Uuid, Vec<Event>>,
    pending_child_event_count: usize,
    pending_child_event_bytes: usize,
    child_membership_cache: BTreeMap<Uuid, bool>,
    accepting_child_events: bool,
    scroll_from_tail: usize,
    turn_started_at: Option<Instant>,
    completed_turn_elapsed: Option<Duration>,
    turn_summary_recorded: bool,
    activity_frame: usize,
    cancelled_seen: bool,
    last_event_id: Option<Uuid>,
    applied_event_ids: HashSet<Uuid>,
    applied_event_order: VecDeque<Uuid>,
    selection: Option<TextSelection>,
    selection_text: Option<String>,
    mouse_anchor: Option<TextPoint>,
}

impl Default for ChatView {
    fn default() -> Self {
        Self::new(ApprovalMode::ApproveAll, Language::English)
    }
}

impl ChatView {
    fn new(approval_mode: ApprovalMode, language: Language) -> Self {
        Self {
            transcript: Vec::new(),
            transcript_text_bytes: 0,
            tools: Vec::new(),
            tool_text_bytes: 0,
            composer: String::new(),
            composer_cursor: 0,
            composer_cursor_set: false,
            pending_approval: None,
            approval_selection: 0,
            approval_mode,
            language,
            permissions_visible: false,
            permission_selection: permission_selection(approval_mode),
            language_visible: false,
            language_selection: language_selection(language),
            help_visible: false,
            slash_menu: SlashMenu::default(),
            details_visible: false,
            error_seen: false,
            assistant_index: None,
            reasoning_buffer: String::new(),
            child_views: BTreeMap::new(),
            child_order: Vec::new(),
            pending_child_events: BTreeMap::new(),
            pending_child_event_count: 0,
            pending_child_event_bytes: 0,
            child_membership_cache: BTreeMap::new(),
            accepting_child_events: false,
            scroll_from_tail: 0,
            turn_started_at: None,
            completed_turn_elapsed: None,
            turn_summary_recorded: false,
            activity_frame: 0,
            cancelled_seen: false,
            last_event_id: None,
            applied_event_ids: HashSet::new(),
            applied_event_order: VecDeque::new(),
            selection: None,
            selection_text: None,
            mouse_anchor: None,
        }
    }

    fn should_refresh_status(&self, active: bool) -> bool {
        active && self.pending_approval.is_none() && self.turn_started_at.is_some()
    }

    fn status_refresh_delay(&self, active: bool, now: Instant) -> Option<Duration> {
        if !self.should_refresh_status(active) {
            return None;
        }
        let elapsed = now.saturating_duration_since(self.turn_started_at?);
        let next_second = Duration::from_secs(elapsed.as_secs().saturating_add(1));
        let interval_ms = ACTIVITY_FRAME_INTERVAL.as_millis();
        let elapsed_in_frame = elapsed.as_millis() % interval_ms;
        let next_frame = Duration::from_millis(
            u64::try_from(interval_ms.saturating_sub(elapsed_in_frame)).unwrap_or(u64::MAX),
        );
        Some(next_second.saturating_sub(elapsed).min(next_frame))
    }

    fn start_task(&mut self, task: &str) {
        self.push_transcript_item(TranscriptItem::User(task.to_owned()));
        self.clear_selection();
        self.assistant_index = None;
        self.child_views.clear();
        self.child_order.clear();
        self.clear_pending_child_events();
        self.child_membership_cache.clear();
        self.accepting_child_events = false;
        self.clear_applied_events();
        self.turn_started_at = Some(Instant::now());
        self.error_seen = false;
        self.scroll_from_tail = 0;
        self.completed_turn_elapsed = None;
        self.turn_summary_recorded = false;
        self.activity_frame = 0;
        self.cancelled_seen = false;
    }

    fn turn_finished(&mut self) {
        self.stop_turn_clock();
        self.clear_pending_child_events();
        self.child_membership_cache.clear();
        self.accepting_child_events = false;
        self.record_turn_summary();
    }

    fn stop_turn_clock(&mut self) {
        self.finish_reasoning();
        if let Some(started) = self.turn_started_at.take() {
            self.completed_turn_elapsed = Some(started.elapsed());
        }
        self.assistant_index = None;
    }

    fn record_turn_summary(&mut self) {
        if self.turn_summary_recorded {
            return;
        }
        if let Some(elapsed) = self.completed_turn_elapsed {
            self.clear_selection();
            self.push_transcript_item(TranscriptItem::TurnSummary(elapsed));
        }
        self.turn_summary_recorded = true;
    }

    fn advance_activity(&mut self) {
        self.activity_frame = self.activity_frame.wrapping_add(1);
    }

    fn push_notice(&mut self, message: impl Into<String>) {
        self.clear_selection();
        self.push_transcript_item(TranscriptItem::Notice(message.into()));
    }

    fn push_activity(&mut self, message: impl Into<String>) {
        self.clear_selection();
        self.push_transcript_item(TranscriptItem::Activity(message.into()));
    }

    fn push_error(&mut self, error: impl Into<String>) {
        self.clear_selection();
        self.stop_turn_clock();
        self.push_transcript_item(TranscriptItem::Error(error.into()));
        self.error_seen = true;
    }

    fn push_active_error(&mut self, error: impl Into<String>) {
        self.clear_selection();
        self.push_transcript_item(TranscriptItem::Error(error.into()));
        self.error_seen = true;
    }

    fn current_turn_has_error(&self, error: &str) -> bool {
        self.error_seen
            && self
                .transcript
                .iter()
                .rev()
                .take_while(|item| !matches!(item, TranscriptItem::User(_)))
                .any(|item| matches!(item, TranscriptItem::Error(text) if text == error))
    }

    fn push_transcript_item(&mut self, item: TranscriptItem) {
        let mut item = item;
        match &mut item {
            TranscriptItem::User(text)
            | TranscriptItem::Assistant(text)
            | TranscriptItem::Reasoning(text)
            | TranscriptItem::Activity(text)
            | TranscriptItem::Notice(text)
            | TranscriptItem::Error(text) => bound_transcript_text(text),
            TranscriptItem::Tool(_) | TranscriptItem::TurnSummary(_) => {}
        }
        self.transcript_text_bytes = self
            .transcript_text_bytes
            .saturating_add(transcript_item_text_bytes(&item));
        self.transcript.push(item);
        self.compact_transcript();
    }

    fn compact_transcript(&mut self) {
        let mut evicted = false;
        while self.transcript.len() > MAX_TRANSCRIPT_ITEMS
            || self.transcript_text_bytes > MAX_TRANSCRIPT_TOTAL_BYTES
        {
            let item = self.transcript.remove(0);
            self.transcript_text_bytes = self
                .transcript_text_bytes
                .saturating_sub(transcript_item_text_bytes(&item));
            if let Some(index) = self.assistant_index {
                self.assistant_index = (index > 0).then_some(index - 1);
            }
            evicted = true;
        }
        if evicted {
            self.clear_selection();
        }
    }

    fn turn_failed(&mut self, error: String) {
        if is_cancellation_error(&error) {
            self.turn_cancelled();
            return;
        }
        if !self.current_turn_has_error(&error) {
            self.push_error(error);
        } else {
            self.stop_turn_clock();
        }
        self.record_turn_summary();
    }

    fn turn_cancelled(&mut self) {
        self.stop_turn_clock();
        if !self.cancelled_seen {
            self.push_notice(self.language.text("Interrupted", "已中断"));
            self.cancelled_seen = true;
        }
        self.record_turn_summary();
    }

    fn clear_visible(&mut self) {
        self.transcript.clear();
        self.transcript_text_bytes = 0;
        self.clear_selection();
        self.tools.clear();
        self.tool_text_bytes = 0;
        self.pending_approval = None;
        self.approval_selection = 0;
        self.permissions_visible = false;
        self.permission_selection = permission_selection(self.approval_mode);
        self.language_visible = false;
        self.language_selection = language_selection(self.language);
        self.help_visible = false;
        self.slash_menu.dismiss();
        self.error_seen = false;
        self.assistant_index = None;
        self.reasoning_buffer.clear();
        self.child_views.clear();
        self.child_order.clear();
        self.clear_pending_child_events();
        self.child_membership_cache.clear();
        self.accepting_child_events = false;
        self.clear_applied_events();
        self.turn_started_at = None;
        self.completed_turn_elapsed = None;
        self.turn_summary_recorded = false;
        self.activity_frame = 0;
        self.cancelled_seen = false;
        self.scroll_from_tail = 0;
        self.reset_composer();
    }

    fn clear_pending_child_events(&mut self) {
        self.pending_child_events.clear();
        self.pending_child_event_count = 0;
        self.pending_child_event_bytes = 0;
    }

    fn is_descendant_session(
        &mut self,
        runtime: &FocusRuntime,
        root: Uuid,
        candidate: Uuid,
    ) -> bool {
        const MAX_CACHED_SESSIONS: usize = 256;
        if let Some(&belongs) = self.child_membership_cache.get(&candidate) {
            return belongs;
        }
        let belongs = is_descendant_session(runtime, root, candidate);
        if self.child_membership_cache.len() >= MAX_CACHED_SESSIONS
            && let Some(first) = self.child_membership_cache.keys().next().copied()
        {
            self.child_membership_cache.remove(&first);
        }
        self.child_membership_cache.insert(candidate, belongs);
        belongs
    }

    fn clear_applied_events(&mut self) {
        self.applied_event_ids.clear();
        self.applied_event_order.clear();
    }

    fn handoff_state(&self, session_id: Option<Uuid>) -> HandoffState {
        HandoffState {
            schema_version: 1,
            session_id,
            event_cursor: self.last_event_id,
            composer: self.composer.clone(),
            composer_cursor: self.composer_cursor,
            scroll_from_tail: self.scroll_from_tail,
            details_visible: self.details_visible,
            language: self.language.code().to_owned(),
        }
    }

    fn restore_handoff(&mut self, state: &HandoffState) -> Result<(), String> {
        let language = Language::parse(&state.language)
            .ok_or_else(|| format!("unsupported handoff language `{}`", state.language))?;
        self.composer = state.composer.clone();
        self.composer_cursor = state.composer_cursor;
        self.composer_cursor_set = true;
        self.ensure_composer_cursor();
        self.scroll_from_tail = state.scroll_from_tail;
        self.details_visible = state.details_visible;
        self.language = language;
        self.language_selection = language_selection(language);
        self.last_event_id = state.event_cursor;
        self.slash_menu.update(&self.composer);
        Ok(())
    }

    fn remember_event(&mut self, event_id: Uuid) -> bool {
        if !self.applied_event_ids.insert(event_id) {
            return false;
        }
        self.applied_event_order.push_back(event_id);
        while self.applied_event_order.len() > MAX_APPLIED_EVENT_IDS {
            if let Some(oldest) = self.applied_event_order.pop_front() {
                self.applied_event_ids.remove(&oldest);
            }
        }
        true
    }

    fn push_status(&mut self, session: Option<Uuid>, capabilities: RunCapabilities) {
        let session = session.map_or_else(
            || self.language.text("new", "新会话").to_owned(),
            |id| id.to_string(),
        );
        let mode = if capabilities.workflow {
            self.language.text("engineering", "工程")
        } else {
            self.language.text("core", "核心")
        };
        let mode = if capabilities.delegation {
            format!("{mode} {}", self.language.text("+ delegation", "+ 委派"))
        } else {
            mode.to_owned()
        };
        self.push_notice(format!(
            "{}{}{}{}{}{}{}",
            self.language.text("Session", "会话"),
            self.language.text(": ", "："),
            session,
            self.language.text(" | ", "｜"),
            self.language.text("Mode", "模式"),
            self.language.text(": ", "："),
            mode
        ));
    }

    fn open_permissions(&mut self) {
        self.permissions_visible = true;
        self.permission_selection = permission_selection(self.approval_mode);
    }

    fn open_language(&mut self) {
        self.language_visible = true;
        self.language_selection = language_selection(self.language);
    }

    fn set_language(&mut self, language: Language) -> Result<(), String> {
        self.clear_selection();
        self.language = language;
        self.language_selection = language_selection(language);
        self.language_visible = false;
        language.save()
    }

    fn copy_current_output(&self) -> Result<(), String> {
        if let Some(selection) = self
            .selection_text
            .as_deref()
            .filter(|text| !text.is_empty())
        {
            return copy_to_clipboard(selection);
        }
        let Some(start) = self
            .transcript
            .iter()
            .rposition(|item| matches!(item, TranscriptItem::User(_)))
        else {
            return Err(self
                .language
                .text("no response is available", "当前没有可复制的回答")
                .to_owned());
        };
        let output = self.transcript[start + 1..]
            .iter()
            .filter_map(|item| match item {
                TranscriptItem::Assistant(text) if !text.trim().is_empty() => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        if output.trim().is_empty() {
            return Err(self
                .language
                .text("no response is available", "当前没有可复制的回答")
                .to_owned());
        }
        copy_to_clipboard(&output)
    }

    fn clear_selection(&mut self) {
        self.selection = None;
        self.selection_text = None;
        self.mouse_anchor = None;
    }

    fn update_selection(&mut self, focus: TextPoint, width: usize) {
        let Some(anchor) = self.mouse_anchor else {
            return;
        };
        let selection = TextSelection { anchor, focus };
        self.selection_text = self.selected_text(selection, width);
        self.selection = self.selection_text.as_ref().map(|_| selection);
    }

    fn selected_text(&self, selection: TextSelection, width: usize) -> Option<String> {
        let (start, end) = selection.ordered();
        if start == end {
            return None;
        }
        let lines = self.lines(width.max(1));
        if start.row >= lines.len() {
            return None;
        }
        let end_row = end.row.min(lines.len().saturating_sub(1));
        let mut selected = Vec::new();
        for (row, line) in lines
            .iter()
            .enumerate()
            .skip(start.row)
            .take(end_row.saturating_sub(start.row).saturating_add(1))
        {
            let text = &line.text;
            let from = if row == start.row { start.column } else { 0 };
            let to = if row == end_row {
                end.column
            } else {
                display_width(text)
            };
            selected.push(slice_display_columns(text, from, to));
        }
        let result = selected.join("\n");
        (!result.is_empty()).then_some(result)
    }

    fn set_approval_mode(&mut self, approval_mode: ApprovalMode) {
        self.approval_mode = approval_mode;
        self.permission_selection = permission_selection(approval_mode);
        self.permissions_visible = false;
        self.push_notice(format!(
            "{}{}{}",
            self.language.text("Permissions", "权限"),
            self.language.text(": ", "："),
            approval_mode_label(approval_mode, self.language)
        ));
    }

    fn apply_event(&mut self, event: &Event) -> bool {
        match &event.kind {
            EventKind::StateChanged { status, .. } => {
                let was_running = self.turn_started_at.is_some();
                if is_terminal_status(*status) {
                    self.stop_turn_clock();
                }
                is_terminal_status(*status) && was_running
            }
            EventKind::Runtime { name, data } => self.apply_runtime_event(name, data),
            EventKind::Model {
                event: ModelEvent::RequestStarted { .. },
            }
            | EventKind::Model {
                event: ModelEvent::ToolArgumentsDelta { .. },
            }
            | EventKind::Model {
                event: ModelEvent::ToolCallReady { .. },
            }
            | EventKind::Model {
                event: ModelEvent::Usage { .. },
            }
            | EventKind::Model {
                event: ModelEvent::Completed { .. },
            }
            | EventKind::Model {
                event: ModelEvent::Cancelled,
            } => false,
            EventKind::Model {
                event: ModelEvent::ToolCallStarted { name, .. },
            } => {
                self.push_activity(format!(
                    "{} {name}",
                    self.language.text("Selecting", "正在选择")
                ));
                true
            }
            EventKind::Model {
                event:
                    ModelEvent::RetryScheduled {
                        attempt,
                        delay_ms,
                        reason,
                    },
            } => {
                self.push_activity(if self.language == Language::Chinese {
                    format!("将在 {delay_ms}ms 后重试：{reason}（第 {attempt} 次）")
                } else {
                    format!("Retrying in {delay_ms}ms: {reason} (attempt {attempt})")
                });
                true
            }
            EventKind::Model {
                event: ModelEvent::ReasoningSummaryDelta { text },
            } => {
                if text.is_empty() {
                    false
                } else {
                    self.append_reasoning(text);
                    true
                }
            }
            EventKind::Model {
                event: ModelEvent::TextDelta { text },
            } => {
                if text.is_empty() {
                    return false;
                }
                self.finish_reasoning();
                self.append_assistant(text);
                true
            }
            EventKind::Model {
                event: ModelEvent::Failed { error },
            } => {
                if is_cancellation_error(error) {
                    self.stop_turn_clock();
                    if !self.cancelled_seen {
                        self.push_notice(self.language.text("Interrupted", "已中断"));
                        self.cancelled_seen = true;
                    }
                } else if self.current_turn_has_error(error) {
                    return false;
                } else {
                    self.push_error(error.clone());
                }
                true
            }
            EventKind::ToolCallRequested { call } => {
                self.finish_reasoning();
                if call.name == "delegate" {
                    self.accepting_child_events = true;
                }
                self.tool_started_with_id(
                    call.id.clone(),
                    call.name.clone(),
                    tool_preview_with_language(&call.name, &call.arguments, self.language),
                );
                true
            }
            EventKind::ToolResultReceived { result } => {
                self.tool_finished_with_id(&result.tool_call_id, result.is_error, &result.content)
            }
            EventKind::MessageAdded { message } if message.role == Role::Assistant => {
                if message.content.trim().is_empty() {
                    false
                } else {
                    self.apply_assistant_message(&message.content);
                    true
                }
            }
            _ => false,
        }
    }

    fn apply_stream_event(&mut self, event: &Event, root_session_id: Uuid) -> bool {
        let is_root = event.session_id == root_session_id;
        let is_child = !is_root && self.has_child_session(event.session_id);
        if !is_root && !is_child {
            return false;
        }
        if !self.remember_event(event.id) {
            return false;
        }
        let changed = if is_child {
            self.apply_child_event(event)
        } else {
            self.apply_event(event)
        };
        self.last_event_id = Some(event.id);
        if changed {
            self.clear_selection();
        }
        changed
    }

    fn defer_child_event(&mut self, event: &Event) {
        const MAX_PENDING_EVENTS_PER_CHILD: usize = 256;
        const MAX_PENDING_EVENTS_TOTAL: usize = 1024;
        const MAX_PENDING_CHILD_SESSIONS: usize = 64;
        if self.pending_child_event_count >= MAX_PENDING_EVENTS_TOTAL {
            return;
        }
        if !self.pending_child_events.contains_key(&event.session_id)
            && self.pending_child_events.len() >= MAX_PENDING_CHILD_SESSIONS
        {
            return;
        }
        let Some(event_bytes) = serialized_event_bytes(event) else {
            return;
        };
        if event_bytes > MAX_PENDING_CHILD_EVENT_BYTES
            || self.pending_child_event_bytes.saturating_add(event_bytes)
                > MAX_PENDING_CHILD_EVENT_TOTAL_BYTES
        {
            return;
        }
        let pending = self
            .pending_child_events
            .entry(event.session_id)
            .or_default();
        // EventHub delivery is at-least-once across reconnects. Keep the
        // deferred queue idempotent so a registration race cannot replay the
        // same child event twice.
        if pending.iter().any(|queued| queued.id == event.id) {
            return;
        }
        if pending.len() < MAX_PENDING_EVENTS_PER_CHILD {
            pending.push(event.clone());
            self.pending_child_event_count += 1;
            self.pending_child_event_bytes =
                self.pending_child_event_bytes.saturating_add(event_bytes);
        }
    }

    fn load_replay(&mut self, _session_id: Uuid, replay: &[Event]) {
        self.transcript.clear();
        self.transcript_text_bytes = 0;
        self.clear_selection();
        self.tools.clear();
        self.tool_text_bytes = 0;
        self.pending_approval = None;
        self.approval_selection = 0;
        self.assistant_index = None;
        self.reasoning_buffer.clear();
        self.child_views.clear();
        self.child_order.clear();
        self.clear_pending_child_events();
        self.child_membership_cache.clear();
        self.accepting_child_events = false;
        self.turn_started_at = None;
        self.completed_turn_elapsed = None;
        self.turn_summary_recorded = false;
        self.cancelled_seen = false;
        self.last_event_id = None;
        self.error_seen = false;
        self.activity_frame = 0;
        self.scroll_from_tail = 0;
        self.clear_applied_events();
        for event in replay {
            let _ = self.apply_replayed_root_event(event);
            self.remember_event(event.id);
            self.last_event_id = Some(event.id);
        }
    }

    fn load_child_replay(&mut self, child_session_id: Uuid, replay: &[Event]) {
        if !self.has_child_session(child_session_id) {
            return;
        }
        for event in replay {
            if event.session_id == child_session_id {
                self.apply_child_event(event);
                self.remember_event(event.id);
                self.last_event_id = Some(event.id);
            }
        }
    }

    fn apply_replayed_root_event(&mut self, event: &Event) -> bool {
        match &event.kind {
            EventKind::MessageAdded { message } => self.apply_replayed_message(message),
            _ => self.apply_event(event),
        }
    }

    fn apply_replayed_message(&mut self, message: &focus_kernel::Message) -> bool {
        match message.role {
            Role::User => {
                self.finish_reasoning();
                self.assistant_index = None;
                self.push_transcript_item(TranscriptItem::User(redact_sensitive_text(
                    &message.content,
                )));
                true
            }
            Role::Assistant if !message.content.trim().is_empty() => {
                self.apply_assistant_message(&message.content);
                true
            }
            Role::Assistant => false,
            Role::System | Role::Tool => {
                self.assistant_index = None;
                false
            }
        }
    }

    fn has_child_session(&self, session_id: Uuid) -> bool {
        self.child_views.contains_key(&session_id)
    }

    fn compact_child_views(&mut self) {
        if self.child_views.len() < MAX_CHILD_VIEWS {
            return;
        }
        if let Some(evicted) = self.child_order.first().copied() {
            self.child_order.remove(0);
            self.child_views.remove(&evicted);
            if let Some(pending) = self.pending_child_events.remove(&evicted) {
                self.pending_child_event_count =
                    self.pending_child_event_count.saturating_sub(pending.len());
                self.pending_child_event_bytes = self
                    .pending_child_event_bytes
                    .saturating_sub(serialized_events_bytes(&pending));
            }
            self.child_membership_cache.remove(&evicted);
            self.clear_selection();
        }
    }

    fn apply_runtime_event(&mut self, name: &str, data: &serde_json::Value) -> bool {
        if name == "subagent_started"
            && let Some(child_session_id) = uuid_field(data, "child_session_id")
        {
            self.accepting_child_events = true;
            let mut changed = false;
            if !self.child_views.contains_key(&child_session_id) {
                self.compact_child_views();
                self.child_order.push(child_session_id);
                self.child_views.insert(
                    child_session_id,
                    ChildView {
                        role: bounded_text(role_field(data, self.language), MAX_TOOL_NAME_BYTES),
                        state: "starting".into(),
                        ..ChildView::default()
                    },
                );
                changed = true;
            }
            if let Some(events) = self.pending_child_events.remove(&child_session_id) {
                self.pending_child_event_count =
                    self.pending_child_event_count.saturating_sub(events.len());
                self.pending_child_event_bytes = self
                    .pending_child_event_bytes
                    .saturating_sub(serialized_events_bytes(&events));
                for event in events {
                    if self.remember_event(event.id) {
                        changed |= self.apply_child_event(&event);
                    }
                }
            }
            return changed;
        }
        let message = match name {
            "subagent_queued" => {
                self.accepting_child_events = true;
                format!(
                    "{}{}{}",
                    self.language.text("Queueing delegate", "排队委派"),
                    self.language.text(": ", "："),
                    role_field(data, self.language)
                )
            }
            "subagent_completed" => format!(
                "{}{}{}",
                self.language.text("Delegate completed", "委派完成"),
                self.language.text(": ", "："),
                role_field(data, self.language)
            ),
            "subagent_failed" => format!(
                "{}{}{}{}{}",
                self.language.text("Delegate failed", "委派失败"),
                self.language.text(": ", "："),
                role_field(data, self.language),
                self.language.text(" - ", "："),
                string_field(data, "summary")
                    .unwrap_or_else(|| self.language.text("unknown error", "未知错误").into())
            ),
            "subagent_cancelled" => format!(
                "{}{}{}",
                self.language.text("Delegate cancelled", "委派已取消"),
                self.language.text(": ", "："),
                role_field(data, self.language)
            ),
            "subagent_batch_completed" => {
                let count = data
                    .get("results")
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, Vec::len);
                if self.language == Language::Chinese {
                    format!(
                        "{}：{} 个任务",
                        self.language.text("Delegation completed", "委派完成"),
                        count
                    )
                } else {
                    format!(
                        "{}: {} task{}",
                        self.language.text("Delegation completed", "委派完成"),
                        count,
                        if count == 1 { "" } else { "s" }
                    )
                }
            }
            "workflow_stage_entered" => workflow_stage_label(
                &string_field(data, "stage").unwrap_or_else(|| "unknown".into()),
                self.language,
            ),
            "workflow_gate_failed" => {
                let missing = data
                    .get("missing")
                    .and_then(serde_json::Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(serde_json::Value::as_str)
                            .map(|item| workflow_requirement_label(item, self.language))
                            .collect::<Vec<_>>()
                            .join(if self.language == Language::Chinese {
                                "、"
                            } else {
                                ", "
                            })
                    })
                    .filter(|missing| !missing.is_empty())
                    .unwrap_or_else(|| self.language.text("missing evidence", "缺少证据").into());
                format!(
                    "{}{}{}",
                    self.language.text("Rechecking", "重新检查"),
                    self.language.text(": ", "："),
                    missing
                )
            }
            "workflow_evidence_recorded" => {
                let evidence = data
                    .get("evidence")
                    .and_then(|evidence| evidence.get("kind"))
                    .and_then(serde_json::Value::as_str)
                    .map(|kind| workflow_evidence_label(kind, self.language))
                    .unwrap_or_else(|| self.language.text("recorded", "已记录"));
                if self.language == Language::Chinese {
                    format!("已记录：{evidence}")
                } else {
                    format!("Recorded {evidence}")
                }
            }
            _ => return false,
        };
        self.push_activity(message);
        true
    }

    fn apply_child_event(&mut self, event: &Event) -> bool {
        let Some(child) = self.child_views.get_mut(&event.session_id) else {
            return false;
        };
        let role = child.role.clone();
        let mut notice = None;
        let changed = match &event.kind {
            EventKind::StateChanged { status, .. } => {
                let state = format!("{status:?}").to_lowercase();
                if child.state == state {
                    false
                } else {
                    child.state = state;
                    true
                }
            }
            EventKind::Model {
                event: ModelEvent::ReasoningSummaryDelta { text },
            } => append_bounded_text(
                &mut child.reasoning,
                &redact_sensitive_text(text),
                MAX_CHILD_TEXT_BYTES,
            ),
            EventKind::Model {
                event: ModelEvent::TextDelta { text },
            } => append_bounded_text(
                &mut child.output,
                &redact_sensitive_text(text),
                MAX_CHILD_TEXT_BYTES,
            ),
            EventKind::MessageAdded { message } if message.role == Role::Assistant => {
                let content = bounded_text(
                    redact_sensitive_text(&message.content),
                    MAX_CHILD_TEXT_BYTES,
                );
                if !content.trim().is_empty() && child.output != content {
                    child.output = content;
                    true
                } else {
                    false
                }
            }
            EventKind::Model {
                event: ModelEvent::ToolCallStarted { name, .. },
            } => {
                notice = Some(format!(
                    "{role} {} {name}",
                    self.language.text("selecting", "正在选择")
                ));
                true
            }
            EventKind::ToolCallRequested { call } => {
                notice = Some(format!(
                    "{} {} {}{}{}",
                    role,
                    self.language.text("using", "使用"),
                    call.name,
                    self.language.text(": ", "："),
                    tool_preview_with_language(&call.name, &call.arguments, self.language)
                ));
                true
            }
            EventKind::ToolResultReceived { result } => {
                notice = Some(format!(
                    "{} {}{}{}",
                    role,
                    if result.is_error {
                        self.language.text("failed", "失败")
                    } else {
                        self.language.text("finished", "已完成")
                    },
                    self.language.text(": ", "："),
                    result.name
                ));
                true
            }
            _ => false,
        };
        if let Some(notice) = notice {
            self.push_activity(notice);
        }
        changed
    }

    fn append_assistant(&mut self, text: &str) {
        let text = redact_sensitive_text(text);
        let index = match self.assistant_index {
            Some(index) => index,
            None => {
                self.push_transcript_item(TranscriptItem::Assistant(String::new()));
                let index = self.transcript.len() - 1;
                self.assistant_index = Some(index);
                index
            }
        };
        if let Some(TranscriptItem::Assistant(content)) = self.transcript.get_mut(index) {
            let previous_len = content.len();
            content.push_str(&text);
            bound_transcript_text(content);
            self.transcript_text_bytes = self
                .transcript_text_bytes
                .saturating_sub(previous_len)
                .saturating_add(content.len());
            self.compact_transcript();
        }
    }

    fn apply_assistant_message(&mut self, text: &str) {
        if text.trim().is_empty() {
            return;
        }
        self.finish_reasoning();
        let mut content = redact_sensitive_text(text);
        bound_transcript_text(&mut content);
        if let Some(index) = self.assistant_index.take()
            && let Some(TranscriptItem::Assistant(existing)) = self.transcript.get_mut(index)
        {
            let previous_len = existing.len();
            *existing = content;
            self.transcript_text_bytes = self
                .transcript_text_bytes
                .saturating_sub(previous_len)
                .saturating_add(existing.len());
            self.compact_transcript();
        } else {
            self.push_transcript_item(TranscriptItem::Assistant(content));
        }
    }

    fn append_reasoning(&mut self, text: &str) {
        self.reasoning_buffer.push_str(&redact_sensitive_text(text));
        bound_transcript_text(&mut self.reasoning_buffer);
    }

    fn finish_reasoning(&mut self) {
        let summary = self.reasoning_buffer.trim();
        if !summary.is_empty() {
            self.push_transcript_item(TranscriptItem::Reasoning(summary.to_owned()));
        }
        self.reasoning_buffer.clear();
    }

    fn thinking_header(&self) -> Option<String> {
        let header = self
            .reasoning_buffer
            .lines()
            .find(|line| !line.trim().is_empty())?;
        let header = header.trim();
        let header = header
            .strip_prefix("**")
            .and_then(|value| value.split("**").next())
            .unwrap_or(header)
            .trim();
        (!header.is_empty()).then(|| truncate(header, 56))
    }

    #[cfg(test)]
    fn tool_started(&mut self, name: &str, preview: &str) {
        self.tool_started_with_id(String::new(), name.to_owned(), preview.to_owned());
    }

    fn tool_started_with_id(&mut self, call_id: String, name: String, preview: String) {
        let tool_index = self.tools.len();
        let call_key = (!call_id.is_empty()).then(|| ToolCallKey::from_call_id(&call_id));
        let name = bounded_text(name, MAX_TOOL_NAME_BYTES);
        let preview = bounded_text(preview, MAX_TOOL_PREVIEW_BYTES);
        self.tool_text_bytes = self
            .tool_text_bytes
            .saturating_add(name.len().saturating_add(preview.len()));
        self.tools.push(ToolView {
            call_key,
            name,
            preview,
            state: ToolState::Running,
            detail: None,
        });
        self.push_transcript_item(TranscriptItem::Tool(tool_index));
        self.assistant_index = None;
        self.compact_tool_history();
    }

    fn compact_tool_history(&mut self) {
        let mut remove = 0;
        let mut retained_text_bytes = self.tool_text_bytes;
        while self.tools.len().saturating_sub(remove) > MAX_TOOL_VIEWS
            || retained_text_bytes > MAX_TOOL_TEXT_BYTES
        {
            let Some(tool) = self.tools.get(remove) else {
                break;
            };
            retained_text_bytes = retained_text_bytes.saturating_sub(tool_text_bytes(tool));
            remove += 1;
        }
        if remove == 0 {
            return;
        }
        self.tools.drain(..remove);
        self.tool_text_bytes = retained_text_bytes;
        let assistant_index = self.assistant_index;
        let mut retained = Vec::with_capacity(self.transcript.len());
        let mut new_assistant_index = None;
        for (old_index, mut item) in self.transcript.drain(..).enumerate() {
            if let TranscriptItem::Tool(index) = &mut item {
                if *index < remove {
                    continue;
                }
                *index -= remove;
            }
            if assistant_index == Some(old_index) {
                new_assistant_index = Some(retained.len());
            }
            retained.push(item);
        }
        self.transcript = retained;
        self.assistant_index = new_assistant_index;
        self.clear_selection();
    }

    #[cfg(test)]
    fn tool_finished(&mut self, name: &str, failed: bool) {
        if let Some(tool) = self.tools.iter_mut().rev().find(|tool| tool.name == name) {
            tool.state = if failed {
                ToolState::Failed
            } else {
                ToolState::Completed
            };
        }
    }

    fn tool_finished_with_id(&mut self, call_id: &str, failed: bool, detail: &str) -> bool {
        if let Some(tool) = self.tools.iter_mut().rev().find(|tool| {
            tool.call_key
                .as_ref()
                .is_some_and(|key| key.matches(call_id))
        }) {
            tool.state = if failed {
                ToolState::Failed
            } else {
                ToolState::Completed
            };
            let previous_detail_bytes = tool.detail.as_ref().map_or(0, String::len);
            tool.detail = Some(truncate(&redact_sensitive_text(detail), 2_000));
            self.tool_text_bytes = self
                .tool_text_bytes
                .saturating_sub(previous_detail_bytes)
                .saturating_add(tool.detail.as_ref().map_or(0, String::len));
            self.compact_tool_history();
            true
        } else {
            false
        }
    }

    fn reset_composer(&mut self) {
        self.composer.clear();
        self.composer_cursor = 0;
        self.composer_cursor_set = false;
        self.slash_menu.update(&self.composer);
    }

    fn replace_composer(&mut self, value: String) {
        self.composer = bounded_text(value, MAX_COMPOSER_BYTES);
        self.composer_cursor = self.composer.len();
        self.composer_cursor_set = true;
        self.slash_menu.update(&self.composer);
    }

    fn insert_composer_text(&mut self, text: &str) -> bool {
        if text.is_empty()
            || self.pending_approval.is_some()
            || self.help_visible
            || self.permissions_visible
            || self.language_visible
        {
            return false;
        }
        self.clear_selection();
        self.ensure_composer_cursor();
        let normalized = bounded_text(
            text.replace("\r\n", "\n").replace('\r', "\n"),
            MAX_COMPOSER_BYTES,
        );
        let remaining = MAX_COMPOSER_BYTES.saturating_sub(self.composer.len());
        let normalized = bounded_text(normalized, remaining);
        if normalized.is_empty() {
            return false;
        }
        self.composer.insert_str(self.composer_cursor, &normalized);
        self.composer_cursor += normalized.len();
        self.slash_menu.update(&self.composer);
        true
    }

    fn ensure_composer_cursor(&mut self) {
        if !self.composer_cursor_set {
            self.composer_cursor = self.composer.len();
            self.composer_cursor_set = true;
        }
        self.composer_cursor = self.composer_cursor.min(self.composer.len());
        while self.composer_cursor > 0 && !self.composer.is_char_boundary(self.composer_cursor) {
            self.composer_cursor -= 1;
        }
    }

    fn submit(&mut self, input: &str) -> UiAction {
        let input = input.trim();
        self.reset_composer();
        match input {
            "" => UiAction::None,
            "/help" => {
                self.help_visible = true;
                UiAction::Redraw
            }
            "/copy" => UiAction::Copy,
            "/clear" => UiAction::Clear,
            "/details" => UiAction::ToggleDetails,
            "/permissions" => UiAction::OpenPermissions,
            "/language" => UiAction::OpenLanguage,
            "/new" => UiAction::NewSession,
            "/status" => UiAction::Status,
            "/review" => UiAction::Submit(REVIEW_TASK.into()),
            "/simplify" => UiAction::Submit(SIMPLIFY_TASK.into()),
            "/update" => UiAction::Update,
            "/exit" | "/quit" => UiAction::Exit,
            task => UiAction::Submit(task.to_owned()),
        }
    }

    #[cfg(test)]
    fn handle_key(&mut self, key: KeyEvent, running: bool) -> UiAction {
        self.handle_key_with_width(key, running, 78)
    }

    fn handle_key_with_width(
        &mut self,
        key: KeyEvent,
        running: bool,
        composer_width: usize,
    ) -> UiAction {
        if self.pending_approval.is_some() {
            return match key.code {
                KeyCode::Char('y' | 'Y') => UiAction::Approval(ApprovalDecision::ApproveOnce),
                KeyCode::Char('s' | 'S') => UiAction::Approval(ApprovalDecision::ApproveSession),
                KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                    UiAction::Approval(ApprovalDecision::Deny)
                }
                KeyCode::Up => {
                    self.approval_selection = self.approval_selection.saturating_sub(1);
                    UiAction::Redraw
                }
                KeyCode::Down => {
                    self.approval_selection = (self.approval_selection + 1).min(2);
                    UiAction::Redraw
                }
                KeyCode::Enter => UiAction::Approval(ApprovalView::decision_for_selection(
                    self.approval_selection,
                )),
                _ => UiAction::None,
            };
        }
        if self.permissions_visible {
            return match key.code {
                KeyCode::Char('1') => UiAction::SetApprovalMode(ApprovalMode::ApproveAll),
                KeyCode::Char('2') => UiAction::SetApprovalMode(ApprovalMode::Interactive),
                KeyCode::Char('3') => UiAction::SetApprovalMode(ApprovalMode::DenyAll),
                KeyCode::Up => {
                    self.permission_selection = self.permission_selection.saturating_sub(1);
                    UiAction::Redraw
                }
                KeyCode::Down => {
                    self.permission_selection = (self.permission_selection + 1).min(2);
                    UiAction::Redraw
                }
                KeyCode::Enter => UiAction::SetApprovalMode(approval_mode_for_selection(
                    self.permission_selection,
                )),
                KeyCode::Esc => {
                    self.permissions_visible = false;
                    UiAction::Redraw
                }
                _ => UiAction::None,
            };
        }
        if self.language_visible {
            return match key.code {
                KeyCode::Char('1') => UiAction::SetLanguage(Language::English),
                KeyCode::Char('2') => UiAction::SetLanguage(Language::Chinese),
                KeyCode::Up => {
                    self.language_selection = self.language_selection.saturating_sub(1);
                    UiAction::Redraw
                }
                KeyCode::Down => {
                    self.language_selection = (self.language_selection + 1).min(1);
                    UiAction::Redraw
                }
                KeyCode::Enter => {
                    UiAction::SetLanguage(language_for_selection(self.language_selection))
                }
                KeyCode::Esc => {
                    self.language_visible = false;
                    UiAction::Redraw
                }
                _ => UiAction::None,
            };
        }
        if self.help_visible {
            if matches!(key.code, KeyCode::Esc | KeyCode::Enter) {
                self.help_visible = false;
                return UiAction::Redraw;
            }
            return UiAction::None;
        }
        if self.slash_menu.is_visible(&self.composer) {
            match key.code {
                KeyCode::Up => {
                    self.slash_menu.move_up();
                    return UiAction::Redraw;
                }
                KeyCode::Down => {
                    self.slash_menu.move_down();
                    return UiAction::Redraw;
                }
                KeyCode::Tab => {
                    if let Some(completed) = self.slash_menu.complete() {
                        self.replace_composer(completed);
                    }
                    return UiAction::Redraw;
                }
                KeyCode::Enter => {
                    let selected = self
                        .slash_menu
                        .selected_name()
                        .map(|name| format!("/{name}"));
                    if selected.as_deref() == Some(self.composer.trim()) {
                        let input = self.composer.clone();
                        return self.submit(&input);
                    }
                    if let Some(completed) = self.slash_menu.complete() {
                        self.replace_composer(completed);
                    }
                    return UiAction::Redraw;
                }
                KeyCode::Esc => {
                    self.slash_menu.dismiss();
                    return UiAction::Redraw;
                }
                _ => {}
            }
        }
        if key
            .modifiers
            .contains(KeyModifiers::CONTROL | KeyModifiers::SHIFT)
            && matches!(key.code, KeyCode::Char('c' | 'C'))
        {
            return UiAction::Copy;
        }
        if key.modifiers == KeyModifiers::CONTROL && matches!(key.code, KeyCode::Char('c')) {
            return if running {
                UiAction::Cancel
            } else {
                UiAction::Exit
            };
        }
        match key.code {
            KeyCode::Esc if running => UiAction::Cancel,
            KeyCode::Esc => UiAction::Exit,
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                if self.insert_composer_text("\n") {
                    UiAction::Redraw
                } else {
                    UiAction::None
                }
            }
            KeyCode::Enter if running => UiAction::None,
            KeyCode::Enter => {
                let input = self.composer.clone();
                self.submit(&input)
            }
            KeyCode::Backspace => {
                self.clear_selection();
                self.ensure_composer_cursor();
                if self.composer_cursor > 0 {
                    let start = previous_grapheme_boundary(&self.composer, self.composer_cursor);
                    self.composer.drain(start..self.composer_cursor);
                    self.composer_cursor = start;
                }
                self.slash_menu.update(&self.composer);
                UiAction::Redraw
            }
            KeyCode::Delete => {
                self.clear_selection();
                self.ensure_composer_cursor();
                if self.composer_cursor < self.composer.len() {
                    let end = next_grapheme_boundary(&self.composer, self.composer_cursor);
                    self.composer.drain(self.composer_cursor..end);
                }
                self.slash_menu.update(&self.composer);
                UiAction::Redraw
            }
            KeyCode::Left => {
                self.clear_selection();
                self.ensure_composer_cursor();
                self.composer_cursor =
                    previous_grapheme_boundary(&self.composer, self.composer_cursor);
                UiAction::Redraw
            }
            KeyCode::Right => {
                self.clear_selection();
                self.ensure_composer_cursor();
                self.composer_cursor = next_grapheme_boundary(&self.composer, self.composer_cursor);
                UiAction::Redraw
            }
            KeyCode::Char('?') if self.composer.is_empty() => {
                self.help_visible = true;
                UiAction::Redraw
            }
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                if self.insert_composer_text(&character.to_string()) {
                    UiAction::Redraw
                } else {
                    UiAction::None
                }
            }
            KeyCode::Up if self.composer_cursor_set => {
                self.clear_selection();
                self.move_composer_vertical(-1, composer_width);
                UiAction::Redraw
            }
            KeyCode::Down if self.composer_cursor_set => {
                self.clear_selection();
                self.move_composer_vertical(1, composer_width);
                UiAction::Redraw
            }
            KeyCode::Up => {
                self.clear_selection();
                self.scroll_from_tail = self.scroll_from_tail.saturating_add(3);
                UiAction::Redraw
            }
            KeyCode::Down => {
                self.clear_selection();
                self.scroll_from_tail = self.scroll_from_tail.saturating_sub(3);
                UiAction::Redraw
            }
            KeyCode::PageUp => {
                self.clear_selection();
                self.scroll_from_tail = self.scroll_from_tail.saturating_add(10);
                UiAction::Redraw
            }
            KeyCode::PageDown => {
                self.clear_selection();
                self.scroll_from_tail = self.scroll_from_tail.saturating_sub(10);
                UiAction::Redraw
            }
            KeyCode::Home => {
                self.clear_selection();
                if self.composer_cursor_set && !self.composer.is_empty() {
                    self.ensure_composer_cursor();
                    let (row, _) = cursor_position(
                        &self.composer[..self.composer_cursor],
                        composer_width.max(1),
                    );
                    self.composer_cursor =
                        composer_cursor_at(&self.composer, row, 0, composer_width);
                } else {
                    self.scroll_from_tail = usize::MAX;
                }
                UiAction::Redraw
            }
            KeyCode::End => {
                self.clear_selection();
                if self.composer_cursor_set && !self.composer.is_empty() {
                    self.ensure_composer_cursor();
                    let (row, _) = cursor_position(
                        &self.composer[..self.composer_cursor],
                        composer_width.max(1),
                    );
                    self.composer_cursor =
                        composer_cursor_at(&self.composer, row, usize::MAX, composer_width);
                } else {
                    self.scroll_from_tail = 0;
                }
                UiAction::Redraw
            }
            _ => UiAction::None,
        }
    }

    fn move_composer_vertical(&mut self, direction: isize, width: usize) {
        self.ensure_composer_cursor();
        let (row, column) = cursor_position(&self.composer[..self.composer_cursor], width.max(1));
        let target_row = if direction < 0 {
            row.saturating_sub(1)
        } else {
            row.saturating_add(1)
        };
        self.composer_cursor = composer_cursor_at(&self.composer, target_row, column, width);
    }

    fn handle_mouse(
        &mut self,
        mouse: crossterm::event::MouseEvent,
        running: bool,
        size: (u16, u16),
    ) -> bool {
        if size.0 < MIN_WIDTH || size.1 < MIN_HEIGHT {
            return false;
        }
        if self.pending_approval.is_some()
            || self.help_visible
            || self.permissions_visible
            || self.language_visible
        {
            return false;
        }
        if mouse.modifiers.contains(KeyModifiers::SHIFT) {
            return false;
        }
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                if let Some(point) = self.transcript_point(mouse, size, running) {
                    self.mouse_anchor = Some(point);
                    self.selection = None;
                    self.selection_text = None;
                    return true;
                }
                let pane = bottom_pane(self, usize::from(size.0), running, "");
                let pane_height = u16::try_from(pane.lines.len().saturating_add(1))
                    .unwrap_or(size.1)
                    .min(size.1);
                let inner_y = size.1.saturating_sub(pane_height).saturating_add(1);
                let row = usize::from(mouse.row.saturating_sub(inner_y));
                let (_, end) = visible_bottom_range(
                    pane.lines.len(),
                    usize::from(pane_height.saturating_sub(1)),
                );
                let start = pane
                    .lines
                    .len()
                    .saturating_sub(usize::from(pane_height.saturating_sub(1)));
                if mouse.row < inner_y || row + start >= end {
                    return false;
                }
                let Some((composer_start, composer_end)) = pane.composer_rows else {
                    return false;
                };
                let absolute_row = row + start;
                if absolute_row < composer_start || absolute_row >= composer_end {
                    return false;
                }
                self.clear_selection();
                let target_column = usize::from(mouse.column.saturating_sub(2));
                let row_offset = absolute_row.saturating_sub(composer_start);
                self.composer_cursor = composer_cursor_at(
                    &self.composer,
                    row_offset,
                    target_column,
                    usize::from(size.0).saturating_sub(2),
                );
                self.composer_cursor_set = true;
                true
            }
            MouseEventKind::Drag(MouseButton::Left) => {
                let Some(point) = self.transcript_point(mouse, size, running) else {
                    return false;
                };
                self.update_selection(point, usize::from(size.0));
                true
            }
            MouseEventKind::Up(MouseButton::Left) => {
                if self.mouse_anchor.is_none() {
                    return false;
                }
                if let Some(point) = self.transcript_point(mouse, size, running) {
                    self.update_selection(point, usize::from(size.0));
                }
                self.mouse_anchor = None;
                self.selection.is_some()
            }
            MouseEventKind::ScrollUp => {
                self.clear_selection();
                self.scroll_from_tail = self.scroll_from_tail.saturating_add(3);
                true
            }
            MouseEventKind::ScrollDown => {
                self.clear_selection();
                self.scroll_from_tail = self.scroll_from_tail.saturating_sub(3);
                true
            }
            _ => false,
        }
    }

    fn transcript_point(
        &self,
        mouse: crossterm::event::MouseEvent,
        size: (u16, u16),
        running: bool,
    ) -> Option<TextPoint> {
        let width = usize::from(size.0);
        let pane = bottom_pane(self, width, running, "");
        let pane_height = u16::try_from(pane.lines.len().saturating_add(1))
            .ok()?
            .min(size.1);
        let content_height = usize::from(size.1.saturating_sub(pane_height));
        let row = usize::from(mouse.row);
        if row >= content_height {
            return None;
        }
        let lines = self.lines(width);
        let (start, end) =
            visible_transcript_range(lines.len(), content_height, self.scroll_from_tail);
        let absolute_row = start + row;
        (absolute_row < end).then_some(TextPoint {
            row: absolute_row,
            column: usize::from(mouse.column),
        })
    }

    fn lines(&self, width: usize) -> Vec<Line> {
        let mut lines = Vec::new();
        for item in &self.transcript {
            match item {
                TranscriptItem::User(text) => {
                    lines.extend(wrap_line("› ", text, width, Tone::User))
                }
                TranscriptItem::Assistant(text) if !text.is_empty() => {
                    lines.extend(
                        markdown::render_markdown(text, width)
                            .into_iter()
                            .map(Line::markdown),
                    );
                }
                TranscriptItem::Assistant(_) => {}
                TranscriptItem::Reasoning(text) if !text.is_empty() => {
                    lines.push(Line::new(
                        format!("╰─ {}", self.language.text("Thinking", "思考")),
                        Tone::Reasoning,
                    ));
                    lines.extend(
                        markdown::render_markdown(text, width.saturating_sub(2))
                            .into_iter()
                            .map(|line| Line::markdown_with_tone(line, Tone::Reasoning)),
                    );
                }
                TranscriptItem::Reasoning(_) => {}
                TranscriptItem::Activity(text) => {
                    lines.extend(wrap_line("• ", text, width, Tone::Activity))
                }
                TranscriptItem::Notice(text) => {
                    lines.extend(wrap_line("· ", text, width, Tone::Muted))
                }
                TranscriptItem::Error(text) => {
                    lines.extend(wrap_line("✗ ", text, width, Tone::Error))
                }
                TranscriptItem::Tool(index) => {
                    if let Some(tool) = self.tools.get(*index) {
                        let (title, tone) = match tool.state {
                            ToolState::Running => (
                                format!(
                                    "• {} {} ({})",
                                    self.language.text("Running", "运行中"),
                                    tool.name,
                                    tool.preview
                                ),
                                Tone::Activity,
                            ),
                            ToolState::Completed => (
                                format!(
                                    "• {} {} ({})",
                                    self.language.text("Ran", "已运行"),
                                    tool.name,
                                    tool.preview
                                ),
                                Tone::Muted,
                            ),
                            ToolState::Failed => (
                                format!(
                                    "• {} {} ({})",
                                    self.language.text("Failed", "失败"),
                                    tool.name,
                                    tool.preview
                                ),
                                Tone::Error,
                            ),
                        };
                        lines.extend(wrap_with_prefixes("", "  ", &title, width, tone));
                        if let Some(detail) =
                            tool.detail.as_deref().filter(|detail| !detail.is_empty())
                        {
                            let detail = if self.details_visible {
                                detail
                            } else {
                                first_detail_line(detail, self.language)
                            };
                            lines.extend(wrap_with_prefixes(
                                "  └ ",
                                "    ",
                                detail,
                                width,
                                Tone::Muted,
                            ));
                        }
                    }
                }
                TranscriptItem::TurnSummary(elapsed) => {
                    lines.push(Line::new(
                        format_turn_summary(*elapsed, width, self.language),
                        Tone::Muted,
                    ));
                }
            }
        }
        for child_session_id in &self.child_order {
            let Some(child) = self.child_views.get(child_session_id) else {
                continue;
            };
            lines.push(Line::new(
                format!(
                    "╭─ {} {} [{}]",
                    self.language.text("delegate", "委派"),
                    child.role,
                    child_state_label(&child.state, self.language)
                ),
                Tone::Delegation,
            ));
            if !child.reasoning.trim().is_empty() {
                lines.push(Line::new(
                    format!("│  {}", self.language.text("Thinking", "思考")),
                    Tone::Reasoning,
                ));
                lines.extend(
                    markdown::render_markdown(&child.reasoning, width.saturating_sub(2))
                        .into_iter()
                        .map(|line| Line::markdown_with_tone(line, Tone::Reasoning)),
                );
            }
            if !child.output.trim().is_empty() {
                lines.extend(
                    markdown::render_markdown(&child.output, width.saturating_sub(2))
                        .into_iter()
                        .map(|line| Line::markdown_with_tone(line, Tone::Delegation)),
                );
            }
        }
        lines
    }
}

#[derive(Debug, Clone, Copy)]
enum Tone {
    User,
    Assistant,
    Muted,
    Activity,
    Reasoning,
    Delegation,
    Error,
}

#[derive(Debug, Clone)]
struct Line {
    text: String,
    tone: Tone,
    markdown: Option<markdown::MarkdownLine>,
}

impl Line {
    fn new(text: impl Into<String>, tone: Tone) -> Self {
        Self {
            text: text.into(),
            tone,
            markdown: None,
        }
    }

    fn markdown(markdown: markdown::MarkdownLine) -> Self {
        Self::markdown_with_tone(markdown, Tone::Assistant)
    }

    fn markdown_with_tone(markdown: markdown::MarkdownLine, tone: Tone) -> Self {
        Self {
            text: markdown.plain_text(),
            tone,
            markdown: Some(markdown),
        }
    }
}

struct BottomPane {
    lines: Vec<Line>,
    cursor: Option<(usize, usize)>,
    composer_rows: Option<(usize, usize)>,
}

fn render(
    terminal: &mut TerminalSession,
    view: &ChatView,
    _title: &str,
    model_label: &str,
    active: bool,
) -> io::Result<()> {
    let cursor_visible = view.pending_approval.is_none()
        && !view.help_visible
        && !view.permissions_visible
        && !view.language_visible;
    terminal.set_cursor_visible(cursor_visible)?;
    terminal
        .terminal
        .draw(|frame| render_frame(frame, view, model_label, active))?;
    Ok(())
}

fn render_frame(frame: &mut Frame, view: &ChatView, model_label: &str, active: bool) {
    let area = frame.area();
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        frame.render_widget(
            Paragraph::new(view.language.text(
                "Focus needs a terminal of at least 36x10. Resize to continue.",
                "Focus 需要至少 36x10 的终端窗口。请调整窗口大小后继续。",
            ))
            .style(Style::default().fg(Color::Yellow)),
            area,
        );
        return;
    }

    let pane = bottom_pane(view, usize::from(area.width), active, model_label);
    let pane_height = u16::try_from(pane.lines.len().saturating_add(1))
        .unwrap_or(area.height)
        .min(area.height);
    let regions = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(pane_height)])
        .split(area);
    let content = regions[0];
    let pane_area = regions[1];

    let lines = view.lines(usize::from(content.width));
    let (start, end) = visible_transcript_range(
        lines.len(),
        usize::from(content.height),
        view.scroll_from_tail,
    );
    let selection = view.selection.map(TextSelection::ordered);
    frame.render_widget(
        Paragraph::new(Text::from(
            lines[start..end]
                .iter()
                .enumerate()
                .map(|(offset, line)| {
                    let row = start + offset;
                    let range = selection.and_then(|(selected_start, selected_end)| {
                        if row < selected_start.row || row > selected_end.row {
                            return None;
                        }
                        let from = if row == selected_start.row {
                            selected_start.column
                        } else {
                            0
                        };
                        let to = if row == selected_end.row {
                            selected_end.column
                        } else {
                            display_width(&line.text)
                        };
                        Some((from, to))
                    });
                    line.to_ratatui_selected(range)
                })
                .collect::<Vec<_>>(),
        )),
        content,
    );

    let border = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = border.inner(pane_area);
    frame.render_widget(border, pane_area);
    let (pane_start, pane_end) = visible_bottom_range(pane.lines.len(), usize::from(inner.height));
    frame.render_widget(
        Paragraph::new(Text::from(
            pane.lines[pane_start..pane_end]
                .iter()
                .map(Line::to_ratatui)
                .collect::<Vec<_>>(),
        )),
        inner,
    );

    if let Some((column, row)) = pane.cursor
        && row >= pane_start
        && row < pane_end
    {
        frame.set_cursor_position((
            u16::try_from(column)
                .unwrap_or(inner.width.saturating_sub(1))
                .min(inner.width.saturating_sub(1)),
            inner.y
                + u16::try_from(row.saturating_sub(pane_start))
                    .unwrap_or(inner.height.saturating_sub(1)),
        ));
    }
}

fn visible_transcript_range(
    line_count: usize,
    viewport_height: usize,
    scroll_from_tail: usize,
) -> (usize, usize) {
    let max_offset = line_count.saturating_sub(viewport_height);
    let offset = scroll_from_tail.min(max_offset);
    let end = line_count.saturating_sub(offset);
    (end.saturating_sub(viewport_height), end)
}

fn visible_bottom_range(line_count: usize, viewport_height: usize) -> (usize, usize) {
    let end = line_count;
    (end.saturating_sub(viewport_height), end)
}

fn bottom_pane(view: &ChatView, width: usize, active: bool, model_label: &str) -> BottomPane {
    if let Some(approval) = &view.pending_approval {
        return text_pane(
            &approval.rendered_with_language(view.approval_selection, view.language),
            width,
            Tone::Assistant,
        );
    }
    if view.help_visible {
        return text_pane(
            view.language.text(
                "  Commands\n\n  /copy        Copy the current response\n  /language    Change interface language\n  /permissions Change approval mode\n  /new         Start a new session\n  /clear       Clear this screen\n  /details     Toggle tool output\n  /status      Show session mode\n  /review      Review and fix real repository issues\n  /simplify    Simplify while preserving behavior\n  /exit        Leave Focus\n  /quit        Leave Focus\n\n  Press esc to return",
                "  命令\n\n  /copy        复制本轮回答\n  /language    切换界面语言\n  /permissions 更改权限模式\n  /new         开始新会话\n  /clear       清空当前界面\n  /details     切换工具详情\n  /status      查看会话模式\n  /review      审查并修复仓库中的真实问题\n  /simplify    在保持行为的前提下简化\n  /exit        退出 Focus\n  /quit        退出 Focus\n\n  按 Esc 返回",
            ),
            width,
            Tone::Muted,
        );
    }
    if view.permissions_visible {
        return permissions_pane(view.permission_selection, width, view.language);
    }
    if view.language_visible {
        return language_pane(view.language_selection, view.language, width);
    }

    let mut lines = Vec::new();
    if view.should_refresh_status(active) {
        let elapsed = view
            .turn_started_at
            .map(|started| started.elapsed().as_secs())
            .unwrap_or(0);
        lines.push(Line::new(
            format!(
                "{} {} ({elapsed}s • {})",
                activity_glyph(view.activity_frame),
                view.language.text("Working", "工作中"),
                view.language.text("esc to interrupt", "Esc 中断"),
            ),
            Tone::Activity,
        ));
        if !view.reasoning_buffer.trim().is_empty() {
            let label = view.thinking_header().map_or_else(
                || view.language.text("Thinking", "思考").into(),
                |header| {
                    format!(
                        "{}{}{}",
                        view.language.text("Thinking", "思考"),
                        view.language.text(": ", "："),
                        header
                    )
                },
            );
            lines.push(Line::new(format!("╰─ {label}"), Tone::Reasoning));
            lines.extend(
                markdown::render_markdown(&view.reasoning_buffer, width.saturating_sub(2))
                    .into_iter()
                    .map(|line| Line::markdown_with_tone(line, Tone::Reasoning)),
            );
        }
        lines.push(Line::new("", Tone::Muted));
    }

    if view.slash_menu.is_visible(&view.composer) {
        for command in view.slash_menu.matches().into_iter().take(6) {
            let selected = view.slash_menu.selected_name() == Some(command.name());
            let cursor = if selected { "›" } else { " " };
            lines.push(Line::new(
                format!(
                    "{cursor} /{:<12} {}",
                    command.name(),
                    command.localized_description(view.language == Language::Chinese)
                ),
                if selected {
                    Tone::Assistant
                } else {
                    Tone::Muted
                },
            ));
        }
        lines.push(Line::new("", Tone::Muted));
    }

    let composer_row = lines.len();
    let placeholder = view
        .language
        .text("Ask Focus to do anything", "告诉 Focus 你要做什么");
    let composer = if view.composer.is_empty() {
        placeholder
    } else {
        view.composer.as_str()
    };
    let composer_lines = wrap_line("› ", composer, width, Tone::User);
    let composer_line_count = composer_lines.len();
    let cursor = if view.composer.is_empty() {
        Some((display_width("› "), composer_row))
    } else {
        let prefix_width = display_width("› ");
        let cursor = if view.composer_cursor_set {
            view.composer_cursor
        } else {
            view.composer.len()
        };
        let (row_offset, column) = cursor_position(
            &view.composer[..cursor.min(view.composer.len())],
            width.saturating_sub(prefix_width),
        );
        let row = composer_row + row_offset;
        Some((
            prefix_width + column,
            row.min(composer_row + composer_lines.len().saturating_sub(1)),
        ))
    };
    lines.extend(composer_lines);
    lines.push(Line::new("", Tone::Muted));
    lines.push(Line::new(
        footer_context_line(width, model_label, view.approval_mode, view.language),
        Tone::Muted,
    ));

    BottomPane {
        lines,
        cursor,
        composer_rows: Some((composer_row, composer_row + composer_line_count)),
    }
}

fn text_pane(text: &str, width: usize, tone: Tone) -> BottomPane {
    let mut lines = Vec::new();
    for line in text.lines() {
        lines.extend(wrap_with_prefixes("", "", line, width, tone));
    }
    BottomPane {
        lines,
        cursor: None,
        composer_rows: None,
    }
}

fn composer_cursor_at(text: &str, target_row: usize, target_column: usize, width: usize) -> usize {
    let width = width.max(1);
    let mut row = 0;
    let mut column = 0;
    let mut byte = 0;

    for grapheme in text.graphemes(true) {
        if matches!(grapheme, "\n" | "\r" | "\r\n") {
            if row == target_row {
                return byte;
            }
            row += 1;
            column = 0;
            byte += grapheme.len();
            continue;
        }

        let grapheme_width = UnicodeWidthStr::width(grapheme);
        if column > 0 && column + grapheme_width > width {
            row += 1;
            column = 0;
        }
        if row > target_row {
            return byte;
        }
        if row == target_row && target_column < column + grapheme_width {
            return byte;
        }
        column += grapheme_width;
        byte += grapheme.len();
    }

    if row == target_row { byte } else { text.len() }
}

fn footer_context_line(
    width: usize,
    model_label: &str,
    approval_mode: ApprovalMode,
    language: Language,
) -> String {
    let left = language.text("  ? for shortcuts", "  ? 查看快捷键");
    let right = truncate(
        &format!(
            "{} · {}",
            model_label,
            approval_mode_label(approval_mode, language)
        ),
        32,
    );
    let padding = width.saturating_sub(display_width(left) + display_width(&right));
    format!("{left}{}{}", " ".repeat(padding.max(1)), right)
}

impl Tone {
    fn style(self) -> Style {
        match self {
            Self::User => Style::default().fg(Color::Cyan),
            Self::Assistant => Style::default().fg(Color::White),
            Self::Muted => Style::default().fg(Color::DarkGray),
            Self::Activity => Style::default().fg(Color::Yellow),
            Self::Reasoning => Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::DIM | Modifier::ITALIC),
            Self::Delegation => Style::default().fg(Color::Magenta),
            Self::Error => Style::default().fg(Color::Red),
        }
    }
}

impl Line {
    fn markdown_style(tone: Tone, markdown: markdown::MarkdownStyle) -> Style {
        let mut style = tone.style();
        if markdown.heading.is_some() || markdown.strong {
            style = style.add_modifier(Modifier::BOLD);
        }
        if markdown.emphasis {
            style = style.add_modifier(Modifier::ITALIC);
        }
        if markdown.code {
            style = style.fg(Color::Yellow);
        }
        if markdown.link {
            style = style.fg(Color::Blue).add_modifier(Modifier::UNDERLINED);
        }
        style
    }

    fn to_ratatui(&self) -> RatLine<'static> {
        let Some(markdown) = &self.markdown else {
            return RatLine::from(Span::styled(self.text.clone(), self.tone.style()));
        };
        RatLine::from(
            markdown
                .spans
                .iter()
                .map(|span| {
                    Span::styled(
                        span.text.clone(),
                        Self::markdown_style(self.tone, span.style),
                    )
                })
                .collect::<Vec<_>>(),
        )
    }

    fn to_ratatui_selected(&self, range: Option<(usize, usize)>) -> RatLine<'static> {
        let Some((start, end)) = range.filter(|(start, end)| start < end) else {
            return self.to_ratatui();
        };
        let mut spans = Vec::new();
        let mut column = 0;
        if let Some(markdown) = &self.markdown {
            for span in &markdown.spans {
                let base = Self::markdown_style(self.tone, span.style);
                for grapheme in span.text.graphemes(true) {
                    let width = UnicodeWidthStr::width(grapheme);
                    let mut style = base;
                    if column < end && column + width > start {
                        style = style.bg(Color::DarkGray).fg(Color::White);
                    }
                    spans.push(Span::styled(grapheme.to_owned(), style));
                    column += width;
                }
            }
            return RatLine::from(spans);
        }
        for grapheme in self.text.graphemes(true) {
            let width = UnicodeWidthStr::width(grapheme);
            let mut style = self.tone.style();
            if column < end && column + width > start {
                style = style.bg(Color::DarkGray).fg(Color::White);
            }
            spans.push(Span::styled(grapheme.to_owned(), style));
            column += width;
        }
        RatLine::from(spans)
    }
}

fn first_detail_line(detail: &str, language: Language) -> &str {
    detail
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(language.text("completed", "已完成"))
}

fn format_turn_summary(elapsed: Duration, width: usize, language: Language) -> String {
    let seconds = elapsed.as_secs();
    let elapsed = if language == Language::Chinese {
        if seconds >= 60 {
            format!("{} 分 {:02} 秒", seconds / 60, seconds % 60)
        } else {
            format!("{seconds} 秒")
        }
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    };
    let label = format!(" {} {} ", language.text("Worked for", "用时"), elapsed);
    let remaining = width.saturating_sub(display_width(&label));
    let left = remaining / 2;
    format!(
        "{}{}{}",
        "─".repeat(left),
        label,
        "─".repeat(remaining.saturating_sub(left))
    )
}

fn activity_glyph(frame: usize) -> &'static str {
    const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    SPINNER[frame % SPINNER.len()]
}

fn permissions_pane(selected: usize, width: usize, language: Language) -> BottomPane {
    let options = [
        (
            "YOLO mode",
            "Allow Focus to run tools without asking.",
            "YOLO 模式",
            "允许 Focus 直接运行工具。",
        ),
        (
            "Ask for approval",
            "Confirm each write, execute, and network action.",
            "询问批准",
            "每次写入、执行和网络操作前请求确认。",
        ),
        (
            "Read-only",
            "Deny write, execute, and network actions.",
            "只读模式",
            "拒绝写入、执行和网络操作。",
        ),
    ];
    let mut lines = vec![
        Line::new(
            language.text("  Update Focus permissions", "  更新 Focus 权限"),
            Tone::Assistant,
        ),
        Line::new("", Tone::Muted),
    ];
    for (index, (name, description, name_zh, description_zh)) in options.into_iter().enumerate() {
        let cursor = if index == selected { '›' } else { ' ' };
        let tone = if index == selected {
            Tone::Assistant
        } else {
            Tone::Muted
        };
        lines.extend(wrap_line(
            &format!("{cursor} "),
            &format!(
                "{}  {}",
                language.text(name, name_zh),
                language.text(description, description_zh)
            ),
            width,
            tone,
        ));
    }
    lines.push(Line::new("", Tone::Muted));
    lines.push(Line::new(
        language.text(
            "  Enter to confirm · Esc to cancel",
            "  回车确认 · Esc 取消",
        ),
        Tone::Muted,
    ));
    BottomPane {
        lines,
        cursor: None,
        composer_rows: None,
    }
}

fn language_pane(selected: usize, current: Language, width: usize) -> BottomPane {
    let options = [
        (
            Language::English,
            current.text("English", "English"),
            current.text("Use the English interface.", "使用英文界面。"),
        ),
        (
            Language::Chinese,
            current.text("Chinese", "中文"),
            current.text("Use the Chinese interface.", "使用中文界面。"),
        ),
    ];
    let mut lines = vec![
        Line::new(
            current.text("  Choose Focus language", "  选择 Focus 语言"),
            Tone::Assistant,
        ),
        Line::new("", Tone::Muted),
    ];
    for (index, (language, name, description)) in options.into_iter().enumerate() {
        let cursor = if index == selected { '›' } else { ' ' };
        let marker = if language == current { " *" } else { "  " };
        lines.extend(wrap_line(
            &format!("{cursor} {index_plus}. ", index_plus = index + 1),
            &format!("{name}{marker}  {description}"),
            width,
            if index == selected {
                Tone::Assistant
            } else {
                Tone::Muted
            },
        ));
    }
    lines.push(Line::new("", Tone::Muted));
    lines.push(Line::new(
        current.text(
            "  Enter to confirm · Esc to cancel",
            "  回车确认 · Esc 取消",
        ),
        Tone::Muted,
    ));
    BottomPane {
        lines,
        cursor: None,
        composer_rows: None,
    }
}

fn permission_selection(mode: ApprovalMode) -> usize {
    match mode {
        ApprovalMode::ApproveAll => 0,
        ApprovalMode::Interactive => 1,
        ApprovalMode::DenyAll => 2,
    }
}

fn language_selection(language: Language) -> usize {
    match language {
        Language::English => 0,
        Language::Chinese => 1,
    }
}

fn language_for_selection(selection: usize) -> Language {
    if selection == 1 {
        Language::Chinese
    } else {
        Language::English
    }
}

fn approval_mode_for_selection(selection: usize) -> ApprovalMode {
    match selection {
        1 => ApprovalMode::Interactive,
        2 => ApprovalMode::DenyAll,
        _ => ApprovalMode::ApproveAll,
    }
}

fn approval_mode_label(mode: ApprovalMode, language: Language) -> &'static str {
    match mode {
        ApprovalMode::ApproveAll => language.text("YOLO mode", "YOLO 模式"),
        ApprovalMode::Interactive => language.text("Ask for approval", "询问批准"),
        ApprovalMode::DenyAll => language.text("Read-only", "只读模式"),
    }
}

fn previous_grapheme_boundary(text: &str, cursor: usize) -> usize {
    text[..cursor.min(text.len())]
        .grapheme_indices(true)
        .next_back()
        .map_or(0, |(index, _)| index)
}

fn next_grapheme_boundary(text: &str, cursor: usize) -> usize {
    let cursor = cursor.min(text.len());
    text[cursor..]
        .graphemes(true)
        .next()
        .map_or(text.len(), |grapheme| cursor + grapheme.len())
}

fn cursor_position(text: &str, width: usize) -> (usize, usize) {
    let mut row = 0;
    let mut column = 0;
    for grapheme in text.graphemes(true) {
        if matches!(grapheme, "\n" | "\r" | "\r\n")
            || (column > 0 && column + UnicodeWidthStr::width(grapheme) > width.max(1))
        {
            row += 1;
            column = 0;
            if matches!(grapheme, "\n" | "\r" | "\r\n") {
                continue;
            }
        }
        column += UnicodeWidthStr::width(grapheme);
    }
    (row, column)
}

fn wrap_line(prefix: &str, text: &str, width: usize, tone: Tone) -> Vec<Line> {
    wrap_with_prefixes(prefix, "  ", text, width, tone)
}

fn wrap_with_prefixes(
    prefix: &str,
    continuation: &str,
    text: &str,
    width: usize,
    tone: Tone,
) -> Vec<Line> {
    wrap_with_widths(
        text,
        width.saturating_sub(display_width(prefix)).max(1),
        width.saturating_sub(display_width(continuation)).max(1),
    )
    .into_iter()
    .enumerate()
    .map(|(index, line)| {
        let leader = if index == 0 { prefix } else { continuation };
        Line::new(format!("{leader}{line}"), tone)
    })
    .collect()
}

fn wrap_with_widths(text: &str, first_width: usize, continuation_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut line_width = 0;
    let mut width = first_width.max(1);
    for grapheme in text.graphemes(true) {
        if matches!(grapheme, "\n" | "\r" | "\r\n") {
            lines.push(std::mem::take(&mut line));
            line_width = 0;
            width = continuation_width.max(1);
            continue;
        }
        let grapheme_width = UnicodeWidthStr::width(grapheme);
        if !line.is_empty() && line_width + grapheme_width > width {
            lines.push(std::mem::take(&mut line));
            line_width = 0;
            width = continuation_width.max(1);
        }
        line.push_str(grapheme);
        line_width += grapheme_width;
    }
    lines.push(line);
    lines
}

fn truncate(value: &str, width: usize) -> String {
    if display_width(value) <= width {
        return value.to_owned();
    }
    if width <= 3 {
        let mut rendered = String::new();
        let mut rendered_width = 0;
        for grapheme in value.graphemes(true) {
            let grapheme_width = UnicodeWidthStr::width(grapheme);
            if rendered_width + grapheme_width > width {
                break;
            }
            rendered.push_str(grapheme);
            rendered_width += grapheme_width;
        }
        return rendered;
    }
    let mut rendered = String::new();
    let mut rendered_width = 0;
    for grapheme in value.graphemes(true) {
        let grapheme_width = UnicodeWidthStr::width(grapheme);
        if rendered_width + grapheme_width > width - 3 {
            break;
        }
        rendered.push_str(grapheme);
        rendered_width += grapheme_width;
    }
    rendered.push_str("...");
    rendered
}

fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn bound_transcript_text(text: &mut String) {
    bound_text(text, MAX_TRANSCRIPT_TEXT_BYTES);
}

fn bounded_text(mut text: String, max_bytes: usize) -> String {
    bound_text(&mut text, max_bytes);
    text
}

fn bound_text(text: &mut String, max_bytes: usize) {
    if text.len() <= max_bytes {
        return;
    }
    let mut end = max_bytes.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
}

fn append_bounded_text(target: &mut String, text: &str, max_bytes: usize) -> bool {
    let remaining = max_bytes.saturating_sub(target.len());
    if remaining == 0 {
        return false;
    }
    let mut text = text.to_owned();
    bound_text(&mut text, remaining);
    if text.is_empty() {
        return false;
    }
    target.push_str(&text);
    true
}

fn serialized_event_bytes(event: &Event) -> Option<usize> {
    serde_json::to_vec(event).ok().map(|payload| payload.len())
}

fn serialized_events_bytes(events: &[Event]) -> usize {
    events.iter().filter_map(serialized_event_bytes).sum()
}

fn slice_display_columns(text: &str, start: usize, end: usize) -> String {
    if start >= end {
        return String::new();
    }
    let mut result = String::new();
    let mut column = 0;
    for grapheme in text.graphemes(true) {
        let next = column + UnicodeWidthStr::width(grapheme);
        if next > start && column < end {
            result.push_str(grapheme);
        }
        if column >= end {
            break;
        }
        column = next;
    }
    result
}

fn copy_to_clipboard(text: &str) -> Result<(), String> {
    #[cfg(windows)]
    {
        let mut child = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$text = [Console]::In.ReadToEnd(); Set-Clipboard -Value $text",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("failed to start clipboard helper: {error}"))?;
        child
            .stdin
            .take()
            .ok_or_else(|| "clipboard helper stdin was unavailable".to_owned())?
            .write_all(text.as_bytes())
            .map_err(|error| format!("failed to send clipboard text: {error}"))?;
        let output = child
            .wait_with_output()
            .map_err(|error| format!("failed to wait for clipboard helper: {error}"))?;
        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).trim().to_owned())
        }
    }
    #[cfg(not(windows))]
    {
        for program in ["pbcopy", "xclip", "xsel"] {
            let mut command = Command::new(program);
            if program == "xclip" {
                command.args(["-selection", "clipboard"]);
            } else if program == "xsel" {
                command.args(["--clipboard", "--input"]);
            }
            if let Ok(mut child) = command.stdin(Stdio::piped()).stdout(Stdio::null()).spawn() {
                if let Some(mut stdin) = child.stdin.take() {
                    stdin
                        .write_all(text.as_bytes())
                        .map_err(|error| error.to_string())?;
                }
                if child.wait().map_err(|error| error.to_string())?.success() {
                    return Ok(());
                }
            }
        }
        Err("no clipboard helper is available".into())
    }
}

#[cfg(test)]
fn tool_preview(name: &str, arguments: &serde_json::Value) -> String {
    tool_preview_with_language(name, arguments, Language::English)
}

fn tool_preview_with_language(
    name: &str,
    arguments: &serde_json::Value,
    language: Language,
) -> String {
    let Some(arguments) = arguments.as_object() else {
        return language.text("arguments omitted", "未提供参数").into();
    };
    for key in ["command", "path", "url"] {
        if let Some(value) = arguments.get(key).and_then(serde_json::Value::as_str) {
            return redact_sensitive_text(value);
        }
    }
    match name {
        "search" => {
            let query = arguments
                .get("query")
                .and_then(serde_json::Value::as_str)
                .map(redact_sensitive_text);
            let limit = arguments
                .get("max_results")
                .and_then(serde_json::Value::as_u64);
            match (query, limit) {
                (Some(query), Some(limit)) => format!(
                    "{}\"{query}\"{}{}",
                    language.text("query=", "查询="),
                    language.text(", max_results=", "，最多结果数="),
                    limit
                ),
                (Some(query), None) => format!("{}\"{query}\"", language.text("query=", "查询=")),
                _ => language.text("query omitted", "未提供查询").into(),
            }
        }
        "delegate" => delegate_preview_with_language(arguments, language),
        "workflow_checkpoint" => arguments
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .map_or_else(
                || language.text("checkpoint", "检查点").into(),
                |kind| {
                    format!(
                        "{} {}",
                        workflow_checkpoint_kind_label(kind, language),
                        language.text("checkpoint", "检查点")
                    )
                },
            ),
        _ => language.text("arguments omitted", "未提供参数").into(),
    }
}

fn child_state_label(state: &str, language: Language) -> &str {
    if language == Language::Chinese {
        match state {
            "idle" => "空闲",
            "starting" => "启动中",
            "awaitingmodel" => "等待模型",
            "executingtools" => "执行工具",
            "complete" => "已完成",
            "failed" => "失败",
            "cancelled" => "已取消",
            _ => "未知状态",
        }
    } else {
        state
    }
}

fn workflow_checkpoint_kind_label(kind: &str, language: Language) -> String {
    if language == Language::Chinese {
        match kind {
            "plan" => "规划".into(),
            "no_change" => "无需修改".into(),
            "review" => "审查".into(),
            _ => "检查点".into(),
        }
    } else {
        kind.to_owned()
    }
}

fn delegate_preview_with_language(
    arguments: &serde_json::Map<String, serde_json::Value>,
    language: Language,
) -> String {
    let Some(tasks) = arguments.get("tasks").and_then(serde_json::Value::as_array) else {
        return language.text("tasks omitted", "未提供任务").into();
    };
    let label = if language == Language::Chinese {
        format!("{} 个任务", tasks.len())
    } else {
        format!(
            "{} task{}",
            tasks.len(),
            if tasks.len() == 1 { "" } else { "s" }
        )
    };
    let details = tasks
        .iter()
        .enumerate()
        .map(|(index, task)| {
            let role = task
                .get("role")
                .and_then(serde_json::Value::as_str)
                .map(redact_sensitive_text)
                .unwrap_or_else(|| language.text("worker", "工作者").into());
            let objective = task
                .get("objective")
                .and_then(serde_json::Value::as_str)
                .map(redact_sensitive_text)
                .unwrap_or_else(|| language.text("objective omitted", "未提供目标").into());
            if language == Language::Chinese {
                format!("{}：{}——{}", index + 1, role, objective)
            } else {
                format!("{}: {} — {}", index + 1, role, objective)
            }
        })
        .collect::<Vec<_>>()
        .join(if language == Language::Chinese {
            "；"
        } else {
            "; "
        });
    if language == Language::Chinese {
        format!("{label}：{details}")
    } else {
        format!("{label}: {details}")
    }
}

fn string_field(data: &serde_json::Value, field: &str) -> Option<String> {
    data.get(field)
        .and_then(serde_json::Value::as_str)
        .map(redact_sensitive_text)
}

fn role_field(data: &serde_json::Value, language: Language) -> String {
    string_field(data, "role").unwrap_or_else(|| language.text("worker", "工作者").to_owned())
}

fn workflow_stage_label(stage: &str, language: Language) -> String {
    match stage {
        "explore" => language.text("Exploring", "探索中").into(),
        "plan" => language.text("Planning", "规划中").into(),
        "implement" => language.text("Implementing", "实现中").into(),
        "verify" => language.text("Verifying", "验证中").into(),
        "review" => language.text("Reviewing", "审查中").into(),
        _ => format!(
            "{}{}{}",
            language.text("Workflow", "工作流"),
            language.text(": ", "："),
            if language == Language::Chinese {
                "未知阶段"
            } else {
                workflow_requirement_label(stage, language)
            }
        ),
    }
}

fn workflow_requirement_label(value: &str, language: Language) -> &'static str {
    if language == Language::Chinese {
        match value {
            "explore" => "探索",
            "plan" => "规划",
            "implement_or_no_change" => "实现或确认无需修改",
            "verify" => "验证",
            "review" => "审查",
            _ => "未知要求",
        }
    } else {
        match value {
            "explore" => "explore",
            "plan" => "plan",
            "implement_or_no_change" => "implement_or_no_change",
            "verify" => "verify",
            "review" => "review",
            _ => "unknown requirement",
        }
    }
}

fn workflow_evidence_label(value: &str, language: Language) -> &'static str {
    if language == Language::Chinese {
        match value {
            "repository_inspected" => "仓库已检查",
            "plan_recorded" => "计划已记录",
            "mutation_applied" => "修改已应用",
            "no_change_required" => "已确认无需修改",
            "verification_run" => "验证已执行",
            "review_recorded" => "审查已记录",
            _ => "证据",
        }
    } else {
        match value {
            "repository_inspected" => "repository_inspected",
            "plan_recorded" => "plan_recorded",
            "mutation_applied" => "mutation_applied",
            "no_change_required" => "no_change_required",
            "verification_run" => "verification_run",
            "review_recorded" => "review_recorded",
            _ => "recorded",
        }
    }
}

fn uuid_field(data: &serde_json::Value, field: &str) -> Option<Uuid> {
    data.get(field)
        .and_then(serde_json::Value::as_str)
        .and_then(|value| Uuid::parse_str(value).ok())
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

fn localized_operation_label(operation: &str, language: Language) -> &str {
    if language == Language::Chinese {
        match operation {
            "read" => "读取",
            "write" => "写入",
            "execute" => "执行",
            "network" => "联网",
            "extension" => "扩展",
            _ => "扩展",
        }
    } else {
        operation
    }
}

struct TerminalSession {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    cursor_visible: bool,
    preserve_terminal: bool,
}

impl TerminalSession {
    fn enter() -> io::Result<Self> {
        enable_tui_colors();
        enable_ansi_output()?;
        if env::var_os(supervisor::HANDOFF_PATH_ENV).is_some() {
            let stdout = io::stdout();
            let backend = CrosstermBackend::new(stdout);
            let mut terminal = Terminal::new(backend)?;
            terminal.hide_cursor()?;
            return Ok(Self {
                terminal,
                cursor_visible: false,
                preserve_terminal: true,
            });
        }
        terminal::enable_raw_mode()?;
        let mut guard = TerminalInitGuard {
            raw_mode: true,
            alternate_screen: false,
            mouse_capture: false,
            bracketed_paste: false,
            cursor_hidden: false,
        };
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        guard.alternate_screen = true;
        enable_tui_input_protocols(&mut stdout, &mut guard)?;
        execute!(stdout, Hide)?;
        guard.cursor_hidden = true;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.hide_cursor()?;
        guard.disarm();
        Ok(Self {
            terminal,
            cursor_visible: false,
            preserve_terminal: false,
        })
    }

    fn preserve_for_handoff(&mut self) {
        self.preserve_terminal = true;
    }

    fn complete_handoff(&mut self) {
        self.preserve_terminal = false;
    }

    fn set_cursor_visible(&mut self, visible: bool) -> io::Result<()> {
        if visible == self.cursor_visible {
            return Ok(());
        }
        if visible {
            self.terminal.show_cursor()?;
        } else {
            self.terminal.hide_cursor()?;
        }
        self.cursor_visible = visible;
        Ok(())
    }
}

struct TerminalInitGuard {
    raw_mode: bool,
    alternate_screen: bool,
    mouse_capture: bool,
    bracketed_paste: bool,
    cursor_hidden: bool,
}

impl TerminalInitGuard {
    fn disarm(&mut self) {
        self.raw_mode = false;
        self.alternate_screen = false;
        self.mouse_capture = false;
        self.bracketed_paste = false;
        self.cursor_hidden = false;
    }
}

impl Drop for TerminalInitGuard {
    fn drop(&mut self) {
        let mut stdout = io::stdout();
        if self.cursor_hidden {
            let _ = execute!(stdout, Show);
        }
        if self.alternate_screen {
            if self.mouse_capture || self.bracketed_paste {
                let _ = disable_tui_input_protocols(&mut stdout);
            }
            let _ = execute!(stdout, LeaveAlternateScreen);
        }
        if self.raw_mode {
            let _ = terminal::disable_raw_mode();
        }
    }
}

fn enable_tui_input_protocols(
    output: &mut impl Write,
    guard: &mut TerminalInitGuard,
) -> io::Result<()> {
    execute!(output, EnableMouseCapture)?;
    guard.mouse_capture = true;
    execute!(output, EnableBracketedPaste)?;
    guard.bracketed_paste = true;
    Ok(())
}

fn disable_tui_input_protocols(output: &mut impl Write) -> io::Result<()> {
    execute!(output, DisableBracketedPaste, DisableMouseCapture)
}

fn enable_tui_colors() {
    // The full-screen host owns a colored terminal surface; NO_COLOR remains
    // respected by the line-oriented output path.
    Colored::set_ansi_color_disabled(false);
}

fn enable_ansi_output() -> io::Result<()> {
    #[cfg(windows)]
    {
        for handle in [
            crossterm_winapi::Handle::new(crossterm_winapi::HandleType::OutputHandle),
            crossterm_winapi::Handle::new(crossterm_winapi::HandleType::CurrentOutputHandle),
        ]
        .into_iter()
        .flatten()
        {
            enable_console_mode(handle)?;
        }
    }
    Ok(())
}

#[cfg(windows)]
fn enable_console_mode(handle: crossterm_winapi::Handle) -> io::Result<()> {
    let mode = crossterm_winapi::ConsoleMode::from(handle);
    let current = match mode.mode() {
        Ok(current) => current,
        Err(_) => return Ok(()),
    };
    apply_console_mode(current, |requested| mode.set_mode(requested))
}

#[cfg(windows)]
fn requested_console_mode(current: u32) -> u32 {
    const ENABLE_PROCESSED_OUTPUT: u32 = 0x0001;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

    current | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
}

#[cfg(windows)]
fn apply_console_mode(
    current: u32,
    set_mode: impl FnOnce(u32) -> io::Result<()>,
) -> io::Result<()> {
    let requested = requested_console_mode(current);
    if requested != current {
        set_mode(requested)?;
    }
    Ok(())
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        if self.preserve_terminal {
            return;
        }
        let _ = self.terminal.show_cursor();
        let backend = self.terminal.backend_mut();
        let _ = disable_tui_input_protocols(backend);
        let _ = execute!(backend, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ApprovalView, ChatView, Color, Language, Line, RunCapabilities, TerminalInitGuard, Tone,
        ToolState, UiAction,
    };
    use crate::{REVIEW_TASK, SIMPLIFY_TASK};
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEventKind};
    use focus_kernel::{Event, EventKind, Role, ToolCall, ToolResult};
    use focus_runtime::{FocusRuntime, RuntimeConfig};
    use ratatui::{Terminal, backend::TestBackend, layout::Position};
    use serde_json::json;
    use std::{sync::mpsc, thread};
    use uuid::Uuid;

    #[test]
    fn tool_lifecycle_updates_a_single_compact_activity_entry() {
        let mut view = ChatView::default();

        view.tool_started("shell", "cargo test");
        view.tool_finished("shell", false);

        assert_eq!(view.tools.len(), 1);
        assert_eq!(view.tools[0].state, ToolState::Completed);
        assert_eq!(view.tools[0].preview, "cargo test");
    }

    #[test]
    fn tool_preview_keeps_structured_arguments_without_dumping_json() {
        assert_eq!(
            super::tool_preview("search", &json!({"query":"README", "max_results":20})),
            "query=\"README\", max_results=20"
        );
        assert_eq!(
            super::tool_preview(
                "delegate",
                &json!({
                    "tasks":[
                        {"role":"reviewer","objective":"inspect the provider"},
                        {"role":"tester","objective":"run the regression"}
                    ]
                })
            ),
            "2 tasks: 1: reviewer — inspect the provider; 2: tester — run the regression"
        );
    }

    #[test]
    fn tui_drain_ignores_child_session_events() {
        let root = Uuid::new_v4();
        let child = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel();
        sender
            .send(Event::now(
                child,
                EventKind::ToolCallRequested {
                    call: ToolCall {
                        id: "child-call".into(),
                        name: "search".into(),
                        arguments: json!({"query":"child"}),
                    },
                },
            ))
            .unwrap();
        sender
            .send(Event::now(
                root,
                EventKind::ToolCallRequested {
                    call: ToolCall {
                        id: "root-call".into(),
                        name: "search".into(),
                        arguments: json!({"query":"root"}),
                    },
                },
            ))
            .unwrap();
        sender
            .send(Event::now(
                root,
                EventKind::ToolResultReceived {
                    result: ToolResult {
                        tool_call_id: "root-call".into(),
                        name: "search".into(),
                        content: "root result".into(),
                        is_error: false,
                    },
                },
            ))
            .unwrap();

        let mut view = ChatView::default();
        assert!(super::drain_runtime_events(&receiver, &mut view, root).changed);
        assert_eq!(view.tools.len(), 1);
        assert_eq!(view.tools[0].preview, "query=\"root\"");
    }

    #[test]
    fn disconnected_event_stream_is_reported_for_recovery() {
        let root = Uuid::new_v4();
        let (sender, receiver) = mpsc::sync_channel(1);
        drop(sender);
        let outcome = super::drain_runtime_events(&receiver, &mut ChatView::default(), root);

        assert!(!outcome.changed);
        assert!(outcome.disconnected);
    }

    #[test]
    fn replay_restores_user_assistant_and_child_projection() {
        let root = Uuid::new_v4();
        let child = Uuid::new_v4();
        let mut view = ChatView::default();
        let replay = vec![
            Event::now(
                root,
                EventKind::MessageAdded {
                    message: focus_kernel::Message::text(Role::User, "previous request"),
                },
            ),
            Event::now(
                root,
                EventKind::MessageAdded {
                    message: focus_kernel::Message::text(Role::Assistant, "previous answer"),
                },
            ),
            Event::now(
                root,
                EventKind::Runtime {
                    name: "subagent_started".into(),
                    data: json!({"child_session_id": child, "role": "reviewer"}),
                },
            ),
        ];

        view.load_replay(root, &replay);
        view.load_child_replay(
            child,
            &[Event::now(
                child,
                EventKind::Model {
                    event: focus_kernel::ModelEvent::TextDelta {
                        text: "child answer".into(),
                    },
                },
            )],
        );
        let rendered = view
            .lines(120)
            .into_iter()
            .map(|line| line.text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("› previous request"));
        assert!(rendered.contains("previous answer"));
        assert!(rendered.contains("delegate reviewer"));
        assert!(rendered.contains("child answer"));
    }

    #[test]
    fn replay_merges_streamed_assistant_deltas_across_multiple_turns() {
        let root = Uuid::new_v4();
        let event = |kind| Event::now(root, kind);
        let replay = vec![
            event(EventKind::MessageAdded {
                message: focus_kernel::Message::text(Role::User, "first request"),
            }),
            event(EventKind::Model {
                event: focus_kernel::ModelEvent::TextDelta {
                    text: "first answer".into(),
                },
            }),
            event(EventKind::MessageAdded {
                message: focus_kernel::Message::text(Role::Assistant, "first answer"),
            }),
            event(EventKind::MessageAdded {
                message: focus_kernel::Message::text(Role::User, "second request"),
            }),
            event(EventKind::Model {
                event: focus_kernel::ModelEvent::TextDelta {
                    text: "second answer".into(),
                },
            }),
            event(EventKind::MessageAdded {
                message: focus_kernel::Message::text(Role::Assistant, "second answer"),
            }),
        ];
        let mut view = ChatView::default();

        view.load_replay(root, &replay);

        let lines = view.lines(120);
        let rendered = lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            rendered
                .iter()
                .filter(|line| **line == "first answer")
                .count(),
            1
        );
        assert_eq!(
            rendered
                .iter()
                .filter(|line| **line == "second answer")
                .count(),
            1
        );
        assert!(rendered.contains(&"› first request"));
        assert!(rendered.contains(&"› second request"));
    }

    #[test]
    fn live_final_assistant_message_is_visible_without_streaming_deltas() {
        let mut view = ChatView::default();
        view.start_task("answer directly");
        view.apply_event(&Event::now(
            Uuid::new_v4(),
            EventKind::MessageAdded {
                message: focus_kernel::Message::text(Role::Assistant, "final answer"),
            },
        ));

        let rendered = view
            .lines(120)
            .into_iter()
            .map(|line| line.text)
            .collect::<Vec<_>>();
        assert_eq!(
            rendered
                .iter()
                .filter(|line| *line == "final answer")
                .count(),
            1
        );
    }

    #[test]
    fn final_assistant_message_closes_pending_reasoning_before_answer() {
        let mut view = ChatView::default();
        view.append_reasoning("planning");
        view.apply_assistant_message("final answer");

        let lines = view
            .lines(120)
            .into_iter()
            .map(|line| line.text)
            .collect::<Vec<_>>();
        let thinking = lines
            .iter()
            .position(|line| line.contains("Thinking"))
            .expect("reasoning should be rendered");
        let answer = lines
            .iter()
            .position(|line| line == "final answer")
            .expect("assistant answer should be rendered");
        assert!(thinking < answer);
    }

    #[test]
    fn child_final_assistant_message_replaces_streamed_output() {
        let root = Uuid::new_v4();
        let child = Uuid::new_v4();
        let final_message = format!("final {}", "x".repeat(super::MAX_CHILD_TEXT_BYTES * 2));
        let mut view = ChatView::default();
        view.apply_runtime_event(
            "subagent_started",
            &json!({"child_session_id": child, "role": "reviewer"}),
        );
        view.load_child_replay(
            child,
            &[
                Event::now(
                    child,
                    EventKind::Model {
                        event: focus_kernel::ModelEvent::TextDelta {
                            text: "partial".into(),
                        },
                    },
                ),
                Event::now(
                    child,
                    EventKind::MessageAdded {
                        message: focus_kernel::Message::text(Role::Assistant, final_message),
                    },
                ),
            ],
        );

        assert!(view.child_views[&child].output.starts_with("final "));
        assert!(view.child_views[&child].output.len() <= super::MAX_CHILD_TEXT_BYTES);
        assert!(!view.child_views[&child].output.contains("partial"));
        assert!(!view.has_child_session(root));
    }

    #[test]
    fn duplicate_subagent_start_keeps_existing_child_projection_without_redraw() {
        let child = Uuid::new_v4();
        let mut view = ChatView::default();
        let started = json!({"child_session_id": child, "role": "reviewer"});

        assert!(view.apply_runtime_event("subagent_started", &started));
        assert!(view.apply_child_event(&Event::now(
            child,
            EventKind::StateChanged {
                status: focus_kernel::AgentStatus::AwaitingModel,
                turn: 1,
            },
        )));
        assert!(view.apply_child_event(&Event::now(
            child,
            EventKind::Model {
                event: focus_kernel::ModelEvent::TextDelta {
                    text: "partial answer".into(),
                },
            },
        )));

        assert!(!view.apply_runtime_event(
            "subagent_started",
            &json!({"child_session_id": child, "role": "replacement"}),
        ));
        let child_view = &view.child_views[&child];
        assert_eq!(child_view.role, "reviewer");
        assert_eq!(child_view.state, "awaitingmodel");
        assert_eq!(child_view.output, "partial answer");
        assert_eq!(view.child_order, vec![child]);
    }

    #[test]
    fn cancelled_turn_is_notice_not_error() {
        let mut view = ChatView::default();
        view.start_task("interrupt me");
        view.turn_failed("task was cancelled".into());

        assert!(view.transcript.iter().any(
            |item| matches!(item, super::TranscriptItem::Notice(text) if text == "Interrupted")
        ));
        assert!(
            !view
                .transcript
                .iter()
                .any(|item| matches!(item, super::TranscriptItem::Error(_)))
        );
        assert!(!view.should_refresh_status(true));
    }

    #[test]
    fn streamed_cancellation_is_not_rendered_as_a_duplicate_error_or_notice() {
        let mut view = ChatView::default();
        view.start_task("interrupt me");
        view.apply_event(&Event::now(
            Uuid::new_v4(),
            EventKind::Model {
                event: focus_kernel::ModelEvent::Failed {
                    error: "task was cancelled".into(),
                },
            },
        ));
        view.turn_failed("task was cancelled".into());

        assert_eq!(
            view.transcript
                .iter()
                .filter(|item| matches!(item, super::TranscriptItem::Notice(text) if text == "Interrupted"))
                .count(),
            1
        );
        assert!(
            !view
                .transcript
                .iter()
                .any(|item| matches!(item, super::TranscriptItem::Error(_)))
        );
    }

    #[test]
    fn destructive_session_controls_are_blocked_while_active() {
        assert!(super::active_action_message(&UiAction::Clear, Language::English).is_some());
        assert!(super::active_action_message(&UiAction::NewSession, Language::English).is_some());
        assert!(super::active_action_message(&UiAction::Exit, Language::English).is_none());
    }

    #[test]
    fn active_turn_scroll_supports_pages_and_grapheme_deletion() {
        let mut view = ChatView {
            composer: "👍🏻".into(),
            ..ChatView::default()
        };
        assert_eq!(
            view.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE), false),
            UiAction::Redraw
        );
        assert!(view.composer.is_empty());

        assert_eq!(
            view.handle_key(KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE), false),
            UiAction::Redraw
        );
        assert_eq!(view.scroll_from_tail, 10);
        assert_eq!(
            view.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE), false),
            UiAction::Redraw
        );
        assert_eq!(view.scroll_from_tail, usize::MAX);
        assert_eq!(
            view.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE), false),
            UiAction::Redraw
        );
        assert_eq!(view.scroll_from_tail, 0);
    }

    #[test]
    fn mouse_wheel_scrolls_only_without_shift_selection_modifier() {
        let mut view = ChatView::default();
        let up = crossterm::event::MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        assert!(view.handle_mouse(up, false, (80, 24)));
        assert_eq!(view.scroll_from_tail, 3);

        let shift_up = crossterm::event::MouseEvent {
            modifiers: KeyModifiers::SHIFT,
            ..up
        };
        assert!(!view.handle_mouse(shift_up, false, (80, 24)));
        assert_eq!(view.scroll_from_tail, 3);
    }

    #[test]
    fn undersized_terminal_does_not_accept_hidden_mouse_targets() {
        let mut view = ChatView::default();
        view.start_task("hidden target");
        view.append_assistant("answer");
        let mouse = crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 2,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };

        assert!(!view.handle_mouse(mouse, false, (20, 5)));
        assert!(view.mouse_anchor.is_none());
        assert!(view.selection.is_none());
    }

    #[test]
    fn mouse_click_places_cursor_on_any_visible_composer_line() {
        let mut view = ChatView {
            composer: "first line\nsecond line".into(),
            ..ChatView::default()
        };
        let pane = super::bottom_pane(&view, 80, false, "");
        let pane_height = u16::try_from(pane.lines.len() + 1).unwrap();
        let inner_y = 24u16 - pane_height + 1;
        let click = crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 3,
            row: inner_y,
            modifiers: KeyModifiers::NONE,
        };

        assert!(view.handle_mouse(click, false, (80, 24)));
        assert_eq!(view.composer_cursor, 1);
        assert!(view.composer_cursor_set);
    }

    #[test]
    fn mouse_drag_selects_transcript_without_breaking_composer_clicks() {
        let mut view = ChatView::default();
        view.start_task("prompt to copy");
        view.append_assistant("answer");

        let down = crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 2,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let drag = crossterm::event::MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: 14,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let up = crossterm::event::MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: 14,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };

        assert!(view.handle_mouse(down, false, (80, 24)));
        assert!(view.handle_mouse(drag, false, (80, 24)));
        assert!(view.handle_mouse(up, false, (80, 24)));
        assert_eq!(view.selection_text.as_deref(), Some("prompt to co"));

        let pane = super::bottom_pane(&view, 80, false, "");
        let pane_height = u16::try_from(pane.lines.len() + 1).unwrap();
        let inner_y = 24u16 - pane_height + 1;
        let click = crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 3,
            row: inner_y,
            modifiers: KeyModifiers::NONE,
        };
        assert!(view.handle_mouse(click, false, (80, 24)));
        assert!(view.composer_cursor_set);
        assert!(view.selection.is_none());
    }

    #[test]
    fn transcript_selection_uses_scrolled_rows_and_wide_graphemes() {
        let mut view = ChatView::default();
        view.start_task("滚动选择");
        for index in 0..30 {
            view.push_notice(format!("第 {index} 行"));
        }
        view.scroll_from_tail = 3;

        let lines = view.lines(80);
        let selected = view.selected_text(
            super::TextSelection {
                anchor: super::TextPoint { row: 1, column: 2 },
                focus: super::TextPoint { row: 1, column: 6 },
            },
            80,
        );

        assert_eq!(selected.as_deref(), Some("第 0"));
        assert!(lines.len() > 20);
    }

    #[test]
    fn scrolling_then_dragging_selects_the_scrolled_transcript_rows() {
        let mut view = ChatView::default();
        view.start_task("滚动后复制");
        for index in 0..40 {
            view.push_notice(format!("第 {index} 行"));
        }

        let scroll = crossterm::event::MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 4,
            row: 4,
            modifiers: KeyModifiers::NONE,
        };
        assert!(view.handle_mouse(scroll, false, (80, 24)));

        let down = crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 2,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let drag = crossterm::event::MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: 8,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let up = crossterm::event::MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: 8,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };

        assert!(view.handle_mouse(down, false, (80, 24)));
        assert!(view.handle_mouse(drag, false, (80, 24)));
        assert!(view.handle_mouse(up, false, (80, 24)));
        assert_eq!(view.selection_text.as_deref(), Some("第 17 "));
    }

    #[test]
    fn scrolling_after_selection_invalidates_copy_selection() {
        let mut view = ChatView::default();
        view.start_task("滚动后选择");
        view.append_assistant("answer to copy");

        let down = crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 2,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let drag = crossterm::event::MouseEvent {
            kind: MouseEventKind::Drag(MouseButton::Left),
            column: 12,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let up = crossterm::event::MouseEvent {
            kind: MouseEventKind::Up(MouseButton::Left),
            column: 12,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        assert!(view.handle_mouse(down, false, (80, 24)));
        assert!(view.handle_mouse(drag, false, (80, 24)));
        assert!(view.handle_mouse(up, false, (80, 24)));
        assert_eq!(view.selection_text.as_deref(), Some("滚动后选择"));

        let scroll = crossterm::event::MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 4,
            row: 4,
            modifiers: KeyModifiers::NONE,
        };
        assert!(view.handle_mouse(scroll, false, (80, 24)));
        assert!(view.selection.is_none());
        assert!(view.selection_text.is_none());
    }

    #[test]
    fn shift_mouse_events_are_left_for_native_terminal_selection() {
        let mut view = ChatView::default();
        view.start_task("copy from terminal");
        view.append_assistant("answer");
        let mouse = crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 2,
            row: 0,
            modifiers: KeyModifiers::SHIFT,
        };

        assert!(!view.handle_mouse(mouse, false, (80, 24)));
        assert!(view.mouse_anchor.is_none());
        assert!(view.selection.is_none());
    }

    #[test]
    fn clicking_existing_composer_text_places_cursor_and_allows_insertion() {
        let mut view = ChatView {
            composer: "first line\nsecond line".into(),
            ..ChatView::default()
        };
        let pane = super::bottom_pane(&view, 80, false, "");
        let pane_height = u16::try_from(pane.lines.len() + 1).unwrap();
        let inner_y = 24u16 - pane_height + 1;
        let click = crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 4,
            row: inner_y,
            modifiers: KeyModifiers::NONE,
        };

        assert!(view.handle_mouse(click, false, (80, 24)));
        assert_eq!(view.composer_cursor, 2);
        assert_eq!(
            view.handle_key(KeyEvent::new(KeyCode::Char('X'), KeyModifiers::NONE), false),
            UiAction::Redraw
        );
        assert_eq!(view.composer, "fiXrst line\nsecond line");
    }

    #[test]
    fn vertical_composer_navigation_moves_between_visual_rows() {
        let mut view = ChatView {
            composer: "first line\nsecond line".into(),
            composer_cursor: "first line\nsecond line".len(),
            composer_cursor_set: true,
            scroll_from_tail: 7,
            ..ChatView::default()
        };

        assert_eq!(
            view.handle_key_with_width(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE), false, 20),
            UiAction::Redraw
        );
        assert_eq!(view.composer_cursor, "first line".len());
        assert_eq!(view.scroll_from_tail, 7);
        view.handle_key_with_width(
            KeyEvent::new(KeyCode::Char('X'), KeyModifiers::NONE),
            false,
            20,
        );
        assert_eq!(view.composer, "first lineX\nsecond line");
    }

    #[test]
    fn home_and_end_move_the_existing_composer_cursor_within_visual_line() {
        let mut view = ChatView {
            composer: "first line\nsecond line".into(),
            composer_cursor: "first line\nsecond".len(),
            composer_cursor_set: true,
            scroll_from_tail: 7,
            ..ChatView::default()
        };

        assert_eq!(
            view.handle_key_with_width(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE), false, 20),
            UiAction::Redraw
        );
        assert_eq!(view.composer_cursor, "first line\n".len());
        assert_eq!(view.scroll_from_tail, 7);

        assert_eq!(
            view.handle_key_with_width(KeyEvent::new(KeyCode::End, KeyModifiers::NONE), false, 20),
            UiAction::Redraw
        );
        assert_eq!(view.composer_cursor, "first line\nsecond line".len());
        assert_eq!(view.scroll_from_tail, 7);
    }

    #[test]
    fn chinese_localizes_fallback_runtime_labels() {
        let mut view = ChatView::new(super::ApprovalMode::ApproveAll, Language::Chinese);
        view.apply_runtime_event("subagent_failed", &json!({"role": "reviewer"}));
        view.apply_runtime_event("subagent_queued", &json!({}));
        view.apply_runtime_event(
            "workflow_evidence_recorded",
            &json!({"evidence": {"kind": "repository_inspected"}}),
        );
        view.push_status(None, RunCapabilities::default());

        let rendered = view
            .lines(120)
            .into_iter()
            .map(|line| line.text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("未知错误"));
        assert!(rendered.contains("工作者"));
        assert!(rendered.contains("仓库已检查"));
        assert!(rendered.contains("新会话"));
        assert!(!rendered.contains("unknown error"));
        assert!(!rendered.contains("repository_inspected"));
        assert!(!rendered.contains("new"));
    }

    #[test]
    fn chinese_mode_does_not_leak_unknown_protocol_labels() {
        assert_eq!(
            super::child_state_label("waiting_for_approval", Language::Chinese),
            "未知状态"
        );
        assert_eq!(
            super::localized_operation_label("custom_operation", Language::Chinese),
            "扩展"
        );
    }

    #[test]
    fn chinese_runtime_messages_use_natural_separators_and_stage_fallbacks() {
        let mut view = ChatView::new(super::ApprovalMode::ApproveAll, Language::Chinese);
        view.apply_runtime_event("subagent_queued", &json!({}));
        view.apply_runtime_event("workflow_stage_entered", &json!({"stage": "unexpected"}));
        view.start_task("检查输入");
        view.append_reasoning("检查输入");
        view.push_status(
            None,
            RunCapabilities {
                workflow: false,
                delegation: false,
            },
        );

        let rendered = view
            .lines(120)
            .into_iter()
            .map(|line| line.text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("排队委派：工作者"));
        assert!(rendered.contains("工作流：未知阶段"));
        assert!(rendered.contains("会话：新会话｜模式：核心"));

        let pane = super::bottom_pane(&view, 80, true, "terra");
        assert!(
            pane.lines
                .iter()
                .any(|line| line.text.contains("思考：检查输入"))
        );
    }

    #[test]
    fn composer_cursor_mapping_respects_wraps_and_graphemes() {
        let text = "a👍🏻bc";
        assert_eq!(super::composer_cursor_at(text, 0, 0, 4), 0);
        assert_eq!(super::composer_cursor_at(text, 0, 2, 4), "a".len());
        assert_eq!(super::composer_cursor_at(text, 0, 3, 4), "a👍🏻".len());
        assert_eq!(super::composer_cursor_at(text, 1, 0, 4), "a👍🏻b".len());
        assert_eq!(
            super::composer_cursor_at("first\nsecond", 0, 20, 20),
            "first".len()
        );
    }

    #[test]
    fn runtime_delegation_events_are_projected_while_children_are_running() {
        let root = Uuid::new_v4();
        let child = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel();
        sender
            .send(Event::now(
                root,
                EventKind::Runtime {
                    name: "subagent_started".into(),
                    data: json!({
                        "child_session_id": child,
                        "role": "reviewer"
                    }),
                },
            ))
            .unwrap();
        sender
            .send(Event::now(
                child,
                EventKind::Model {
                    event: focus_kernel::ModelEvent::ReasoningSummaryDelta {
                        text: "Inspecting README".into(),
                    },
                },
            ))
            .unwrap();

        let mut view = ChatView::default();
        assert!(super::drain_runtime_events(&receiver, &mut view, root).changed);
        let rendered = view
            .lines(120)
            .into_iter()
            .map(|line| line.text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("delegate reviewer"));
        assert!(rendered.contains("Inspecting README"));
    }

    #[test]
    fn delegation_projection_exposes_selection_tasks_and_child_progress() {
        let root = Uuid::new_v4();
        let child = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel();
        let events = [
            Event::now(
                root,
                EventKind::Model {
                    event: focus_kernel::ModelEvent::ToolCallStarted {
                        id: "delegate-1".into(),
                        name: "delegate".into(),
                    },
                },
            ),
            Event::now(
                root,
                EventKind::ToolCallRequested {
                    call: ToolCall {
                        id: "delegate-1".into(),
                        name: "delegate".into(),
                        arguments: json!({"tasks":[
                            {"role":"reviewer","objective":"inspect README"},
                            {"role":"tester","objective":"run checks"}
                        ]}),
                    },
                },
            ),
            Event::now(
                root,
                EventKind::Runtime {
                    name: "subagent_queued".into(),
                    data: json!({"role":"reviewer"}),
                },
            ),
            Event::now(
                root,
                EventKind::Runtime {
                    name: "subagent_started".into(),
                    data: json!({"child_session_id":child,"role":"reviewer"}),
                },
            ),
            Event::now(
                child,
                EventKind::Model {
                    event: focus_kernel::ModelEvent::ReasoningSummaryDelta {
                        text: "Checking README".into(),
                    },
                },
            ),
            Event::now(
                child,
                EventKind::Model {
                    event: focus_kernel::ModelEvent::ToolCallStarted {
                        id: "search-1".into(),
                        name: "search".into(),
                    },
                },
            ),
            Event::now(
                child,
                EventKind::ToolCallRequested {
                    call: ToolCall {
                        id: "search-1".into(),
                        name: "search".into(),
                        arguments: json!({"query":"README"}),
                    },
                },
            ),
            Event::now(
                child,
                EventKind::Model {
                    event: focus_kernel::ModelEvent::TextDelta {
                        text: "README inspected.".into(),
                    },
                },
            ),
        ];
        for event in events {
            sender.send(event).unwrap();
        }

        let mut view = ChatView::default();
        assert!(super::drain_runtime_events(&receiver, &mut view, root).changed);
        let rendered = view
            .lines(120)
            .into_iter()
            .map(|line| line.text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("• Selecting delegate"));
        assert!(rendered.contains("inspect README"));
        assert!(rendered.contains("• Queueing delegate: reviewer"));
        assert!(rendered.contains("╭─ delegate reviewer"));
        assert!(rendered.contains("reviewer selecting search"));
        assert!(rendered.contains("Checking README"));
        assert!(rendered.contains("README inspected."));
    }

    #[test]
    fn unknown_child_event_is_not_marked_as_applied_before_child_registration() {
        let root = Uuid::new_v4();
        let child = Uuid::new_v4();
        let event = Event::now(
            child,
            EventKind::Model {
                event: focus_kernel::ModelEvent::TextDelta {
                    text: "child output".into(),
                },
            },
        );
        let mut view = ChatView::default();

        assert!(!view.apply_stream_event(&event, root));
        view.defer_child_event(&event);
        view.apply_runtime_event(
            "subagent_started",
            &json!({"child_session_id": child, "role": "reviewer"}),
        );
        assert!(view.child_views[&child].output.contains("child output"));
    }

    #[test]
    fn duplicate_deferred_child_event_is_replayed_once() {
        let root = Uuid::new_v4();
        let child = Uuid::new_v4();
        let event = Event::now(
            child,
            EventKind::Model {
                event: focus_kernel::ModelEvent::TextDelta { text: "x".into() },
            },
        );
        let (sender, receiver) = mpsc::channel();
        sender
            .send(Event::now(
                root,
                EventKind::Runtime {
                    name: "subagent_queued".into(),
                    data: json!({"role": "reviewer"}),
                },
            ))
            .unwrap();
        sender.send(event.clone()).unwrap();
        sender.send(event).unwrap();
        sender
            .send(Event::now(
                root,
                EventKind::Runtime {
                    name: "subagent_started".into(),
                    data: json!({"child_session_id": child, "role": "reviewer"}),
                },
            ))
            .unwrap();
        let mut view = ChatView::default();
        view.start_task("delegate work");
        assert!(super::drain_runtime_events(&receiver, &mut view, root).changed);
        assert_eq!(view.pending_child_event_count, 0);
        assert_eq!(view.child_views[&child].output, "x");
    }

    #[test]
    fn child_event_before_registration_is_replayed_after_root_notice() {
        let root = Uuid::new_v4();
        let child = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel();
        sender
            .send(Event::now(
                root,
                EventKind::Runtime {
                    name: "subagent_queued".into(),
                    data: json!({"role": "reviewer"}),
                },
            ))
            .unwrap();
        sender
            .send(Event::now(
                child,
                EventKind::Model {
                    event: focus_kernel::ModelEvent::TextDelta {
                        text: "early child output".into(),
                    },
                },
            ))
            .unwrap();
        sender
            .send(Event::now(
                root,
                EventKind::Runtime {
                    name: "subagent_started".into(),
                    data: json!({"child_session_id": child, "role": "reviewer"}),
                },
            ))
            .unwrap();

        let mut view = ChatView::default();
        view.start_task("delegate work");
        assert!(super::drain_runtime_events(&receiver, &mut view, root).changed);
        assert_eq!(view.child_views[&child].output, "early child output");
        assert!(view.pending_child_events.is_empty());
    }

    #[test]
    fn unknown_child_event_buffer_has_a_global_bound() {
        let mut view = ChatView::default();
        view.start_task("delegate work");

        for _ in 0..8 {
            let child = Uuid::new_v4();
            for _ in 0..256 {
                view.defer_child_event(&Event::now(
                    child,
                    EventKind::Model {
                        event: focus_kernel::ModelEvent::TextDelta { text: "x".into() },
                    },
                ));
            }
        }

        assert_eq!(view.pending_child_event_count, 1024);
        assert!(view.pending_child_events.len() <= 4);
    }

    #[test]
    fn runtime_ancestry_filters_unrelated_broadcast_child_events() {
        let workspace = std::env::temp_dir().join(format!("focus-tui-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&workspace).unwrap();
        let runtime = FocusRuntime::open(RuntimeConfig::for_workspace(&workspace)).unwrap();
        let root = runtime.create_session("root").unwrap();
        let child = runtime.fork_subagent_session(root.id, "child").unwrap();
        let unrelated_root = runtime.create_session("unrelated").unwrap();
        let unrelated_child = runtime
            .fork_subagent_session(unrelated_root.id, "unrelated-child")
            .unwrap();

        let (sender, receiver) = mpsc::channel();
        for session_id in [child.id, unrelated_child.id] {
            sender
                .send(Event::now(
                    session_id,
                    EventKind::Model {
                        event: focus_kernel::ModelEvent::TextDelta {
                            text: session_id.to_string(),
                        },
                    },
                ))
                .unwrap();
        }

        let mut view = ChatView::default();
        view.start_task("delegate work");
        view.accepting_child_events = true;
        let outcome =
            super::drain_runtime_events_with_runtime(&receiver, &mut view, root.id, Some(&runtime));

        assert!(!outcome.changed);
        assert_eq!(view.pending_child_events.len(), 1);
        assert!(view.pending_child_events.contains_key(&child.id));
        assert!(!view.pending_child_events.contains_key(&unrelated_child.id));
        drop(runtime);
        let _ = std::fs::remove_dir_all(workspace);
    }

    #[test]
    fn applied_event_deduplication_has_a_bounded_window() {
        let mut view = ChatView::default();
        for _ in 0..(super::MAX_APPLIED_EVENT_IDS + 128) {
            assert!(view.remember_event(Uuid::new_v4()));
        }

        assert_eq!(view.applied_event_ids.len(), super::MAX_APPLIED_EVENT_IDS);
        assert_eq!(view.applied_event_order.len(), super::MAX_APPLIED_EVENT_IDS);
    }

    #[test]
    fn runtime_event_drain_is_bounded_per_poll() {
        let root = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel();
        for index in 0..(super::MAX_RUNTIME_EVENTS_PER_DRAIN + 7) {
            sender
                .send(Event::now(
                    root,
                    EventKind::Runtime {
                        name: "workflow_stage_entered".into(),
                        data: json!({"stage": if index % 2 == 0 { "explore" } else { "verify" }}),
                    },
                ))
                .unwrap();
        }

        let mut view = ChatView::default();
        let outcome = super::drain_runtime_events(&receiver, &mut view, root);

        assert!(outcome.changed);
        assert_eq!(receiver.try_iter().count(), 7);
    }

    #[test]
    fn invisible_stream_events_do_not_redraw_or_clear_a_copy_selection() {
        let root = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel();
        sender
            .send(Event::now(
                root,
                EventKind::Model {
                    event: focus_kernel::ModelEvent::RequestStarted {
                        provider: "fixture".into(),
                        model: "fixture".into(),
                        endpoint: "https://example.test".into(),
                    },
                },
            ))
            .unwrap();
        let mut view = ChatView {
            selection: Some(super::TextSelection {
                anchor: super::TextPoint { row: 0, column: 0 },
                focus: super::TextPoint { row: 0, column: 1 },
            }),
            selection_text: Some("selected output".into()),
            ..ChatView::default()
        };

        let outcome = super::drain_runtime_events(&receiver, &mut view, root);

        assert!(!outcome.changed);
        assert_eq!(view.selection_text.as_deref(), Some("selected output"));
    }

    #[test]
    fn duplicate_model_failures_do_not_duplicate_error_projection_or_redraw() {
        let session_id = Uuid::new_v4();
        let failure = |error: &str| {
            Event::now(
                session_id,
                EventKind::Model {
                    event: focus_kernel::ModelEvent::Failed {
                        error: error.into(),
                    },
                },
            )
        };
        let first = failure("provider failed");
        let second = failure("provider failed");
        let mut view = ChatView::default();
        view.start_task("retry me");

        assert!(view.apply_stream_event(&first, session_id));
        view.selection_text = Some("selected output".into());
        view.selection = Some(super::TextSelection {
            anchor: super::TextPoint { row: 0, column: 0 },
            focus: super::TextPoint { row: 0, column: 1 },
        });

        assert!(!view.apply_stream_event(&second, session_id));
        assert_eq!(
            view.transcript
                .iter()
                .filter(|item| matches!(item, super::TranscriptItem::Error(text) if text == "provider failed"))
                .count(),
            1
        );
        assert_eq!(view.selection_text.as_deref(), Some("selected output"));
    }

    #[test]
    fn duplicate_terminal_state_does_not_clear_a_copy_selection_or_redraw() {
        let session_id = Uuid::new_v4();
        let terminal = || {
            Event::now(
                session_id,
                EventKind::StateChanged {
                    status: focus_kernel::AgentStatus::Complete,
                    turn: 1,
                },
            )
        };
        let first = terminal();
        let second = terminal();
        let mut view = ChatView::default();
        view.start_task("finish me");

        assert!(view.apply_stream_event(&first, session_id));
        view.selection_text = Some("selected output".into());
        view.selection = Some(super::TextSelection {
            anchor: super::TextPoint { row: 0, column: 0 },
            focus: super::TextPoint { row: 0, column: 1 },
        });

        assert!(!view.apply_stream_event(&second, session_id));
        assert_eq!(view.selection_text.as_deref(), Some("selected output"));
    }

    #[test]
    fn model_tool_decisions_are_visible_before_tool_execution_completes() {
        let mut view = ChatView::default();
        view.apply_event(&Event::now(
            Uuid::new_v4(),
            EventKind::Model {
                event: focus_kernel::ModelEvent::ToolCallStarted {
                    id: "call-1".into(),
                    name: "delegate".into(),
                },
            },
        ));
        let rendered = view
            .lines(120)
            .into_iter()
            .map(|line| line.text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("• Selecting delegate"));
    }

    #[test]
    fn transport_lifecycle_events_do_not_render_as_transcript_logs() {
        let mut view = ChatView::default();
        let session_id = Uuid::new_v4();

        view.apply_event(&Event::now(
            session_id,
            EventKind::StateChanged {
                status: focus_kernel::AgentStatus::AwaitingModel,
                turn: 0,
            },
        ));
        view.apply_event(&Event::now(
            session_id,
            EventKind::Model {
                event: focus_kernel::ModelEvent::RequestStarted {
                    provider: "fixture".into(),
                    model: "fixture".into(),
                    endpoint: "fixture://provider".into(),
                },
            },
        ));

        assert!(view.lines(120).is_empty());
    }

    #[test]
    fn model_tool_selection_uses_a_structured_activity_cell() {
        let mut view = ChatView::default();
        view.apply_event(&Event::now(
            Uuid::new_v4(),
            EventKind::Model {
                event: focus_kernel::ModelEvent::ToolCallStarted {
                    id: "call-1".into(),
                    name: "delegate".into(),
                },
            },
        ));

        let lines = view.lines(120);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "• Selecting delegate");
    }

    #[test]
    fn workflow_stages_use_human_activity_labels() {
        let mut view = ChatView::default();
        view.apply_runtime_event("workflow_stage_entered", &json!({"stage":"explore"}));

        let lines = view.lines(120);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "• Exploring");
    }

    #[test]
    fn active_turn_shows_elapsed_working_time() {
        let mut view = ChatView::default();

        view.start_task("inspect the workspace");
        let pane = super::bottom_pane(&view, 120, true, "gpt-test");

        assert!(
            pane.lines.iter().any(|line| line.text.contains("Working (")
                && line.text.contains("s • esc to interrupt")),
            "active status must retain an elapsed duration: {:#?}",
            pane.lines
        );
    }

    #[test]
    fn active_status_uses_a_local_animation_tick() {
        let mut view = ChatView::default();
        view.start_task("inspect the workspace");

        let delay = view
            .status_refresh_delay(true, std::time::Instant::now())
            .expect("active status should schedule a frame");
        assert!(delay <= std::time::Duration::from_millis(100));

        let first = super::activity_glyph(view.activity_frame);
        view.advance_activity();
        let second = super::activity_glyph(view.activity_frame);
        assert_ne!(first, second);
    }

    #[test]
    fn finished_turn_removes_elapsed_working_status() {
        let mut view = ChatView::default();

        view.start_task("inspect the workspace");
        view.turn_finished();
        let pane = super::bottom_pane(&view, 120, true, "gpt-test");

        assert!(!pane.lines.iter().any(|line| line.text.contains("Working")));
        assert!(
            view.lines(120)
                .iter()
                .any(|line| line.text.contains("Worked for")),
            "finished turns must preserve a frozen elapsed-duration summary"
        );
        assert!(!view.should_refresh_status(true));
        assert!(
            view.status_refresh_delay(true, std::time::Instant::now())
                .is_none()
        );
    }

    #[test]
    fn worker_join_drains_final_events_before_freezing_duration_summary() {
        let session_id = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            sender
                .send(Event::now(
                    session_id,
                    EventKind::StateChanged {
                        status: focus_kernel::AgentStatus::Complete,
                        turn: 1,
                    },
                ))
                .unwrap();
            sender
                .send(Event::now(
                    session_id,
                    EventKind::Model {
                        event: focus_kernel::ModelEvent::TextDelta {
                            text: "final response".into(),
                        },
                    },
                ))
                .unwrap();
            (Some(session_id), Ok(()))
        });
        while !worker.is_finished() {
            thread::yield_now();
        }

        let mut active = Some(super::ActiveTurn {
            session_id,
            worker: Some(worker),
            cancellation: focus_runtime::subagent::CancellationToken::default(),
            inbox: None,
            pending: None,
        });
        let mut session = Some(session_id);
        let mut view = ChatView::default();
        view.start_task("inspect the workspace");
        assert!(
            super::finish_turn(None, &mut active, &mut session, &mut view, &receiver,).unwrap()
        );
        let lines = view.lines(120);
        let response_index = lines
            .iter()
            .position(|line| line.text == "final response")
            .expect("final response should be rendered");
        let summary_index = lines
            .iter()
            .position(|line| line.text.contains("Worked for"))
            .expect("duration summary should be rendered");
        assert!(response_index < summary_index);
    }

    #[test]
    fn worker_join_drains_event_flood_before_freezing_duration_summary() {
        let session_id = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel();
        for _ in 0..super::MAX_RUNTIME_EVENTS_PER_DRAIN {
            sender
                .send(Event::now(
                    session_id,
                    EventKind::Runtime {
                        name: "workflow_stage_entered".into(),
                        data: json!({"stage": "explore"}),
                    },
                ))
                .unwrap();
        }
        sender
            .send(Event::now(
                session_id,
                EventKind::Model {
                    event: focus_kernel::ModelEvent::TextDelta {
                        text: "final response after flood".into(),
                    },
                },
            ))
            .unwrap();
        let worker = thread::spawn(move || (Some(session_id), Ok(())));
        while !worker.is_finished() {
            thread::yield_now();
        }

        let mut active = Some(super::ActiveTurn {
            session_id,
            worker: Some(worker),
            cancellation: focus_runtime::subagent::CancellationToken::default(),
            inbox: None,
            pending: None,
        });
        let mut session = Some(session_id);
        let mut view = ChatView::default();
        view.start_task("inspect the workspace");

        assert!(super::finish_turn(None, &mut active, &mut session, &mut view, &receiver).unwrap());
        let lines = view.lines(120);
        let response_index = lines
            .iter()
            .position(|line| line.text == "final response after flood")
            .expect("final response should be rendered");
        let summary_index = lines
            .iter()
            .position(|line| line.text.contains("Worked for"))
            .expect("duration summary should be rendered");
        assert!(response_index < summary_index);
    }

    #[test]
    fn runtime_backed_completion_keeps_unrelated_broadcast_events_queued() {
        let workspace = std::env::temp_dir().join(format!("focus-tui-finish-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&workspace).unwrap();
        let runtime = FocusRuntime::open(RuntimeConfig::for_workspace(&workspace)).unwrap();
        let session_id = runtime.create_session("root").unwrap().id;
        let unrelated = Uuid::new_v4();
        let (sender, receiver) = mpsc::channel();
        for _ in 0..(super::MAX_RUNTIME_EVENTS_PER_DRAIN * 2) {
            sender
                .send(Event::now(
                    unrelated,
                    EventKind::Runtime {
                        name: "workflow_stage_entered".into(),
                        data: json!({"stage": "explore"}),
                    },
                ))
                .unwrap();
        }
        let worker = thread::spawn(move || (Some(session_id), Ok(())));
        while !worker.is_finished() {
            thread::yield_now();
        }

        let mut active = Some(super::ActiveTurn {
            session_id,
            worker: Some(worker),
            cancellation: focus_runtime::subagent::CancellationToken::default(),
            inbox: None,
            pending: None,
        });
        let mut session = Some(session_id);
        let mut view = ChatView::default();
        view.start_task("finish without draining other sessions");
        view.scroll_from_tail = 7;

        assert!(
            super::finish_turn(
                Some(&runtime),
                &mut active,
                &mut session,
                &mut view,
                &receiver,
            )
            .unwrap()
        );
        assert!(
            receiver.try_recv().is_ok(),
            "finishing a runtime-backed turn must not consume unrelated broadcast events"
        );
        assert_eq!(
            view.scroll_from_tail, 7,
            "runtime-backed completion must not force a scrolled reader back to the tail"
        );

        drop(sender);
        drop(runtime);
        let _ = std::fs::remove_dir_all(workspace);
    }

    #[test]
    fn runtime_backed_completion_keeps_the_composed_follow_up_prompt() {
        let workspace =
            std::env::temp_dir().join(format!("focus-tui-follow-up-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&workspace).unwrap();
        let runtime = FocusRuntime::open(RuntimeConfig::for_workspace(&workspace)).unwrap();
        let session_id = runtime.create_session("root").unwrap().id;
        let (_sender, receiver) = mpsc::channel();
        let worker = thread::spawn(move || (Some(session_id), Ok(())));
        while !worker.is_finished() {
            thread::yield_now();
        }

        let mut active = Some(super::ActiveTurn {
            session_id,
            worker: Some(worker),
            cancellation: focus_runtime::subagent::CancellationToken::default(),
            inbox: None,
            pending: None,
        });
        let mut session = Some(session_id);
        let mut view = ChatView::default();
        view.start_task("first prompt");
        view.replace_composer("下一条 prompt".into());

        assert!(
            super::finish_turn(
                Some(&runtime),
                &mut active,
                &mut session,
                &mut view,
                &receiver,
            )
            .unwrap()
        );
        assert_eq!(
            view.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), false),
            UiAction::Submit("下一条 prompt".into())
        );

        drop(runtime);
        let _ = std::fs::remove_dir_all(workspace);
    }

    #[test]
    fn terminal_state_event_stops_elapsed_working_status_before_worker_join() {
        let mut view = ChatView::default();
        let session_id = Uuid::new_v4();

        view.start_task("inspect the workspace");
        view.apply_event(&Event::now(
            session_id,
            EventKind::StateChanged {
                status: focus_kernel::AgentStatus::Complete,
                turn: 1,
            },
        ));

        assert!(!view.should_refresh_status(true));
        assert!(
            view.status_refresh_delay(true, std::time::Instant::now())
                .is_none()
        );
    }

    #[test]
    fn clearing_visible_state_stops_elapsed_working_status() {
        let mut view = ChatView::default();

        view.start_task("inspect the workspace");
        view.clear_visible();

        assert!(!view.should_refresh_status(true));
        assert!(
            view.status_refresh_delay(true, std::time::Instant::now())
                .is_none()
        );
    }

    #[test]
    fn clearing_visible_state_resets_transient_ui_and_event_deduplication() {
        let mut view = ChatView {
            pending_approval: Some(ApprovalView::new("shell", "execute", "dir", "inspect")),
            help_visible: true,
            permissions_visible: true,
            error_seen: true,
            ..ChatView::default()
        };
        view.applied_event_ids.insert(Uuid::new_v4());
        view.start_task("inspect the workspace");
        view.clear_visible();

        assert!(view.pending_approval.is_none());
        assert!(!view.help_visible);
        assert!(!view.permissions_visible);
        assert!(!view.error_seen);
        assert!(view.applied_event_ids.is_empty());
    }

    #[test]
    fn transcript_uses_codex_style_user_and_tool_cells() {
        let mut view = ChatView::default();

        view.start_task("inspect the workspace");
        view.append_assistant("I will inspect the project.");
        view.tool_started("shell", "dir");
        view.tool_finished("shell", false);

        let lines = view.lines(120);

        assert_eq!(lines[0].text, "› inspect the workspace");
        assert_eq!(lines[1].text, "I will inspect the project.");
        assert_eq!(lines[2].text, "• Ran shell (dir)");
    }

    #[test]
    fn background_events_do_not_reset_manual_scroll_position() {
        let mut view = ChatView::default();
        for index in 0..30 {
            view.push_notice(format!("event {index}"));
        }
        view.scroll_from_tail = 9;

        view.push_notice("new runtime event");

        assert_eq!(view.scroll_from_tail, 9);
    }

    #[test]
    fn excessive_scroll_stays_at_the_oldest_visible_lines() {
        assert_eq!(super::visible_transcript_range(30, 10, usize::MAX), (0, 10));
        assert_eq!(super::visible_transcript_range(6, 10, 3), (0, 6));
        assert_eq!(super::visible_bottom_range(30, 10), (20, 30));
    }

    #[test]
    fn long_composer_keeps_tail_and_cursor_in_bottom_viewport() {
        let view = ChatView {
            composer: "line\n".repeat(20),
            ..ChatView::default()
        };
        let pane = super::bottom_pane(&view, 36, false, "adapter");
        let (start, end) = super::visible_bottom_range(pane.lines.len(), 8);
        let (_, cursor_row) = pane.cursor.expect("composer cursor");

        assert!(cursor_row >= start);
        assert!(cursor_row < end);
        assert_eq!(end, pane.lines.len());
    }

    #[test]
    fn active_stream_error_preserves_working_clock() {
        let mut view = ChatView::default();
        view.start_task("inspect the workspace");
        view.push_active_error("event stream disconnected");

        assert!(view.should_refresh_status(true));
        assert!(view
            .transcript
            .iter()
            .any(|item| matches!(item, super::TranscriptItem::Error(text) if text == "event stream disconnected")));
    }

    #[test]
    fn starting_a_new_task_drops_previous_delegation_projection() {
        let child = Uuid::new_v4();
        let mut view = ChatView::default();
        view.apply_runtime_event(
            "subagent_started",
            &json!({"child_session_id": child, "role": "reviewer"}),
        );
        assert!(view.has_child_session(child));

        view.start_task("next task");

        assert!(!view.has_child_session(child));
    }

    #[test]
    fn delegation_projection_preserves_child_start_order() {
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let mut view = ChatView::default();
        view.apply_runtime_event(
            "subagent_started",
            &json!({"child_session_id": first, "role": "reviewer"}),
        );
        view.apply_runtime_event(
            "subagent_started",
            &json!({"child_session_id": second, "role": "tester"}),
        );

        let lines = view.lines(120);
        let reviewer = lines
            .iter()
            .position(|line| line.text.contains("delegate reviewer"))
            .unwrap();
        let tester = lines
            .iter()
            .position(|line| line.text.contains("delegate tester"))
            .unwrap();
        assert!(reviewer < tester);
    }

    #[test]
    fn submitting_a_new_task_returns_the_view_to_the_tail() {
        let mut view = ChatView {
            scroll_from_tail: 9,
            ..ChatView::default()
        };

        view.start_task("inspect the workspace");

        assert_eq!(view.scroll_from_tail, 0);
    }

    #[test]
    fn tone_projection_retains_codex_palette() {
        assert_eq!(
            Line::new("user", Tone::User).to_ratatui().spans[0].style.fg,
            Some(Color::Cyan)
        );
        assert_eq!(
            Line::new("activity", Tone::Activity).to_ratatui().spans[0]
                .style
                .fg,
            Some(Color::Yellow)
        );
        assert_eq!(
            Line::new("error", Tone::Error).to_ratatui().spans[0]
                .style
                .fg,
            Some(Color::Red)
        );
    }

    #[test]
    fn markdown_code_uses_low_contrast_terminal_style() {
        let line = super::markdown::render_markdown("inline `code`", 80)
            .into_iter()
            .next()
            .expect("markdown line");
        let code_text = line
            .spans
            .iter()
            .find(|span| span.style.code)
            .map(|span| span.text.clone())
            .expect("code span");
        let rendered = Line::markdown(line).to_ratatui();
        let code_style = rendered
            .spans
            .iter()
            .find(|span| span.content == code_text)
            .expect("rendered code span")
            .style;

        assert_eq!(code_style.fg, Some(Color::Yellow));
        assert_eq!(code_style.bg, None);
    }

    #[test]
    fn markdown_selection_preserves_colors_outside_the_selected_range() {
        let line = Line::markdown(
            super::markdown::render_markdown("**bold** `code`", 80)
                .into_iter()
                .next()
                .expect("markdown line"),
        );
        let rendered = line.to_ratatui_selected(Some((0, 1)));
        let code_style = rendered
            .spans
            .iter()
            .find(|span| span.style.fg == Some(Color::Yellow))
            .expect("code span");

        assert_eq!(code_style.style.fg, Some(Color::Yellow));
        assert_eq!(code_style.style.bg, None);
    }

    #[test]
    fn transcript_selection_is_invalidated_when_live_layout_changes() {
        let root = Uuid::new_v4();
        let mut view = ChatView::default();
        view.start_task("select me");
        view.mouse_anchor = Some(super::TextPoint { row: 0, column: 0 });
        view.update_selection(super::TextPoint { row: 0, column: 4 }, 80);
        assert!(view.selection.is_some());

        view.apply_stream_event(
            &Event::now(
                root,
                EventKind::Runtime {
                    name: "workflow_stage_entered".into(),
                    data: json!({"stage": "verify"}),
                },
            ),
            root,
        );

        assert!(view.selection.is_none());
        assert!(view.selection_text.is_none());
    }

    #[test]
    fn transcript_distinguishes_reasoning_and_tool_lifecycles() {
        let mut view = ChatView::default();
        view.append_reasoning("planning");
        view.finish_reasoning();
        view.tool_started("shell", "cargo test");
        let running = view.lines(120);
        assert!(
            running
                .iter()
                .any(|line| line.text.contains("• Running shell (cargo test)"))
        );
        assert!(running.iter().any(|line| line.text.contains("Thinking")));

        view.tool_finished("shell", false);
        let completed = view.lines(120);
        assert!(
            completed
                .iter()
                .any(|line| line.text.contains("• Ran shell (cargo test)"))
        );
    }

    #[test]
    fn tui_color_setup_overrides_no_color_for_full_screen_host() {
        super::enable_tui_colors();
        assert!(!crossterm::style::Colored::ansi_color_disabled_memoized());
    }

    #[test]
    fn rendered_composer_cell_retains_user_color() {
        let view = ChatView::default();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        let frame = terminal
            .draw(|frame| super::render_frame(frame, &view, "gpt-test", false))
            .unwrap();

        let cell = (0..24)
            .flat_map(|y| (0..80).map(move |x| Position::new(x, y)))
            .find_map(|position| {
                let cell = frame.buffer.cell(position)?;
                (cell.symbol() == "›").then_some(cell)
            })
            .expect("composer cell should be rendered");
        assert_eq!(cell.fg, Color::Cyan);
    }

    #[test]
    fn composer_uses_terminal_width_and_preserves_graphemes() {
        let view = ChatView {
            composer: "中".repeat(18),
            ..ChatView::default()
        };

        let pane = super::bottom_pane(&view, 36, false, "adapter");

        assert_eq!(pane.lines[0].text, format!("› {}", "中".repeat(17)));
        assert_eq!(pane.lines[1].text, "  中");
        assert_eq!(pane.cursor, Some((4, 1)));
        assert_eq!(super::wrap_with_widths("👍🏻", 1, 1), vec!["👍🏻"]);
        assert_eq!(super::truncate("中中a", 4), "...");
    }

    #[test]
    fn cursor_position_keeps_the_first_wide_grapheme_on_the_current_row() {
        assert_eq!(super::cursor_position("中", 1), (0, 2));
    }

    #[test]
    fn approval_view_exposes_once_session_and_deny_actions() {
        let approval = ApprovalView::new("shell", "execute", "cargo test", "verify");

        let rendered = approval.rendered_with_selection(0);
        assert!(rendered.contains("Would you like to run the following command?"));
        assert!(rendered.contains("› 1. Yes, just this once (y)"));
        assert!(rendered.contains("2. Yes, allow for this session (s)"));
        assert!(rendered.contains("3. No, continue without permission (n)"));
    }

    #[test]
    fn slash_commands_keep_session_control_in_the_tui() {
        let mut view = ChatView::default();

        assert_eq!(view.submit("/help"), UiAction::Redraw);
        assert_eq!(view.submit("/new"), UiAction::NewSession);
        assert!(view.help_visible);
    }

    #[test]
    fn every_supported_slash_command_has_a_deterministic_ui_action() {
        let cases = [
            ("/help", UiAction::Redraw),
            ("/copy", UiAction::Copy),
            ("/language", UiAction::OpenLanguage),
            ("/permissions", UiAction::OpenPermissions),
            ("/new", UiAction::NewSession),
            ("/clear", UiAction::Clear),
            ("/details", UiAction::ToggleDetails),
            ("/status", UiAction::Status),
            ("/review", UiAction::Submit(REVIEW_TASK.into())),
            ("/simplify", UiAction::Submit(SIMPLIFY_TASK.into())),
            ("/update", UiAction::Update),
            ("/exit", UiAction::Exit),
            ("/quit", UiAction::Exit),
        ];
        for (command, expected) in cases {
            assert_eq!(ChatView::default().submit(command), expected, "{command}");
        }
    }

    #[test]
    fn maintenance_commands_submit_explicit_self_tasks() {
        let mut view = ChatView::default();

        let UiAction::Submit(review) = view.submit("/review") else {
            panic!("/review should submit a self-review task");
        };
        assert!(review.contains("Review the current Focus repository"));
        assert!(review.contains("implement the smallest justified fixes"));

        let UiAction::Submit(simplify) = view.submit("/simplify") else {
            panic!("/simplify should submit a simplification task");
        };
        assert!(simplify.contains("Simplify the current Focus repository"));
        assert!(simplify.contains("preserve functionality"));
        assert!(simplify.contains("performance"));
    }

    #[test]
    fn help_panel_lists_self_maintenance_commands() {
        let view = ChatView {
            help_visible: true,
            ..ChatView::default()
        };
        let pane = super::bottom_pane(&view, 120, false, "adapter");
        let text = pane
            .lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("/review"));
        assert!(text.contains("/simplify"));
    }

    #[test]
    fn copy_command_only_selects_the_current_assistant_response() {
        let mut view = ChatView::default();
        view.start_task("first");
        view.append_assistant("old answer");
        view.turn_finished();
        view.start_task("second");
        view.append_assistant("current answer");

        let start = view
            .transcript
            .iter()
            .rposition(|item| matches!(item, super::TranscriptItem::User(_)))
            .unwrap();
        let copied = view.transcript[start + 1..]
            .iter()
            .filter_map(|item| match item {
                super::TranscriptItem::Assistant(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(copied, vec!["current answer"]);
        assert_eq!(view.submit("/copy"), UiAction::Copy);
    }

    #[test]
    fn tui_handoff_preserves_composer_view_and_event_cursor() {
        let session_id = Uuid::new_v4();
        let event_id = Uuid::new_v4();
        let view = ChatView {
            composer: "继续编辑中文".into(),
            composer_cursor: "继续".len(),
            composer_cursor_set: true,
            scroll_from_tail: 7,
            details_visible: true,
            language: Language::Chinese,
            last_event_id: Some(event_id),
            ..ChatView::default()
        };

        let state = view.handoff_state(Some(session_id));
        let mut restored = ChatView::default();
        restored.restore_handoff(&state).unwrap();

        assert_eq!(state.session_id, Some(session_id));
        assert_eq!(state.event_cursor, Some(event_id));
        assert_eq!(restored.composer, view.composer);
        assert_eq!(restored.composer_cursor, view.composer_cursor);
        assert_eq!(restored.scroll_from_tail, 7);
        assert!(restored.details_visible);
        assert_eq!(restored.language, Language::Chinese);
    }

    #[test]
    fn update_command_maps_to_a_supervisor_handoff_action() {
        let mut view = ChatView::default();
        assert_eq!(view.submit("/update"), UiAction::Update);
    }

    #[test]
    fn update_is_blocked_while_a_turn_is_active() {
        let message = super::active_action_message(&UiAction::Update, Language::Chinese);
        assert_eq!(message, Some("请先完成或取消当前回合，再更新 Focus。"));
    }

    #[test]
    fn submitting_a_turn_resets_composer_cursor_before_the_next_input() {
        let mut view = ChatView {
            composer: "old prompt".into(),
            composer_cursor: 4,
            composer_cursor_set: true,
            ..ChatView::default()
        };

        assert_eq!(
            view.submit("old prompt"),
            UiAction::Submit("old prompt".into())
        );
        assert_eq!(view.composer, "");
        assert_eq!(view.composer_cursor, 0);
        assert!(!view.composer_cursor_set);

        view.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE), false);
        assert_eq!(view.composer, "n");
    }

    #[test]
    fn pasted_crlf_multiline_text_inserts_at_the_existing_grapheme_cursor() {
        let mut view = ChatView {
            composer: "前后".into(),
            composer_cursor: "前".len(),
            composer_cursor_set: true,
            ..ChatView::default()
        };

        view.insert_composer_text("中\r\n文");

        assert_eq!(view.composer, "前中\n文后");
        assert_eq!(&view.composer[..view.composer_cursor], "前中\n文");
    }

    #[test]
    fn repeated_key_events_continue_editing_the_composer() {
        let repeat = KeyEvent {
            code: KeyCode::Char('中'),
            modifiers: KeyModifiers::NONE,
            kind: crossterm::event::KeyEventKind::Repeat,
            state: crossterm::event::KeyEventState::NONE,
        };

        assert!(super::should_handle_key_event(repeat));
    }

    #[test]
    fn full_composer_rejects_input_without_requesting_a_redraw() {
        let mut view = ChatView::default();
        view.replace_composer("x".repeat(super::MAX_COMPOSER_BYTES));

        assert!(!view.insert_composer_text("中"));
        assert_eq!(view.composer.len(), super::MAX_COMPOSER_BYTES);
    }

    #[test]
    fn cursor_position_treats_crlf_as_one_visual_newline() {
        assert_eq!(super::cursor_position("前\r\n中", 20), (1, 2));
    }

    #[test]
    fn language_command_changes_persistent_selection_state() {
        let mut view = ChatView::default();
        assert_eq!(view.submit("/language"), UiAction::OpenLanguage);
        view.open_language();
        assert!(view.language_visible);
        assert_eq!(
            view.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE), false),
            UiAction::Redraw
        );
        assert_eq!(
            view.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), false),
            UiAction::SetLanguage(Language::Chinese)
        );
    }

    #[test]
    fn language_file_round_trips_without_process_global_environment_state() {
        let directory = std::env::temp_dir().join(format!("focus-tui-language-{}", Uuid::new_v4()));
        let path = directory.join("language");

        Language::Chinese.save_to(&path).unwrap();

        assert_eq!(Language::load_from(&path), Language::Chinese);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn chinese_language_localizes_active_status_and_composer_placeholder() {
        let mut view = ChatView::new(super::ApprovalMode::ApproveAll, Language::Chinese);
        view.start_task("inspect the workspace");

        let pane = super::bottom_pane(&view, 80, true, "terra");
        assert!(pane.lines.iter().any(|line| line.text.contains("工作中")));
        assert!(
            pane.lines
                .iter()
                .any(|line| line.text.contains("告诉 Focus 你要做什么"))
        );
        assert_eq!(
            super::active_action_message(&UiAction::Clear, Language::Chinese),
            Some("请先完成或取消当前回合，再清空界面。")
        );
    }

    #[test]
    fn chinese_localizes_delegation_tool_and_duration_labels() {
        let mut view = ChatView::new(super::ApprovalMode::ApproveAll, Language::Chinese);
        view.apply_runtime_event(
            "subagent_batch_completed",
            &json!({"results": [{}, {}, {}]}),
        );
        view.tool_started("shell", "cargo test");
        view.tool_finished("shell", false);
        view.tools[0].detail = Some(" \n".into());
        view.transcript.push(super::TranscriptItem::TurnSummary(
            std::time::Duration::from_secs(62),
        ));

        let rendered = view
            .lines(120)
            .into_iter()
            .map(|line| line.text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("3 个任务"));
        assert!(rendered.contains("已运行 shell"));
        assert!(rendered.contains("已完成"));
        assert!(rendered.contains("用时 1 分 02 秒"));
        assert!(!rendered.contains("tasks"));
    }

    #[test]
    fn chinese_localizes_approval_labels_and_terminal_size_message() {
        let approval = ApprovalView::new("shell", "execute", "cargo test", "verify");
        let rendered = approval.rendered_with_language(0, Language::Chinese);
        assert!(rendered.contains("目标：cargo test"));
        assert!(!rendered.contains("Target:"));
        assert!(!rendered.contains("execute"));

        let view = ChatView::new(super::ApprovalMode::ApproveAll, Language::Chinese);
        let mut terminal = Terminal::new(TestBackend::new(20, 5)).unwrap();
        terminal
            .draw(|frame| super::render_frame(frame, &view, "terra", false))
            .unwrap();
        let rendered = format!("{}", terminal.backend());
        assert!(rendered.contains("Focus 需要"));
        assert!(!rendered.contains("Focus needs"));
    }

    #[test]
    fn permissions_command_changes_the_mode_for_later_turns() {
        let mut view = ChatView::default();

        assert_eq!(view.submit("/permissions"), UiAction::OpenPermissions);
        view.open_permissions();
        assert!(view.permissions_visible);
        assert_eq!(
            view.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE), false),
            UiAction::Redraw
        );
        assert_eq!(
            view.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), false),
            UiAction::SetApprovalMode(super::ApprovalMode::Interactive)
        );
        view.set_approval_mode(super::ApprovalMode::Interactive);
        assert_eq!(view.approval_mode, super::ApprovalMode::Interactive);
    }

    #[test]
    fn reasoning_summary_is_status_first_then_a_transcript_block() {
        let mut view = ChatView::default();

        view.append_reasoning("**Inspect** the workspace");
        assert_eq!(view.thinking_header().as_deref(), Some("Inspect"));
        view.finish_reasoning();

        assert!(matches!(
            view.transcript.last(),
            Some(super::TranscriptItem::Reasoning(summary)) if summary == "**Inspect** the workspace"
        ));
    }

    #[test]
    fn ratatui_projection_is_stable_for_an_unchanged_streaming_view() {
        let mut view = ChatView::default();
        view.start_task("inspect the workspace");
        view.append_assistant("# Findings\n\n- `Cargo.toml`");
        view.composer = "/per".into();
        view.slash_menu.update(&view.composer);
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();

        let first = terminal
            .draw(|frame| super::render_frame(frame, &view, "gpt-test", true))
            .unwrap()
            .buffer
            .clone();
        let second = terminal
            .draw(|frame| super::render_frame(frame, &view, "gpt-test", true))
            .unwrap()
            .buffer
            .clone();

        assert_eq!(first, second);
        assert!(format!("{}", terminal.backend()).contains("/permissions"));
        assert!(terminal.backend().cursor_visible());
    }

    #[test]
    fn help_panel_lists_both_exit_aliases() {
        let view = ChatView {
            help_visible: true,
            ..ChatView::default()
        };
        let pane = super::bottom_pane(&view, 120, false, "");
        let rendered = pane
            .lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("/exit"));
        assert!(rendered.contains("/quit"));
    }

    #[test]
    fn transcript_projection_has_a_bounded_history() {
        let mut view = ChatView::default();
        for index in 0..(super::MAX_TRANSCRIPT_ITEMS + 128) {
            view.push_notice(format!("event {index}"));
        }

        assert!(view.transcript.len() <= super::MAX_TRANSCRIPT_ITEMS);
        assert!(view.transcript.iter().any(
            |item| matches!(item, super::TranscriptItem::Notice(text) if text == "event 128")
        ));
    }

    #[test]
    fn transcript_projection_enforces_a_total_text_budget() {
        let mut view = ChatView::default();
        for _ in 0..64 {
            view.push_notice("中".repeat(super::MAX_TRANSCRIPT_TEXT_BYTES / 3));
        }

        assert!(view.transcript_text_bytes <= super::MAX_TRANSCRIPT_TOTAL_BYTES);
        assert!(view.transcript.len() < super::MAX_TRANSCRIPT_ITEMS);
    }

    #[test]
    fn streamed_assistant_items_use_the_same_history_bound() {
        let mut view = ChatView::default();
        for index in 0..(super::MAX_TRANSCRIPT_ITEMS + 1) {
            view.assistant_index = None;
            view.append_assistant(&format!("answer {index}"));
        }

        assert!(view.transcript.len() <= super::MAX_TRANSCRIPT_ITEMS);
        assert!(view.transcript.iter().any(|item| matches!(
            item,
            super::TranscriptItem::Assistant(text) if text == "answer 4096"
        )));
    }

    #[test]
    fn child_projection_text_is_bounded_under_streaming_deltas() {
        let child = Uuid::new_v4();
        let mut view = ChatView::default();
        view.apply_runtime_event(
            "subagent_started",
            &json!({"child_session_id": child, "role": "reviewer"}),
        );
        view.apply_child_event(&Event::now(
            child,
            EventKind::Model {
                event: focus_kernel::ModelEvent::TextDelta {
                    text: "x".repeat(super::MAX_TRANSCRIPT_TEXT_BYTES * 2),
                },
            },
        ));

        assert!(view.child_views[&child].output.len() <= 64 * 1024);
    }

    #[test]
    fn composer_paste_is_bounded_without_splitting_utf8() {
        let mut view = ChatView::default();
        let max_bytes = 256 * 1024;
        view.insert_composer_text(&"中".repeat(max_bytes));

        assert!(view.composer.len() <= max_bytes);
        assert!(view.composer.is_char_boundary(view.composer_cursor));
    }

    #[test]
    fn tool_projection_keeps_indices_valid_when_history_is_compacted() {
        let mut view = ChatView::default();
        for index in 0..(super::MAX_TOOL_VIEWS + 1) {
            view.tool_started_with_id(format!("call-{index}"), "shell".into(), "cargo test".into());
        }

        assert_eq!(view.tools.len(), super::MAX_TOOL_VIEWS);
        assert!(view.transcript.iter().all(|item| match item {
            super::TranscriptItem::Tool(index) => *index < view.tools.len(),
            _ => true,
        }));
    }

    #[test]
    fn tool_projection_enforces_a_total_visible_text_budget() {
        let mut view = ChatView::default();
        for index in 0..1_100 {
            view.tool_started_with_id(
                format!("call-{index}"),
                "shell".into(),
                "x".repeat(super::MAX_TOOL_PREVIEW_BYTES),
            );
        }

        let total_text_bytes = view
            .tools
            .iter()
            .map(|tool| {
                tool.name.len() + tool.preview.len() + tool.detail.as_ref().map_or(0, String::len)
            })
            .sum::<usize>();
        assert!(total_text_bytes <= 4 * 1024 * 1024);
    }

    #[test]
    fn bounded_tool_call_ids_keep_long_shared_prefixes_distinct() {
        let mut view = ChatView::default();
        let prefix = "p".repeat(4_096);
        let first = format!("{prefix}first");
        let second = format!("{prefix}second");
        view.tool_started_with_id(first.clone(), "shell".into(), "first".into());
        view.tool_started_with_id(second, "shell".into(), "second".into());

        view.tool_finished_with_id(&first, false, "done");

        assert!(view.tools.iter().all(|tool| {
            tool.call_key
                .as_ref()
                .is_none_or(|key| key.prefix.len() <= 512)
        }));
        assert_eq!(view.tools[0].state, super::ToolState::Completed);
        assert_eq!(view.tools[1].state, super::ToolState::Running);
    }

    #[test]
    fn child_projection_keeps_a_bounded_number_of_sessions() {
        let mut view = ChatView::default();
        for _ in 0..(super::MAX_CHILD_VIEWS + 1) {
            view.apply_runtime_event(
                "subagent_started",
                &json!({"child_session_id": Uuid::new_v4(), "role": "reviewer"}),
            );
        }

        assert_eq!(view.child_views.len(), super::MAX_CHILD_VIEWS);
        assert_eq!(view.child_order.len(), super::MAX_CHILD_VIEWS);
    }

    #[test]
    fn evicting_a_child_invalidates_a_transcript_selection() {
        let mut view = ChatView::default();
        for _ in 0..super::MAX_CHILD_VIEWS {
            view.apply_runtime_event(
                "subagent_started",
                &json!({"child_session_id": Uuid::new_v4(), "role": "reviewer"}),
            );
        }
        view.selection = Some(super::TextSelection {
            anchor: super::TextPoint { row: 0, column: 0 },
            focus: super::TextPoint { row: 0, column: 1 },
        });
        view.selection_text = Some("selected child output".into());

        view.apply_runtime_event(
            "subagent_started",
            &json!({"child_session_id": Uuid::new_v4(), "role": "tester"}),
        );

        assert!(view.selection.is_none());
        assert!(view.selection_text.is_none());
    }

    #[test]
    fn evicting_a_child_releases_its_pending_event_budget() {
        let mut view = ChatView::default();
        let first = Uuid::new_v4();
        view.accepting_child_events = true;
        for _ in 0..256 {
            view.defer_child_event(&Event::now(
                first,
                EventKind::StateChanged {
                    status: focus_kernel::AgentStatus::AwaitingModel,
                    turn: 0,
                },
            ));
        }
        view.child_views.insert(first, super::ChildView::default());
        view.child_order.push(first);
        for _ in 0..super::MAX_CHILD_VIEWS {
            view.apply_runtime_event(
                "subagent_started",
                &json!({"child_session_id": Uuid::new_v4(), "role": "reviewer"}),
            );
        }

        assert!(view.pending_child_event_count < 256);
    }

    #[test]
    fn unknown_child_event_buffer_rejects_oversized_payloads() {
        let mut view = ChatView {
            accepting_child_events: true,
            ..ChatView::default()
        };
        view.defer_child_event(&Event::now(
            Uuid::new_v4(),
            EventKind::Model {
                event: focus_kernel::ModelEvent::TextDelta {
                    text: "x".repeat(64 * 1024),
                },
            },
        ));

        assert_eq!(view.pending_child_event_count, 0);
    }

    #[test]
    fn unknown_child_event_buffer_enforces_a_total_payload_budget() {
        let mut view = ChatView {
            accepting_child_events: true,
            ..ChatView::default()
        };
        for _ in 0..64 {
            view.defer_child_event(&Event::now(
                Uuid::new_v4(),
                EventKind::Model {
                    event: focus_kernel::ModelEvent::TextDelta {
                        text: "x".repeat(32 * 1024),
                    },
                },
            ));
        }

        let retained_bytes = view
            .pending_child_events
            .values()
            .flatten()
            .map(|event| serde_json::to_vec(event).unwrap().len())
            .sum::<usize>();
        assert!(retained_bytes <= 1024 * 1024);
    }

    #[test]
    fn details_command_toggles_collapsed_tool_output() {
        let mut view = ChatView::default();

        assert_eq!(view.submit("/details"), UiAction::ToggleDetails);
    }

    #[test]
    fn repeated_resize_events_do_not_request_repeated_redraws() {
        let mut size = (120, 40);

        assert!(!super::resize_changed(&mut size, (120, 40)));
        assert!(super::resize_changed(&mut size, (100, 40)));
        assert!(!super::resize_changed(&mut size, (100, 40)));
    }

    #[test]
    fn approval_modal_is_rendered_as_a_stateful_view() {
        let mut view = ChatView::default();
        view.tool_started("shell", "cargo test");

        assert!(view.pending_approval.is_none());
        view.pending_approval = Some(ApprovalView::new(
            "tool".repeat(20_000),
            "operation".repeat(20_000),
            "preview".repeat(20_000),
            "rationale".repeat(20_000),
        ));
        let rendered = view
            .pending_approval
            .as_ref()
            .unwrap()
            .rendered_with_selection(0);

        assert!(view.pending_approval.is_some());
        assert!(rendered.len() <= 4 * 16 * 1024 + 512);
    }

    #[test]
    fn terminal_failure_does_not_duplicate_a_streamed_error() {
        let mut view = ChatView::default();
        view.start_task("test");
        view.push_error("stream failed");

        view.turn_failed("stream failed".into());

        assert_eq!(
            view.transcript
                .iter()
                .filter(|item| matches!(item, super::TranscriptItem::Error(_)))
                .count(),
            1
        );
    }

    #[test]
    fn ctrl_shift_c_requests_copy_action() {
        let mut view = ChatView::default();

        assert!(matches!(
            view.handle_key(
                KeyEvent::new(
                    KeyCode::Char('c'),
                    KeyModifiers::CONTROL | KeyModifiers::SHIFT
                ),
                true,
            ),
            UiAction::Copy
        ));
    }

    #[test]
    fn terminal_init_guard_disarm_clears_partial_terminal_state() {
        let mut guard = TerminalInitGuard {
            raw_mode: true,
            alternate_screen: true,
            mouse_capture: true,
            bracketed_paste: true,
            cursor_hidden: true,
        };

        guard.disarm();

        assert!(!guard.raw_mode);
        assert!(!guard.alternate_screen);
        assert!(!guard.mouse_capture);
        assert!(!guard.bracketed_paste);
        assert!(!guard.cursor_hidden);
    }

    #[test]
    fn terminal_input_protocols_enable_and_cleanup_bracketed_paste() {
        let mut enabled = Vec::new();
        let mut guard = TerminalInitGuard {
            raw_mode: false,
            alternate_screen: false,
            mouse_capture: false,
            bracketed_paste: false,
            cursor_hidden: false,
        };

        super::enable_tui_input_protocols(&mut enabled, &mut guard).unwrap();

        assert!(
            String::from_utf8(enabled)
                .unwrap()
                .contains("\u{1b}[?2004h")
        );
        assert!(guard.mouse_capture);
        assert!(guard.bracketed_paste);

        let mut disabled = Vec::new();
        super::disable_tui_input_protocols(&mut disabled).unwrap();

        assert!(
            String::from_utf8(disabled)
                .unwrap()
                .contains("\u{1b}[?2004l")
        );
    }

    #[cfg(windows)]
    #[test]
    fn terminal_setup_enables_ansi_output_before_entering_the_alternate_screen() {
        super::enable_ansi_output().expect("Windows terminal ANSI setup");
    }

    #[cfg(windows)]
    #[test]
    fn terminal_setup_requests_processed_virtual_terminal_output_and_propagates_failures() {
        assert_eq!(super::requested_console_mode(0x0010), 0x0015);

        let error = super::apply_console_mode(0, |_| Err(std::io::Error::other("set mode failed")))
            .expect_err("mode write failure must stop terminal setup");
        assert_eq!(error.to_string(), "set mode failed");
    }
}
