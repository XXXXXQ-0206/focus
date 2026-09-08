# Architecture Decisions

## Product boundary

Focus is a Codex-inspired, evidence-driven Rust coding agent with an auditable
host harness. The coding agent is the product. `focus-harness` is its thin CLI,
diagnostic surface, and acceptance host; `FocusRuntime` and `focus-kernel` are
internal product components rather than competing product identities.

The Focus-owned kernel follows loop semantics derived from the pinned original
MIT Pi source. Codex inspires the product experience and engineering workflow;
Pi supplies the semantic baseline for the minimal agent loop.

## Internal ownership

```text
focus-harness CLI / external host
        |
        v
FocusRuntime
  core: streaming | context | sessions | events | policy | native tools | web fetch
  explicit: workflow | delegation | MCP | command/container adapters
        |
        v
focus-kernel MIT Pi semantic kernel
  normalized messages | provider | tool | events | cancellation
  direct streaming loop | ordered concurrent tools | terminal convergence
```

Within the product, the Runtime is the sole orchestrator of coding-agent
behavior. Host interfaces select configuration, supply an `ApprovalHandler`,
subscribe to events, and consume `RunResult`; they do not assemble a second
context, tool registry, workflow, or session model.

`focus-kernel` is the one Focus-owned agent implementation. Its `AgentLoop`
keeps normalized transcript and provider/tool contracts, consumes streaming
model events, persists the canonical event stream, executes independent tools
through a bounded rolling pool, commits results in model-call order, and
converges cancellation, provider failure, turn limits, and deferred workflow
completion to one terminal state.
The behavioral source is the pinned MIT Pi archive rather than
`pi_agent_rust` or another Rust rewrite.

## Runtime lifecycle

1. An interface opens one `FocusRuntime` with a workspace root, state root,
   policy, command backend, and optional MCP server configurations.
2. Runtime canonicalizes the workspace, validates the optional outbound
   network policy, opens the JSONL event/session/memory stores plus thin
   session-linked goal metadata, registers built-in tools, and connects every
   configured MCP server.
3. Each MCP connection performs `initialize`, sends
   `notifications/initialized`, follows paginated `tools/list`, canonicalizes
   discovered names, and registers them in the same `ToolRegistry` as built-ins.
4. A run creates or resumes a session, compacts its transcript, builds bounded
   context, and appends the new task. An engineering run also restores persisted
   workflow state.
5. Runtime adds `workflow_checkpoint` only for engineering runs and `delegate`
   only when delegation is enabled and configured depth permits. It then binds
   every enabled tool through one `PolicyEngine` and approval handler.
6. The MIT-semantic `AgentLoop` schedules provider turns and tool calls.
   Provider adapters and tools see normalized values; the kernel emits
   `ToolCallRequested` immediately before handler execution and commits
   `ToolResultReceived` in the original model-call order once the contiguous
   result prefix is complete. Parallel calls use the loop's bounded pool;
   static exclusive calls first drain it and block later submission.
7. An engineering run derives workflow evidence from the canonical transcript
   and events. Its tool-free response completes only after Explore, Plan,
   Implement or no-change, successful Verify, and Review evidence satisfy the
   gate. A core run completes directly without this optional gate.
8. Runtime persists the final phase and returns an interface-neutral
   `RunResult`. Replay and fork read the same event/session model.

## CLI and Orca host boundary

`focus` is the primary configuration and terminal host, while `focus-harness`
remains a compatibility alias; neither is an alternate agent runtime.
`run --session` resumes an existing Runtime session. `chat` holds one
stdin-driven terminal conversation open and routes each non-empty line through
the same Runtime session after the first turn. It renders live model deltas on
stderr and emits turn summaries on stdout. `host --stdio` is the structured
external-host boundary: it accepts a single `focus-host-v1` `attach` or
`resume` JSON object, emits only versioned JSONL event/reset/error frames, and
uses `event.id` as the durable cursor.

The TUI slash commands `/review` and `/simplify` are deterministic adapters to
explicit maintenance tasks. They submit through the same Runtime turn path as
ordinary prompts, so review/simplification work shares session persistence,
approval policy, workflow evidence, tool ownership, and cancellation. The
commands do not create a second reviewer runtime or a separate transcript.

Goal records contain only an operator objective, lifecycle phase, and root and
active Runtime session IDs. They do not store a transcript, event stream,
provider settings, or separate execution state. Project and session memory use
the existing Runtime memory store. The `subagent list` CLI view derives its
rows from canonical `subagent_*` events in the parent session JSONL.

Orca is an optional terminal/worktree host. Its compatible Focus registration
detects `focus` (falling back to `focus-harness` for older installs), launches
the equivalent `chat` command, and uses
stdin-after-start prompt injection. An upgraded adapter consumes `host --stdio`
for replayable event rendering and retains the PTY route only for older Focus
binaries. Neither route translates Focus sessions into Pi, Codex, or Orca
shadow-session formats, or creates provider, event, workflow, memory, or
delegation services. Provider endpoint/model/key remain environment
configuration of the Focus process.

### Stable supervisor and TUI handoff

The `focus` wrapper is a process supervisor; `focus-harness` remains a direct
compatibility entrypoint. The supervisor owns one loopback `TcpListener`, a
per-process UUID token, and the restart loop. It launches an app-child with
`FOCUS_APP_CHILD`, `FOCUS_SUPERVISOR_ADDR`, `FOCUS_SUPERVISOR_TOKEN`, and the
selected `FOCUS_UPDATE_ROOT`. The app-child still owns the existing
`FocusRuntime`, `EventHub`, `ApprovalBroker`, session store, transcript
projection, and telemetry path.

Versioned artifacts live under `versions/<version>/` and are never written
over the checkout or the currently running image. `staged.json` identifies
the last hash-verified build; `manifest.json` atomically selects `active` and
retains one `previous` artifact. Startup verifies the active bytes before
launch. A handoff command is accepted only with the supervisor token, an
artifact matching the active manifest, and a bounded JSON handoff record.
The old TUI marks its terminal as preserved and exits with code 75. The new
child attaches to the existing alternate screen, restores the Runtime session
by replay, and sends a TUI-ready control frame. If that frame never arrives,
the supervisor rolls back and re-launches the previous artifact with the same
handoff record.

The handoff record contains only projection state: Runtime `session_id`, the
last observed `event_cursor`, composer text and UTF-8 cursor, scroll offset,
details visibility, and language code. It is capped at 1 MiB (composer at
256 KiB) and is written by temporary-file replacement. No second agent loop,
session, transcript, event stream, or telemetry store is created.

## Provider boundary

Both provider implementations satisfy the same async-only `ModelProvider`
stream contract:

- `OpenAiCompatibleProvider` posts Chat Completions compatible messages and
  function-tool definitions to `{base_url}/chat/completions`. It parses text
  and tool calls, bounds request time and response bytes, supports bearer and
  extra headers, redacts the configured bearer value from surfaced errors, and
  cancels both request send and response body reads.
- `CommandModelProvider` launches an executable directly, writes one
  `ModelRequest` JSON document to stdin, and parses one `ModelResponse` JSON
  document from stdout. Cancellation kills and waits for the adapter process
  while joining its output readers.

Both provider implementations feed the same normalized `ModelProvider` stream;
the kernel consumes it directly, so transport choices never fork the tool
lifecycle. Blocking command adapter work stays behind a private
`spawn_blocking` worker.

## Tool, MCP, and policy boundary

`RuntimeToolSpec` is the canonical description of an enabled built-in, MCP, or
delegation tool. Kernel `Tool` and Runtime `ToolHandler` each expose one async
execution method. Binding a registry produces Runtime tools that execute in
this order:

```text
Pi tool call
  -> PiToolAdapter
  -> RuntimeTool
  -> PolicyEngine
  -> ApprovalHandler when required
  -> built-in / MCP / delegate handler
  -> normalized tool result
  -> Pi agent transcript and Runtime events
```

MCP is a real persistent newline-delimited JSON-RPC stdio transport. It applies
bounded responses, request timeouts, stderr capture, protocol validation,
pagination/cursor limits, cancellation, and child-process cleanup. Remote tool
names are deterministic OpenAI-safe Runtime names, normally `server__tool`,
with normalization and a stable hash when required.

The default policy allows reads, requires approval for writes, execution, and
extension operations, and denies network operations. An explicit Runtime
`NetworkConfig` registers `web_fetch` and changes its `Network` operation to
approval-gated. Its HTTP(S) transport uses bounded responses and redirects,
checks exact/wildcard domain allow/deny rules on every redirect, blocks
local/private resolution unless configured, permits only default web ports
unless explicitly listed, and pins the validated addresses for each request
hop. `web_fetch` arguments and returned URLs are redacted before durable
transcript/event persistence and later model context, while execution retains
the original input. `ApproveSession` is cached per tool and operation inside
the one policy engine for that run. The model provider transport is separate
from the model-visible web tool.

## Workflow contract

`CodingWorkflow` is an explicit executable capability, enabled by the default
user-facing CLI profile through `RunOptions::engineering`. Embedded hosts and
benchmark modes may still select a narrower `RunOptions` profile. The
production profile registers `workflow_checkpoint` and recognizes
evidence only from canonical transcript entries:

| Stage | Required evidence |
| --- | --- |
| Explore | `read_file` or `search` inspected the repository. |
| Plan | `workflow_checkpoint` recorded a plan. |
| Implement | A canonical mutation occurred, or `workflow_checkpoint` recorded a justified no-change result. |
| Verify | A canonical `shell` call marked for verification completed successfully. |
| Review | `workflow_checkpoint` recorded a post-change review. |

State transitions and gate results are emitted as Runtime events. Interrupted
runs recover the latest workflow state. Missing evidence causes a bounded retry;
exhausting the configured retries fails the run instead of accepting an
unsupported completion claim.

## Sandbox boundary

`WorkspaceSandbox` owns path validation, bounded reads/searches, symlink-aware
workspace containment, atomic file replacement, and the replaceable command
executor used by the canonical shell tool.

Native, Docker, and Podman executors share `CommandRequest` and
`CommandOutput`: wall-clock timeout, per-stream retained byte limits, explicit
termination reasons, optional resource limits, and cooperative cancellation.
The native executor uses `command-group` to create a Unix process group or
Windows Job Object before the command starts. Deadline and cancellation checks
continue until the direct process and inherited output pipes are complete;
timeout or cancellation kills and waits for the whole group. Container
backends also force-remove the named container. Docker/Podman execution mounts
the workspace at `/workspace`, disables networking, and applies requested
memory, CPU, and PID flags.

The native backend is a bounded workspace/process adapter, not an OS security
boundary. Its approved `shell` commands retain host-network capability, so the
`web_fetch` policy governs only that Runtime tool. Container execution provides
the Runtime's stronger built-in isolation option, including `--network none`
for shell commands, while preserving the same policy, tool, event, and
workflow path.

Workspace containment is designed for a local, trusted project directory. It
rejects symlink traversal during normal reads and writes, but portable
path-based checks are not a race-free defense against a hostile concurrent
filesystem actor replacing path components between validation and use. Use a
container backend and a trusted mounted workspace when that threat model
applies; handle-relative filesystem operations are required for a stronger
host-level guarantee.

## Subagent boundary

Subagents are first-class Runtime primitives. The default user-facing profile
enables model-visible delegation with `RunOptions::with_delegation`; embedded
hosts and benchmark modes may keep it disabled:

- `SubagentTask`, `SubagentContext`, and `SubagentResult` are typed contracts
  shared by the delegate tool and Runtime runner.
- `SubagentManager` owns ordered bounded-parallel batches through one async
  `SubagentRunner` contract and the shared cooperative cancellation token.
- `RuntimeSubagentRunner` creates ancestry-linked child sessions, supplies
  bounded inherited context and focus paths, uses the shared provider and
  approval path, and emits queued/started/completed/failed/cancelled events.
- The canonical `delegate` tool uses the same manager and runner, enforces task,
  parallelism, and depth limits, and returns ordered child evidence to the
  parent transcript.

Child sessions do not copy the parent transcript. They retain ancestry and
receive only the bounded context built for their objective.

## Ablation benchmark boundary

`focus-runtime::benchmark` is an acceptance harness over the existing kernel
and Runtime APIs, not a second agent implementation or telemetry subsystem.
Its direct `pi-kernel` mode uses the MIT-semantic `AgentLoop`; its Focus modes
use the same `FocusRuntime` entry points and explicit `RunOptions` capabilities
as normal hosts. The `pi-kernel` and `focus-core` modes register the same
delayed extension fixture through the Runtime's canonical tool registry, while
workflow and delegation are distinct capability workloads. Provider, tool,
child, peak-concurrency, and critical-path metrics are pure derivations from
canonical EventHub events; active-run microseconds and end-to-end duration
separately include the execution boundary and setup. The calibrated fixture
uses cooperative deadline waiting so Windows timer granularity does not
dominate the comparison.
Live/replay integrity compares coalesced semantic sequences because live deltas
are finer grained than durable JSONL. Child-session usage is reported separately
and in the total. Real provider/SSE latency stays in separate live acceptance
receipts.

## Acceptance and release boundary

Source inspection, dependency resolution, tests, and runtime probes establish
technical acceptance only for the exercised behavior. They do not themselves
authorize a production distribution or deployment. Release authorization is a
separate decision that must include dependency-license review, artifact
contents, target environment, and operational approval. The incorporated MIT
Pi source attribution and pinned archive evidence are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Release compliance gate

Dependency compliance is a separate executable boundary, not a build-time
side effect. `focus-release-compliance` parses every registry package in
`Cargo.lock`, reads the exact local registry `Cargo.toml` and declared license
file, records direct/transitive relationships and checksums, and emits the
canonical [THIRD_PARTY_LICENSES.json](THIRD_PARTY_LICENSES.json) inventory.

Technical builds use `--mode technical`: metadata gaps and license riders are
reported as `review_required=true` while the command exits successfully.
Production packaging must use `--mode production`; any missing metadata or
detected rider exits with status 2, so a technical build cannot be mistaken
for a releasable distribution. The gate also rejects any reintroduced
`pi_agent_rust` package before producing a result.
Legacy slash license expressions and registry-cache gaps are review findings,
not silently accepted SPDX metadata.



