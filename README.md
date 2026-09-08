<div align="center">

# Focus

**A Codex-inspired, evidence-driven Rust coding agent with an auditable host harness.**

[English](README.md) | [中文](README.zh-CN.md)

![CI](https://img.shields.io/github/actions/workflow/status/XXXXXQ-0206/focus/ci.yml?branch=dev&label=CI)
![Rust](https://img.shields.io/badge/rust-1.97.1-blue)
![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)

</div>

Focus is a coding agent built in Rust. It combines a small, owned agent kernel
with a runtime that provides sessions, memory, policy, tools, provider
adapters, and an auditable host. It is designed to be **evidence-driven**: the
product's claims are tied to durable session events, replayable transcripts,
deterministic ablation receipts, and explicit release gates.

The primary command is `focus`, with `focus-harness` retained as a
compatibility alias for running, inspecting, and verifying the product.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Build Instructions](#build-instructions)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [Disclaimer](#disclaimer)

---

## Project Overview

The coding agent is the product; the host harness is its thin CLI, diagnostic
surface, and acceptance host. `FocusRuntime` and `focus-kernel` are internal
product components rather than competing product identities.

The Focus-owned kernel follows loop semantics derived from the pinned original
MIT Pi source. Codex inspires the product experience and engineering workflow;
Pi supplies the semantic baseline for the minimal agent loop.

Focus is designed around a single asynchronous execution spine. Providers
stream normalized events, tools and Runtime handlers expose one async call
contract, and blocking MCP stdio stays inside a private worker. Synchronous
APIs exist only at explicit host boundaries and reject nested Tokio
invocation.

## Features

- **Interactive TUI** — a Codex-style full-screen interface with a compact
  transcript, incremental model text, collapsed tool activity, a bottom
  composer, approval dialogs, and slash commands (`/permissions`, `/help`,
  `/new`, `/clear`, `/status`, `/details`, `/review`, `/simplify`,
  `/update`, `/exit`).
- **Bounded ordered tools** — at most four independent tool calls run in
  parallel by default, results commit in model-call order, and writes, shell
  commands, workflow checkpoints, delegation, and MCP calls are explicit
  exclusive barriers.
- **One policy/tool/loop path** — enabled built-ins, MCP tools, and delegation
  are wrapped by the same `PolicyEngine` and scheduled by the same
  MIT-semantic `AgentLoop`.
- **Real MCP transport** — persistent stdio JSON-RPC connections perform
  `initialize`, `notifications/initialized`, paginated `tools/list`, and
  cancellable `tools/call`. Discovered tools enter the canonical registry.
- **Two provider adapters** — an OpenAI-compatible HTTP endpoint (Chat
  Completions or Responses wire contract) or a command adapter that exchanges
  one normalized `ModelRequest`/`ModelResponse` JSON document over stdio.
- **First-class subagents** — typed tasks, results, limits, one async manager,
  isolated child sessions, cooperative cancellation, and ordered bounded-
  parallel aggregation, exposed to the model via the canonical `delegate` tool.
- **Bounded command execution** — native, Docker, and Podman backends share
  timeouts, bounded stdout/stderr capture, cooperative cancellation, and
  process/container cleanup. Container backends disable networking and accept
  memory, CPU, and PID limits.
- **Network policy** — networking is allowlisted and disabled by default;
  exact hosts, wildcards, and global allow/deny rules are supported, and URL
  query strings, fragments, and embedded credentials are redacted before they
  enter transcript or model context.
- **Codex-style permissions** — YOLO mode is the default; use `--interactive`
  or `/permissions` for per-operation approval, or `--deny` for read-only
  mode.
- **Seamless self-update** — `focus self-review`, `self-update stage`, and
  `self-update apply` let you review and stage a version without replacing
  the running executable, with an atomic manifest and a `previous` rollback
  slot.
- **Evidence & ablation** — durable session events, replayable transcripts,
  deterministic ablation receipts, and a reproducible release compliance gate.

## Screenshots

Focus is a terminal application. Its primary surface is an interactive TUI
rendered with a cell-diff terminal backend so model streaming updates only
changed cells rather than repainting the whole screen. A typical layout looks
like:

```text
┌ Focus ─────────────────────────────────────────────────────────────┐
│ Session: worker-01    Mode: yolo    ● connected                    │
├────────────────────────────────────────────────────────────────────┤
│ You: inspect this repository and fix the failing test.             │
│ Focus:                                                            │
│   I inspected the workspace and ran the failing test. The issue is  │
│   a stale assertion in ...                                          │
│   ▸ tool: read crates/focus-kernel/src/lib.rs (0.6s)                │
│   ▸ tool: shell cargo test --workspace (4.1s) [completed]           │
├────────────────────────────────────────────────────────────────────┤
│ Thinking ▸ model streaming ▸ ▸ ▸                                   │
├────────────────────────────────────────────────────────────────────┤
│ > review the change_                                           │
└────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Rust toolchain matching [`rust-toolchain.toml`](rust-toolchain.toml)
  (Rust 1.97.1).
- `cargo`, `rustfmt`, and `clippy`.

### Install the CLI

Install the CLI once from a checkout:

```bash
cargo install --path crates/focus-cli --locked
```

This installs both `focus` and the compatibility alias `focus-harness` into
Cargo's user binary directory. After that, run `focus` from any project
directory; the current directory becomes the workspace and Focus opens its
interactive TUI.

### Configure a provider

Configure the provider through the process environment rather than putting
credentials in a script:

```bash
export OPENAI_API_KEY="TOKEN"
export FOCUS_BASE_URL="https://ENDPOINT/v1"
export FOCUS_MODEL="MODEL"
focus
```

On Windows, use `$env:OPENAI_API_KEY = "TOKEN"` instead of `export`.

## Usage

### Interactive session

```bash
focus                     # open the interactive TUI on the current directory
focus chat                # explicit form of the interactive session
```

In the TUI:

- `/review` — inspect the repository, repair concrete issues, and verify.
- `/simplify` — make measured, low-risk reductions while preserving behavior.
- `/permissions` — switch approval mode.
- `/details` — expand collapsed tool activity.
- `/update` — hand an idle session to a verified self-update artifact.
- `Esc` cancels an active turn.

### One-shot run

```bash
focus run --workspace . --provider openai --model MODEL -- "implement the
requested change and run focused checks"
```

### Line-oriented / CI host

Piped stdin/stdout retain the line-oriented host automatically. Use `--plain`
to force that mode in a terminal:

```bash
printf 'inspect this repository\n' | focus --plain --workspace .
```

### Structured host boundary

GUI and IDE hosts can consume a versioned JSONL event stream instead of parsing
PTY text:

```bash
echo '{"protocol":"focus-host-v1","operation":"attach","session_id":"SESSION_ID"}' \
  | focus host --stdio --workspace .
```

### Sessions, goals, and memory

```bash
focus session list --workspace .
focus session replay --workspace . SESSION_ID
focus goal create --workspace . --title "Release" -- "Produce and verify the release artifact"
focus memory add --workspace . --scope project -- "The release uses the native backend"
```

## Build Instructions

```bash
# Formatting
cargo fmt --all -- --check

# Static analysis (deny warnings)
cargo clippy --workspace --all-targets --locked --offline -- -D warnings

# Tests
cargo test --workspace --locked --offline

# Release build
cargo build --workspace --release --locked

# Release compliance gate
cargo run --locked --offline -p focus-release-compliance -- --mode production --workspace .
```

The CI workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs
these checks on both Linux and Windows.

## Project Structure

| Path | Responsibility |
| --- | --- |
| [`crates/focus-kernel`](crates/focus-kernel) | Focus-owned MIT-Pi-semantic kernel for normalized provider/tool/event scheduling, cancellation, ordered concurrent tools, and JSONL events. |
| [`crates/focus-runtime`](crates/focus-runtime) | Context, session ancestry, memory, policy, built-in tools, provider adapters, MCP, workflow gates, sandbox backends, and subagent lifecycle. |
| [`crates/focus-cli`](crates/focus-cli) | Auditable host CLI that selects adapters, hosts interactive approvals, and exposes diagnostics and acceptance commands without duplicating agent behavior. |
| [`crates/focus-release-compliance`](crates/focus-release-compliance) | Reproducible dependency/license compliance gate used before distribution. |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for ownership and execution flow.

## Roadmap

The project is under active development. Areas we intend to grow include:

- Broader provider coverage and toolkit adapters.
- Stronger sandbox isolation and more backend targets.
- Performance and context-composition tuning at scale.
- Additional acceptance evidence and benchmarking harnesses.
- Extended MCP and workflow integrations.

## Contributing

Contributions are welcome. Please read
[`CONTRIBUTING.md`](CONTRIBUTING.md) first, and follow the
[Code of Conduct](CODE_OF_CONDUCT.md). In short:

1. Fork the repository.
2. Create a feature branch.
3. Make and commit changes with a conventional commit message.
4. Open a pull request into `main`.

## License

Focus is dual-licensed under either:

- Apache License, Version 2.0 ([`LICENSE`](LICENSE)); or
- MIT License ([`LICENSE-MIT`](LICENSE-MIT)).

You may use, modify, and distribute this software under the terms of either
license. See the individual license files for the full terms. Third-party
dependency licensing and attribution are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## FAQ

**What is Focus?** Focus is a coding agent, not a model. It wraps a model
provider and exposes tools, sessions, memory, and an auditable event stream.

**Do I need a provider API key?** Yes. Focus reads credentials from the
environment and does not store them in the repository, transcripts, or
replay artifacts.

**Which model providers are supported?** Any OpenAI-compatible HTTP endpoint
(Chat Completions or Responses wire contract) or a command adapter that speaks
the normalized `ModelRequest`/`ModelResponse` JSON protocol.

**Can Focus run without network access?** Yes. `--network-domain` is
allowlisted and networking is disabled by default; enable `web_fetch`
explicitly when you need outbound web access, and use a no-network container
backend for shell-level isolation.

**Is the native sandbox a security boundary?** No. It enforces workspace paths
and process controls but is not an operating-system isolation boundary. Use a
container backend when process-level network isolation is required.

**How do I report a bug or vulnerability?** See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for bugs and
[`SECURITY.md`](SECURITY.md) for vulnerabilities.

## Acknowledgements

Focus is built on the shoulders of several open-source projects:

- **OpenAI Codex** — the product experience, architecture, and capability
  reference used to inspire Focus's design.
- **Pi (MIT)** — the pinned minimal agent-loop semantic source that the
  `focus-kernel` loop semantics are derived from.
- **PIDEX** — existing coding-workflow implementation material used as a
  reference.
- **DeepSeek Harness** — plugin-composition, session-stream, host/UI, and
  sandbox design reference.
- The **Rust ecosystem** and the crates listed in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

These upstream sources are referenced for architecture study and provenance-
reviewed porting. They are not bundled as runtime dependencies unless
explicitly declared in `Cargo.toml`.

## Disclaimer

> **TERMS OF USE AND DISCLAIMER**
>
> THIS SOFTWARE IS PROVIDED **"AS IS"**, WITHOUT WARRANTY OF ANY KIND, EXPRESS
> OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND NON-INFRINGEMENT. IN NO EVENT
> SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR
> OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE,
> ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
> OTHER DEALINGS IN THE SOFTWARE.
>
> This disclaimer also applies to the DISCLAIMER: this project may
> automate actions that have real-world effects, including shell command
> execution, file modification, network requests, and delegation to other
> software. **You are solely responsible** for the configuration you provide
> and the actions you allow this software to take, and for ensuring that your
> use complies with all applicable laws, regulations, and terms of service.
>
> **The authors and contributors accept no responsibility for any direct,
> indirect, incidental, special, or consequential damages, loss of data,
> loss of profit, or loss of business arising from the use of, or reliance
> on, this software, even if advised of the possibility of such damages.**
>
> **No unlawful use.** Do not use this software to engage in any activity
> that is unlawful, harmful, fraudulent, or unauthorized, including without
> limitation: unauthorized access to systems or networks, distribution of
> malicious software, credential theft, spam, or any activity that violates
> applicable laws or the terms of service of third-party services. You are
> responsible for obtaining any permissions or authorizations required for
> your use of this software and for the consequences of failing to do so.
>
> By installing, building, or running this software you acknowledge that you
> have read this disclaimer and accept its terms at your own risk. If you do
> not agree, do not use the software.

---

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for ownership and execution flow and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for third-party
attribution.
