# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-08

Initial public release of Focus, a Codex-inspired, evidence-driven Rust coding
agent with an auditable host harness.

### Added

- `focus-kernel`: an MIT-Pi-semantic agent loop with normalized
  provider/tool contracts, streaming model events, one canonical event stream,
  bounded ordered concurrent tools, and a single terminal state for
  cancellation, failure, and turn limits.
- `focus-runtime`: context, session ancestry, project/session memory, policy
  engine, built-in tools, provider adapters (HTTP and command), real MCP
  transport, workflow gates, sandbox backends (native, Docker, Podman), and a
  first-class subagent lifecycle.
- `focus-cli` (`focus` binary and `focus-harness` compatibility alias): an
  interactive Codex-style TUI, a line-oriented host for PTY/CI, a structured
  JSONL host boundary for GUI hosts, session/goal/memory/subagent commands,
  an ablation benchmark, and self-review/self-update supervision.
- Reproducible dependency compliance gate (`focus-release-compliance`).
- Multilingual documentation: English and Chinese README.
- CI (Linux + Windows), CodeQL, Dependabot, and repository security files.

### Security

- Provider credentials are read from the process environment and redacted
  from transcripts and replay artifacts.
- Network access is allowlisted by default; URL query strings, fragments, and
  embedded credentials are redacted before entering model context.
- Codex-style permission modes: `--yolo`, `--interactive`, and `--deny`.
- Sandbox backends with process/container limits and optional no-network
  containers.

### Notes

- The primary binary is `focus`; `focus-harness` remains as the compatibility
  alias documented in the README.
- The project is dual-licensed under **MIT OR Apache-2.0**.
