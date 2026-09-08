# GitHub Copilot Instructions

This file guides GitHub Copilot when it reviews and contributes to this
repository.

## Project overview

Focus is a Codex-inspired, evidence-driven Rust coding agent with an auditable
host harness. It is a Cargo workspace with four crates:

- `crates/focus-kernel` -- the owned MIT-Pi-semantic agent loop.
- `crates/focus-runtime` -- context, sessions, memory, policy, tools,
  provider adapters, MCP, workflow gates, sandbox backends, and subagents.
- `crates/focus-cli` -- the `focus` binary and the `focus-harness` alias.
- `crates/focus-release-compliance` -- the reproducible dependency/license
  gate.

## Conventions

- Code is formatted with `rustfmt` and linted with Clippy (`-D warnings`).
  Keep the code warning-free.
- `unsafe` is forbidden at the workspace level.
- Use `error`/`Result` and `thiserror` for failure handling; avoid panics in
  library code.
- Keep the MIT-Pi loop semantics in `focus-kernel` intentionally minimal and
  avoid adding product-specific behavior there.
- API contracts are documented; missing docs are a warning.

## Review focus

When reviewing a pull request, look for:

- Correctness, concurrency, and cancellation semantics in the async execution
  spine.
- Security-sensitive behavior: credential redaction, URL redaction, network
  allowlist enforcement, sandbox boundaries, and path confinement.
- Cross-platform (Windows / Linux) portability, especially around process
  spawning, terminal handling, and file locking.
- Consistency between the README (English and Chinese versions) and the
  documented behavior.
- Test coverage for the behavior being added or changed.

## Common commands

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --locked --offline
cargo build --workspace --release --locked --offline
cargo run --locked --offline -p focus-release-compliance -- --mode production --workspace .
```

## Do not

- Commit credentials, API keys, or local absolute paths.
- Commit build output, `target/`, `.focus-harness/`, `docs/acceptance/`,
  `references/`, `tools/`, or other internal development artifacts.
- Introduce changes that break the release compliance gate.
