# Third-Party Notices

This file records third-party material directly integrated by this workspace.
It separates dependency and technical-acceptance evidence from any later
production release decision.

## Incorporated MIT Pi source

`focus-kernel::AgentLoop` is a Focus-owned Rust implementation whose behavior
is derived from the original MIT Pi agent loop. The source reference is the
pinned archive
`references/upstream/pi-40a3d8556ab7fb4a6b4da20ffe1f5dfc08ec121d.tar.gz` at
commit `40a3d8556ab7fb4a6b4da20ffe1f5dfc08ec121d`, SHA-256
`93733FD212F809324DFD21B493FBAB310606068A3E94C008A1F7956CA5868959`.
The referenced Pi project is MIT licensed, copyright Mario Zechner (2025).
The port uses the semantics of `packages/agent/src/agent-loop.ts`,
`agent.ts`, and `types.ts`; it does not link or copy code from
`pi_agent_rust` or another Rust rewrite. The MIT notice remains part of this
repository's source provenance.

## Incorporated Codex CLI TUI architecture

Focus's terminal frame scheduling, slash-command filtering, streaming-summary
presentation, and Markdown projection are adapted from the fixed Codex CLI
reference archive
`references/upstream/codex-279b93242cfef379e65da97e87e44b83c5934fd7.tar.gz`
at revision `279b93242cfef379e65da97e87e44b83c5934fd7`, SHA-256
`E7FCDD430E7265ADF139361DF438001701351C0D4031033EA099C7D9AC4A6B0A`.
The reference project is Apache-2.0 licensed. Focus keeps only a local,
focused Rust adaptation of the relevant patterns in
`crates/focus-cli/src/tui/`; it neither links against nor vendors Codex CLI
crates. Source comments identify the corresponding reference modules.

## Automated dependency inventory

The canonical machine-readable inventory is
[THIRD_PARTY_LICENSES.json](THIRD_PARTY_LICENSES.json). It is generated from
the exact `Cargo.lock` package set and resolved Cargo registry manifests; it
records direct/transitive relationship, source, checksum, SPDX `license` or
`license-file`, the first license heading, and review flags for every registry
package.

Regenerate and run the technical gate with:

```powershell
cargo run --locked --offline -p focus-release-compliance -- --mode technical --workspace . --inventory THIRD_PARTY_LICENSES.json
```

The current workspace snapshot contains 314 registry packages: 18 direct and
296 transitive. The resulting inventory has `metadata_issues=0`,
`rider_packages=0`, and `review_required=false`; both technical and production
dependency gates pass.

```powershell
cargo run --locked --offline -p focus-release-compliance -- --mode production --workspace .
# exits 0 when metadata is complete and no Rider package remains
```

Selected direct dependency metadata in this snapshot is:

| Package | Version | License metadata |
| --- | --- | --- |
| `async-trait` | 0.1.92 | MIT OR Apache-2.0 |
| `command-group` | 5.0.1 | Apache-2.0 OR MIT |
| `crossterm` | 0.29.0 | MIT |
| `fs4` | 0.13.1 | MIT OR Apache-2.0 |
| `futures` | 0.3.33 | MIT OR Apache-2.0 |
| `reqwest` | 0.13.4 | MIT OR Apache-2.0 |
| `serde` | 1.0.229 | MIT OR Apache-2.0 |
| `serde_json` | 1.0.151 | MIT OR Apache-2.0 |
| `thiserror` | 2.0.20 | MIT OR Apache-2.0 |
| `tokio` | 1.53.1 | MIT |
| `unicode-segmentation` | 1.13.3 | MIT OR Apache-2.0 |
| `unicode-width` | 0.2.2 | MIT OR Apache-2.0 |
| `uuid` | 1.24.0 | Apache-2.0 OR MIT |
| `pulldown-cmark` | 0.13.4 | MIT |
| `ratatui` | 0.30.2 | MIT |

## command-group 5.0.1

- Cargo package: `command-group`
- Resolved version: `5.0.1`
- Cargo configuration: exact version `=5.0.1`
- Cargo.lock checksum:
  `a68fa787550392a9d58f44c21a3022cfb3ea3e2458b7f85d3b399d0ceeccf409`
- License: `Apache-2.0 OR MIT`

The Runtime uses this crate as the canonical native process-tree adapter:
Unix process groups and Windows Job Objects are exposed through one safe Rust
API, keeping OS-specific unsafe code outside this workspace. The inspected
registry notice is:

```text
%CARGO_HOME%\registry\src\index.crates.io-*\command-group-5.0.1\COPYRIGHT
```

## Acceptance boundary

Technical acceptance means that the workspace resolved the exact package and
that source/build/test evidence demonstrated the claimed integration behavior.
It does not by itself approve publication, redistribution, deployment, or other
production use.

Production release authorization is a separate recorded decision. It must
consider the incorporated MIT Pi notice, the releasing entity and intended
recipients/operators, the produced artifact contents, and any required notices
or permissions. A passing build or test result is not a substitute for that
release decision.

The gate does not infer release approval from a technical build. A production
artifact is publishable only after the production command exits 0 with a
complete registry cache and an explicit review of the incorporated MIT Pi
notice and every flagged package.



