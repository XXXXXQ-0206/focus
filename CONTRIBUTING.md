# Contributing to Focus

Thanks for taking the time to contribute to Focus. We welcome bug reports,
feature requests, documentation improvements, and code changes. This file
explains how to set up the project, how to make a change, and how to get it
reviewed and merged.

## Code of Conduct

By participating in this project, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting started

### Prerequisites

- Rust toolchain matching the pinned version in [`rust-toolchain.toml`](rust-toolchain.toml) (1.97.1).
- `cargo`, `rustfmt`, and `clippy` (installed by the toolchain).
- Git.

### Building

```bash
cargo build --workspace --release --locked
```

### Running the CLI

The primary binary is `focus` (installed into `crates/focus-cli`), with
`focus-harness` retained as a compatibility alias:

```bash
cargo run --release -p focus-harness -- chat --workspace .
```

## Development workflow

We use a simple GitHub-flow style workflow:

1. Fork the repository.
2. Create a feature branch from `main` (e.g. `codex/feat-description`).
3. Make your changes and commit them with a conventional commit message.
4. Push the branch and open a pull request into `main`.
5. Keep the PR small and focused. Reference the relevant issue in the
   description.

### Branch and commit conventions

Branch names use the `codex/` prefix by convention but any descriptive name
is fine. Commit messages follow
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```text
feat: add a new capability
fix: correct a bug in the provider adapter
docs: explain the sandbox modes
refactor: simplify the tool scheduling path
test: add coverage for session recovery
chore: update dependencies
```

Keep each commit atomic and scoped to a single logical change.

## Local checks

All of the following must pass locally before you open or update a pull
request:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --locked --offline
cargo build --workspace --release --locked --offline
cargo run --locked --offline -p focus-release-compliance -- --mode production --workspace .
```

The continuous integration (CI) workflow in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs these checks on
both Linux and Windows.

## Security

If you believe you have found a security vulnerability, please **do not** open
a public issue. Follow the disclosure process described in
[`SECURITY.md`](SECURITY.md) instead.

## Adding a dependency

Prefer first-party or widely maintained crates. When you add a dependency,
keep the lockfile in sync (`cargo update`) and make sure the release
compliance gate still passes.

## Documentation

API and architecture documentation are maintained in the repository. When
you change user-facing behavior, update the relevant section of the README
(both `README.md` and `README.zh-CN.md` stay in sync) and the architecture
notes if needed.

## Reporting bugs

1. Search the issue tracker to avoid duplicates.
2. Include the Focus version, operating system, and the exact command.
3. Provide a minimal reproduction and the full error output.
4. Attach any relevant logs, stripped of personal information.

## License

By contributing, you agree that your contributions are licensed under the
same terms as the project: **MIT OR Apache-2.0**. See
[`LICENSE`](LICENSE) and [`LICENSE-MIT`](LICENSE-MIT).
