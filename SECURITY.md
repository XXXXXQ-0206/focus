# Security Policy

Focus is an open-source coding agent with an auditable host harness. We take
the security of the software and its users seriously and welcome responsible
disclosure of vulnerabilities.

## Supported versions

Security fixes are provided for the latest tagged release on the `main`
branch. Older releases are supported on a best-effort basis for a reasonable
transition window. We recommend always running the latest release and
keeping dependencies up to date.

## Reporting a vulnerability

Please **do not** open a public issue for a suspected security vulnerability.
Instead, report it privately using one of the following channels:

- Open a GitHub private vulnerability report using the **Security** tab of
  this repository.
- Email the maintainers at the address listed in the issue tracker or the
  maintainer profile.

Please include as much of the following as possible:

1. A description of the vulnerability and the affected version.
2. The affected component or crate (`focus-kernel`, `focus-runtime`,
   `focus-cli`, or `focus-release-compliance`).
3. Steps to reproduce, or a minimal proof-of-concept.
4. The impact and any suggested mitigation.
5. Whether the issue has been disclosed publicly.

We aim to acknowledge reports within a few business days and will keep you
informed of progress. We ask that you allow a reasonable window before public
disclosure.

## Security guarantees and limitations

Focus makes security-related claims that are important to understand:

- Provider credentials are read from the process environment and are never
  stored in repository files, session transcripts, or replay artifacts.
- URLs (including query strings, fragments, and embedded credentials) are
  redacted before they enter the transcript, JSONL replay, approval records,
  or later model context.
- The native sandbox enforces workspace paths and process controls, but it is
  **not** an operating-system isolation boundary. Use the Docker or Podman
  container backends when process-level network isolation is required.
- Network access is allowlisted by default; enable the `web_fetch` tool
  explicitly and restrict domains as needed.
- The project is provided under the terms of the [LICENSE](LICENSE), which
  includes an **AS IS** disclaimer. See the README for the full disclaimer.

## Dependency scanning and updates

The repository enables:

- **Dependabot** for regular dependency update pull requests and automated
  security updates.
- **GitHub Secret Scanning** and **Push Protection** to block accidental
  commits of credentials.
- **CodeQL** static analysis on every push and pull request.

If a dependency has a known security advisory, keep the lockfile current and
open a pull request that resolves the advisory.

## Security-related configuration

See the `ARCHITECTURE.md` document for the detailed threat model and the
boundaries of the sandbox, network, and policy features.
