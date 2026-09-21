# Security Policy

sporewright is a small, pure routing engine shipped as two decision-equivalent
cores (a Rust crate and a TypeScript package) plus an orchestrator runtime. We
take security reports seriously and appreciate responsible disclosure.

## Supported versions

sporewright is pre-1.0 (`0.1.0`). Until a stable `1.0` release, only the latest
published version receives security fixes. Once `1.0` ships, this table will be
updated to reflect the supported release line(s).

| Version  | Supported          |
|----------|--------------------|
| `0.1.x`  | :white_check_mark: (latest) |
| `< 0.1`  | :x:                |

Because the project is not yet published to crates.io or npm, "latest" means the
most recent tagged release / `main` of the repository.

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.** Public issues
disclose the problem before a fix is available and put users at risk.

Instead, report privately through GitHub's built-in private vulnerability
reporting:

1. Go to the repository's **Security** tab.
2. Choose **Report a vulnerability** (GitHub Private Security Advisories).
3. Describe the issue with enough detail for us to reproduce it.

A good report includes:

- The affected component (Rust crate, TypeScript package, or orchestrator
  runtime) and version / commit.
- A clear description of the vulnerability and its impact.
- Step-by-step reproduction, ideally a minimal proof of concept.
- Any suggested remediation, if you have one.

If you are unable to use GitHub's private advisory flow, open a regular issue
that contains **no exploit details** and simply asks a maintainer to enable a
private channel — we will follow up there.

## Disclosure expectations

We aim to handle reports on the following timeline. As a small, pre-1.0 project
maintained on a best-effort basis, these are targets rather than guarantees:

- **Acknowledgement:** within 5 business days of receiving the report.
- **Initial assessment:** within 10 business days, including whether we can
  reproduce the issue and a rough severity.
- **Fix and disclosure:** coordinated with the reporter. We prefer to publish a
  fix and a security advisory together, and to credit the reporter (unless they
  ask to remain anonymous).

Please give us a reasonable window to release a fix before any public
disclosure. We will keep you informed of progress throughout.

## Scope

In scope: the engine cores (`crates/sporewright`, `packages/sporewright`),
including the write-down (capability handle) and the cross-runtime
decision-equivalence / decision-agreement surface.

Out of scope: vulnerabilities in third-party dependencies (please report those
upstream), and issues that require a misconfigured or fully-trusted deployment
environment to exploit.

## Safe harbor

We will not pursue or support legal action against researchers who act in good
faith, follow this policy, avoid privacy violations and service disruption, and
give us a reasonable time to respond before any public disclosure.
