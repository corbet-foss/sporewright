# OpenSSF Best Practices — self-assessment

This maps sporewright against the [OpenSSF Best Practices](https://www.bestpractices.dev/)
**passing** tier. Registering the project for the badge is a maintainer action;
this file is the evidence the registration draws on.

Legend: ✅ met · 🔶 pending (gated) · ➖ not applicable

## Basics

| Criterion | Status | Evidence |
|---|---|---|
| Project URL + clear description | ✅ | repo + `README.md` tagline and "Why" section |
| Contribution process documented | ✅ | `CONTRIBUTING.md`, `AGENTS.md`, PR + issue templates |
| OSI/FLOSS license, in a standard location | ✅ | `LICENSE` (LGPL-3.0-only WITH LGPL-3.0-linking-exception) + `LICENSE.md`, `LICENSES/`, `NOTICE`, SPDX headers on every source file |
| Basic + interface documentation | ✅ | `README.md`, `docs/MODEL.md`, in-source rustdoc/tsdoc (`crates/sporewright/src`, `packages/sporewright/src`) |
| Sites use HTTPS | ✅ | GitHub + crates.io/npm (on publish) |
| Discussion channel | ✅ | GitHub Issues + Discussions |
| English | ✅ | all docs in English |
| Code of Conduct | ✅ | `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) |

## Change control

| Criterion | Status | Evidence |
|---|---|---|
| Public, version-controlled (distributed) source | ✅ | public Git repo |
| Unique, semantic version numbers + tags | ✅ | `0.2.0`, SemVer; tags on release |
| Release notes | ✅ | `CHANGELOG.md` (Keep a Changelog) |

## Reporting

| Criterion | Status | Evidence |
|---|---|---|
| Bug-reporting process | ✅ | GitHub Issues + templates |
| Vulnerability-reporting process (private) | ✅ | `SECURITY.md` — GitHub private security advisories |
| Response expectations stated | ✅ | `SECURITY.md` |

## Quality

| Criterion | Status | Evidence |
|---|---|---|
| Working build from source, FLOSS tools | ✅ | `cargo build` (Rust), `bun` (TS) |
| Automated test suite | ✅ | `cargo test --workspace` + `bun test` (unit + doc + wire round-trip) |
| Tests invocable by a standard command | ✅ | documented in `CONTRIBUTING.md` |
| New functionality must add tests (policy) | ✅ | stated in `CONTRIBUTING.md` |
| Tests run in CI | ✅ | `.github/workflows/ci.yml` |
| Compiler/lint warnings are errors | ✅ | `clippy -D warnings`; strict `tsconfig` (`noUncheckedIndexedAccess`, etc.) |

## Security

| Criterion | Status | Evidence |
|---|---|---|
| Maintainers know secure design / common errors | ✅ | rogue-resistant aggregation, capability gating, least-privilege escalation |
| Cryptography is published, standard, FLOSS | ✅ | HMAC-SHA256 (`hmac`/`sha2` in Rust; dependency-free portable TS), cross-runtime vector-tested |
| Crypto uses an acceptable key length / no known-broken primitives | ✅ | SHA-256 |
| Perfect forward secrecy / password storage | ➖ | no transport crypto or stored passwords in scope |
| No leaked credentials in the repo | ✅ | history rebuilt clean; scanned tree + history |
| Secured delivery (signed packages) | 🔶 | gated on first crates.io/npm publish (registries sign; tagged releases) |

## Analysis

| Criterion | Status | Evidence |
|---|---|---|
| Static analysis | ✅ | `cargo clippy`; **CodeQL** workflow (Rust + TypeScript) |
| Static-analysis findings fixed | ✅ | clippy gate is `-D warnings` |
| Dynamic analysis / assertions in tests | ✅ | test suite + doctests; Rust debug assertions |
| Supply-chain posture | ✅ | **OpenSSF Scorecard** workflow (`.github/workflows/scorecard.yml`) |

## Pending (not blockers for passing)

- 🔶 **Badge registration** at bestpractices.dev — maintainer action.
- 🔶 **Signed package delivery** — happens automatically once published to crates.io + npm (currently gated; install is from source).
