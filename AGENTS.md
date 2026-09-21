# AGENTS.md

Guidance for AI coding agents (and humans) working in **sporewright** — a
self-learning, multi-objective router shipped as a Rust crate and a
decision-equivalent TypeScript package. This is the contributor quick reference;
[`CONTRIBUTING.md`](CONTRIBUTING.md) is authoritative for anything not here.

## Prerequisites

- **Rust** — a current stable toolchain (`rustup`), 2021 edition. `rustfmt` and
  `clippy` components.
- **Bun** — for the TypeScript core (<https://bun.sh>).

## Build, test, lint — run all of these before declaring a change done

**Rust** (repository root):

```bash
cargo build --workspace
cargo test --workspace                          # unit + doc tests
cargo fmt --all --check                         # must be clean
cargo clippy --workspace --all-targets -- -D warnings
```

**TypeScript** (`packages/sporewright`):

```bash
bun test            # unit + wire round-trip tests
bunx tsc --noEmit   # strict type-check, no build output
```

## The one rule you must not break: decision-equivalence

The Rust serde types and the TypeScript types are **decision-equivalent** — a
field serialised by one core must restore in the other, and both cores must agree
on addressed-field decisions, compaction, and discounting. The wire only needs to
round-trip; byte identity is not required. If you change shared state or behavior:

1. Change it on **both** sides in the same PR.
2. Keep field names, encodings, and the canonical form identical.
3. Add or update a shared vector under `tests/field-vectors/`; per-core unit tests
   alone are not a cross-runtime guard.

Purely internal/local types are not wire types and have more freedom — but when in
doubt, treat a type as wire-relevant.

## Project layout

```
crates/sporewright          Rust addressed field, Gaussian messages, curiosity,
                            and temporarily retained legacy modules
packages/sporewright        decision-equivalent TypeScript core
packages/router             LLM adopter and fallback executor
tests/field-vectors         shared cross-runtime behavior corpus
docs/                       normative model, routing guide, assessment
```

New integrations use `field` + `explore`. The old `tensor` modules remain only
while adopters migrate. The Rust public surface is `crates/sporewright/src/lib.rs`;
the TypeScript surface is `packages/sporewright/src/index.ts`.

## Conventions

- Every **new** `.rs`/`.ts` source file starts with
  `// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception`. (No header for `.md`/`.yml`.)
- Match the surrounding style; the core is intentionally small, pure, and
  dependency-light.
- **No private/internal context** anywhere (hostnames, IPs, credentials,
  deployment details). This repo is public.
- **No tool-generated attribution** in commits (no `Co-Authored-By:` lines).

## Contribution workflow

`main` is the release line. Non-trivial changes go through a pull request:

1. Branch off `main`: `git checkout -b <type>/<short-name>`.
2. Make the change; run the full build/test/lint set above.
3. Open a PR (the template lists the checklist). Required CI: `ci.yml` (Rust +
   TypeScript) plus CodeQL.
4. Update `CHANGELOG.md` under `Unreleased` for user-visible changes.

Contributions are licensed under **LGPL-3.0-only WITH
LGPL-3.0-linking-exception** (see `LICENSE`, `LICENSE.md`, `NOTICE`). Before a contribution is merged, its author agrees
to the organization-wide [Individual Contributor License Agreement](https://github.com/corbet-foss/.github/blob/cla-v1.0/CLA.md);
authors retain copyright in their work.

## Reporting

- Bugs / features: <https://github.com/corbet-foss/sporewright/issues> (use the
  templates).
- **Security vulnerabilities:** do not open a public issue — see
  [`SECURITY.md`](SECURITY.md).
