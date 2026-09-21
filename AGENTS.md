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
message serialised by one core must round-trip in the other, and (the load-bearing
part) both cores must agree on the decision (`resolve`/`reduce`/`fold`/
`corroborate`). The wire only needs to round-trip; byte-identity is not required
(a whole-number `f64` may print differently). There is no clock and no dedup in
the core — versioning lives in the persistence port (the product's database). If
you change a wire / message / sync type:

1. Change it on **both** sides in the same PR.
2. Keep field names, encodings, and the canonical form identical.
3. Keep the shared golden-vector guards green — the *same* JSON files under
   `tests/{vectors,gate-vectors,budget-vectors,config-vectors,corroborate-vectors,norm-vectors,reduce-vectors}/*.json`,
   asserted by **both** cores (`crates/sporewright/tests/*_vectors.rs` and
   `packages/sporewright/src/*-vectors.test.ts`). Per-core unit tests alone are
   not a cross-core guard: pin changed decisions with a shared vector.

Purely internal/local types are not wire types and have more freedom — but when in
doubt, treat a type as wire-relevant.

## Project layout

```
crates/sporewright          the Rust core: the tensor (cells, value/weight, the
                            operators fold/resolve/reduce/corroborate), the
                            device-mesh sync, the persistence port, plus the
                            budget/price, declarative-config, normalization and
                            scheduling policy modules
packages/sporewright        the decision-equivalent TypeScript core (tensor, sync,
                            persist, budget, config, normalize, plus TS-side trust
                            math)
packages/router             the LLM adopter: cascade resolution + fallback execution
                            over a sporewright tensor (Vercel AI SDK adapters)
tests/                      the shared golden-vector corpus (one JSON set, asserted
                            by both cores)
docs/                       the model (MODEL.md), the routing guide
                            (ROUTING-MODEL.md), and the OpenSSF self-assessment
```

The Rust public surface is `crates/sporewright/src/lib.rs`; the TS surface is
`packages/sporewright/src/index.ts`. Read the real source for exact
signatures — do not invent API. Note the two intentional asymmetries: `trust`
is TS-only (pure cross-device math, not a tensor method) and `schedule` is
Rust-only (generic work-scheduling model with no TS twin yet).

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
LGPL-3.0-linking-exception** (see `LICENSE`, `LICENSE.md`, `NOTICE`). Every
external contribution must be submitted through a pull request whose description
contains this exact affirmation:

> I have read and agree to version 1.0 of the Individual Contributor License Agreement at https://github.com/corbet-foss/.github/blob/cla-v1.0/CLA.md.

The pull-request record is the acceptance record; authors retain copyright in
their work. See `CONTRIBUTING.md` for the full terms.

## Reporting

- Bugs / features: <https://github.com/corbet-foss/sporewright/issues> (use the
  templates).
- **Security vulnerabilities:** do not open a public issue — see
  [`SECURITY.md`](SECURITY.md).
