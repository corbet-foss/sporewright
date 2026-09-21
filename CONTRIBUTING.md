# Contributing to sporewright

Thanks for your interest in sporewright. It's a small, pure engine shipped as
**two cores** — a Rust crate (`crates/sporewright`) and a TypeScript package
(`packages/sporewright`). The single most important rule for contributors flows
from that: **the two cores must stay decision-equivalent** — they must agree on
addressed-field resolution, compaction, and discounting, and shared state must
round-trip between them. Read
the equivalence section below before touching any serialisable type.

This is a `0.1.0`, pre-1.0 project. Things can still move. Please be accurate
rather than aspirational in anything you document.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting security issues

Please **do not** report security vulnerabilities through public issues or pull
requests. Follow the private disclosure process in [`SECURITY.md`](SECURITY.md)
instead.

## Prerequisites

- **Rust** — a stable toolchain (`rustup` recommended). The crate targets the
  2021 edition; a current stable `rustc`/`cargo` is sufficient. `rustfmt` ships
  with the standard toolchain (`rustup component add rustfmt` if missing).
- **Bun** — for the TypeScript core. Install from <https://bun.sh>. Bun runs the
  tests and we use `bunx` to invoke the TypeScript compiler for type-checking.

No global installs of the library are required — you build and test it from this
checkout.

## Build and test — both sides

Both cores must be green before a change is complete. Run all four:

**Rust** (from the repository root):

```sh
cargo test --workspace        # unit + doc + integration tests, both crates
cargo fmt --all -- --check    # formatting must be clean
```

**TypeScript** (from `packages/sporewright`):

```sh
bun test            # unit + wire tests
bunx tsc --noEmit   # type-check only, no build output
```

If you change a wire/message/protocol type, the wire tests on **both** sides are
the gate — see below.

## Code style

- **Rust:** format with `rustfmt` (`cargo fmt --all`). CI-style checks expect a
  clean `cargo fmt --all -- --check`. Idiomatic, dependency-light Rust; the core
  is intentionally small and pure.
- **TypeScript:** match the surrounding style in `packages/sporewright/src`.
  `bunx tsc --noEmit` must pass with no errors.
- **SPDX header:** every **new** source file (`.rs` or `.ts`) must start with:

  ```
  // SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
  ```

  Markdown (`.md`) and YAML (`.yml`) files do not need this header.

## The equivalence invariant (read this)

sporewright is two implementations of one engine. The load-bearing rule is
**decision-equivalence**: given the same field declarations, observations, and
maintenance operations, both cores must produce the same ranked decision — that is
what lets a device (TypeScript) and an orchestrator (Rust) agree. The serialised wire **round-trips** (each
core parses what the other emits); it is *not* required to be byte-identical
(e.g. a whole-number `f64` may print differently), because the **decision**, not
the bytes, is the contract.

This invariant is **guarded by the shared golden-vector corpus** under
`tests/{field-vectors,vectors,gate-vectors,budget-vectors,config-vectors,corroborate-vectors,norm-vectors}/*.json`:
the *same* JSON files are loaded and asserted by **both** cores
(`crates/sporewright/tests/*_vectors.rs` and `packages/sporewright/src/*-vectors.test.ts`),
so a routing decision that changes on one core breaks the build on both. Those shared
vectors are the real cross-core guard.

The per-core `*.test.ts` and Rust module unit tests are **not** cross-core guards — each
side authors its own expectations, which can drift apart silently; they cover single-core
behaviour. When you change a decision, pin it with a **shared vector**, not a per-core
literal. Run both:

- Rust: `cargo test -p sporewright` (module unit tests **and** the `*_vectors` integration guards).
- TypeScript: in `packages/sporewright`, `bun test` (unit tests **and** the `*-vectors` guards).

**Rules for any change to a serialisable (cell/wire) type:**

1. Make the change on **both** sides — Rust and TypeScript — in the same PR.
   Never land one side alone.
2. Keep the decision identical and the wire round-tripping: field names, the JSON
   field shape and numeric encodings must line up so each core parses
   the other's output and resolves the same.
3. Keep both test suites green (`cargo test --workspace` and, in
   `packages/sporewright`, `bun test`). If you add a field, extend the cross-core
   fixture so the equivalence stays guarded.

Types that are *not* serialised across the boundary (purely internal helpers)
have more freedom, but when in doubt, treat a type as wire-relevant and check
both sides.

## Pull requests

- Keep PRs focused. Touch only the files your change needs.
- Run all four commands above and make sure they pass.
- Update `CHANGELOG.md` under an `Unreleased` section when your change is
  user-visible (new feature, behaviour change, fix, breaking wire change).
- Don't include private or internal context (hostnames, IPs, deployment
  credentials, adopter internals) in code, comments, commits, or PR text. This
  project is headed toward public release; keep everything publication-clean.

## Licensing

Sporewright is licensed under **LGPL-3.0-only WITH LGPL-3.0-linking-exception**
(see [`LICENSE`](LICENSE), [`LICENSE.md`](LICENSE.md) and [`NOTICE`](NOTICE)).
Previously distributed versions remain available under their original grants (see
[`LICENSES/FSL-1.1-ALv2.txt`](LICENSES/FSL-1.1-ALv2.txt),
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) and [`CHANGELOG.md`](CHANGELOG.md)).

To keep the project coherently stewarded, external contributors
retain copyright in their work and grant the rights described in Corbet Labs'
organization-wide [Individual Contributor License Agreement](https://github.com/corbet-foss/.github/blob/cla-v1.0/CLA.md).

Every external contribution must be submitted through a pull request whose
description contains this exact affirmation:

> I have read and agree to version 1.0 of the Individual Contributor License Agreement at https://github.com/corbet-foss/.github/blob/cla-v1.0/CLA.md.

The checked affirmation and pull-request record form the electronic acceptance
record. A maintainer must not merge an external contribution without it. Forking
the repository to prepare a pull request is expected and welcome.

## See also

- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — community expectations.
- [`SECURITY.md`](SECURITY.md) — how to report a vulnerability privately.
- [Individual Contributor License Agreement](https://github.com/corbet-foss/.github/blob/cla-v1.0/CLA.md) — the organization-wide contribution terms.
