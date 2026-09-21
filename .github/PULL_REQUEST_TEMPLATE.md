<!-- Thanks for contributing to sporewright! Keep PRs small and focused. -->

## What this changes

<!-- A short description of the change and why. Link any related issue. -->

## Checklist

- [ ] `cargo test --workspace` passes
- [ ] In `packages/sporewright`: `bun test` and `bunx tsc --noEmit` pass
- [ ] `cargo fmt --all --check` and `cargo clippy --workspace --all-targets -- -D warnings` are clean
- [ ] **Decision-equivalence:** if I touched a wire / message / protocol type, I changed it on **both** the Rust and TypeScript sides and the shared golden-vector guards stay green (see `CONTRIBUTING.md`)
- [ ] New source files carry the `// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception` header
- [ ] No private/internal context (hostnames, IPs, credentials, deployment details) in code, comments, or commits
- [ ] Updated `CHANGELOG.md` under `Unreleased` if the change is user-visible
- [ ] External contribution: the PR description contains the exact CLA affirmation in `CONTRIBUTING.md`
