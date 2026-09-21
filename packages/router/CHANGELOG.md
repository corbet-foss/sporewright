# Changelog

All notable changes to the `router` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-20

**Relicense to LGPL-3.0-only WITH LGPL-3.0-linking-exception** (see the
repository-root `LICENSE`, `LICENSE.md`, `NOTICE`). No code changes beyond SPDX
headers, the injected-logger fix, and the new CI job. Previously distributed
versions remain available under Apache-2.0. Not published (`private`).

> Note: the `0.1.0` identity was never published; the entries below describe
> work first released under `0.2.0`.

### Added

- **Initial `router` package.** The generic provider-fallback LLM execution +
  tensor-cascade resolution layer over the sibling sporewright tensor.
  - `runAttemptChain` — dependency-free fallback engine (timeout/abort/critic/
    skip semantics, per-attempt trace).
  - `callWithChain` + `createModel` — LLM orchestrator and AI-SDK model factory,
    with AI-SDK error classification, `RateLimitError` / `ChainExhaustionError`,
    and an injectable cooldown/dedup `RouterState`.
  - `resolveCascadeTensor` / `resolveChain` + the chain-key address helpers — the
    cascade order emerges from a `resolve` over the override-tier tensor.
  - The Vercel AI SDK adapter registry (`ADAPTERS`, `getAdapter`, `getLlmAdapter`)
    and the provider catalog (`PROVIDERS`, `Capability`).
  - The `KeyStore` interface — the host-supplied key/cascade-config seam.
- Three host-injected seams, all defaulted, so the package compiles bundler-agnostic
  (no `import.meta.env`): an `onTrace` observer for telemetry, an injected `logger`
  (default no-op), and an injectable `RouterState` (default module singleton).
