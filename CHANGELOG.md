# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

**The engine is now one sparse tensor.** Everything sporewright does is a read or
write on `T[level][instance][option][dimension]` (see `docs/MODEL.md`): a cursor
slices it, `fold` collapses the level axis (most-specific inheritance), `resolve`
collapses the dimension axis into the ordered queue, and `reduce` medians measured
values up the instance axis (the one trusted path up). Writes obey write-down via a
capability `Writer`; robustness is the orchestrator reading where evidence is thin
(`support`), not metadata in the cells.

### Added

- **`Tensor`** (`tensor` module, Rust + a decision-equivalent TS core): `fold`,
  `resolve`, `reduce_median`/`roll_up`, `corroborate`, `support`, the `±∞` reach
  gate, and the `Writer` write-down handle; plus a `Device` (`sync`:
  observe/gate/attest) and a `Store` persistence port.

### Removed

- **The entire previous engine** — it is subsumed by the tensor. Gone: the
  `Catalog`, the multi-axis `CostVector`/`Weights` cost model, the `Stack`/`Layer`
  refinement engine + `ResolveTrace`, the `LearningTree`/`Meter`/`TrustLedger`
  learning + trust stack, the `Opt`/`Chain`/escalation/execute failover types, the
  `Hub`/`Orchestrator` protocol, the peer mesh helpers, and the `KeyStore` /
  `TelemetrySink` / `DataStore` ports - plus the `sporewright-catalog` and
  `sporewright-server` crates and all of their tests. The tensor replaces them.

## [0.2.0] - 2026-09-20

**Relicense to LGPL-3.0-only WITH LGPL-3.0-linking-exception.** The library is
now under the LGPL with a static/dynamic linking exception (no relinking or
installation-information duties for combined works; library modifications stay
LGPL) — see `LICENSE`, `LICENSE.md`, `LICENSES/` and `NOTICE`. Previously
distributed versions remain available under Apache-2.0 (`LICENSES/Apache-2.0.txt`
for reference; old tags keep their old grants — the new license rides this new
`0.2.0` version identity). Not published: `publish = false` and both npm
packages stay `private`. No code changes beyond SPDX headers.

### Changed

- **License**: `Apache-2.0` → `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
  in `crates/sporewright/Cargo.toml`, both `packages/*/package.json`, all
  source SPDX headers, and the README badge; new `LICENSE.md` (grant summary +
  Apache-2.0 grant history), `LICENSES/` (LGPL/GPL/exception texts + the
  historical Apache-2.0 copy), `THIRD-PARTY.md` (external deps retain their own
  permissive licenses — no vendored code), and updated `NOTICE` files.
- **Contributions**: `CONTRIBUTING.md`/`AGENTS.md` now point at the
  organization-wide Individual CLA v1.0 (exact PR-description affirmation) in
  place of inbound=outbound/no-CLA.
- **Repository home**: `julian-corbet/sporewright` → `corbet-foss/sporewright`
  in manifests, badges, and docs links.
- **Version**: `0.1.0` → `0.2.0` (Rust crate + both TS packages).
- **CI**: the `router` package gets its own type-check + test job (it was never
  exercised in CI); Rust/TS steps report independent verdicts instead of the
  first failure masking the rest.

### Fixed

- `schedule`: `priority`/`budget_from_saturation` no longer panic on NaN trust
  or saturation (NaN reads as no-signal: the `0.5` trust default / idle);
  `Budget` gains a manual `Default` (identical to `new()`, never a zero step).
- `budget` (TS): per-option pool totals accumulate in code-point order (as Rust's
  `BTreeMap` does), not UTF-16 code-unit order — pinned by the shared
  `08_astral_option_sum_order` golden vector.
- `router`: observer-failure warnings go through the injected `logger`
  (default no-op) instead of `console` directly.

## [0.1.0] - 2026-06-01

Initial release: the sporewright engine as two byte-equivalent cores, a Rust
crate (`crates/sporewright`) and a TypeScript package (`packages/sporewright`).

### Added

- **Multi-axis cost** — `CostVector` over canonical lower-is-better axes
  (financial, latency, reliability, quota, geo) with `Weights`-driven
  **weighted-sum scalarisation** (`Σ wᵢ·costᵢ`) folding the multi-objective cost
  into a single comparable scalar; hard constraints are capability gates /
  workspace blocks, not penalty terms.
- **Gradient model** — per-axis derivative chains, the gradient bundle, Pareto
  domination and `pareto_front`, and a Catalog of known-a-priori option costs.
- **N-layer refinement stack** — an ordered `Stack` of `Layer`s
  (require-capabilities, weights, order-by-cost, …) that resolves candidates
  into an ordered `Chain`. The layer count is a parameter, not baked in (the
  first two adopters instantiate the engine at 6 and 4 layers respectively).
- **ResolveTrace** — an enriched audit trail recording, per `Layer`, the
  surviving option ids plus their per-step weights and scalar cost, so a chain's
  ordering is fully explainable (surfaced on the orchestrator via
  `Orchestrator::explain`).
- **Failover execution** — `execute` runs a chain option-by-option, collecting
  `Protest`s by `ProtestClass` until one option succeeds or the chain reaches
  its escalation point.
- **Pre-computed escalation** — `Plan` with a visible prefix and a withheld
  reserve, protest-driven extension, and depth cutoff
  (`execute_with_escalation`).
- **Borda seed-merge consensus** — `borda` for merging seed orderings.
- **Online learning** — an EMA `Meter` (half-life configurable) that folds real
  measurements back into option scores, robust outlier gating with regime-shift
  escape, and a configurable N-level `LearningTree` whose `LearnConfig` routes
  each axis to its level (the orchestrator runs a `global` mean for fleet
  provider cost and a `workspace` mean for task-fitness `quality`). Per-device
  `DeviceDelta`s overlay locally via `apply_device_deltas` /
  `applyDeviceDeltas`, so the orchestrator stays device-agnostic. A
  resource-aware `ExplorePolicy` governs when to try an uncertain option.
- **Device ⇄ orchestrator protocol** — byte-equivalent `DeviceMessage` /
  `OrchestratorMessage` types and a server-side `Hub` (sessions, push,
  protest→extension, head-flip invalidation). `ProtestClass` routes escalation
  (transient classes reveal the reserve, terminal ones suppress); escalation
  validates a device's `tried` set against the options it was actually shown and
  rate-limits protests; `Hello` carries per-peer `PeerInfo{id, capabilities}`;
  and `Sample` carries an optional `peer`, folded under an `option@peer` key for
  per-peer learning.
- **Ports (vendor-neutral seams)** — a `KeyStore` (BYOK credentials, with a
  redacted `Secret`, a `MapKeyStore` default, and `store_capabilities`) and a
  `TelemetrySink` (structured `Event`s over `AttrValue`, emitted at the Hub's
  decision points, with `NoopSink`/`RecordingSink` defaults), both shipped in
  Rust and TypeScript; and a `DataStore` (Rust-only) for Hub persistence.
- **Server persistence** — `Hub::snapshot`/`Hub::restore` over a serialisable
  `HubState`, and a file-backed `FileStore` `DataStore` in
  `crates/sporewright-server`: with `SPOREWRIGHT_DATA_DIR` set, the server
  auto-restores on boot and persists after every mutation.
- **P2P peer-route** — abstract peer slots that the orchestrator authorises and
  the device binds to a concrete peer at call time.
- **Trust / attestation** — rogue-resistant per-device median aggregation
  (resistant to a single liar flooding measurements, with a configurable
  `min_reporters` quorum) and cross-runtime HMAC attestation over a canonical
  measurement encoding.
- **Deploy artifacts** — a `Dockerfile` for `sporewright-server` plus
  vendor-neutral `deploy/` recipes (`docker-compose.yml`, a systemd unit, and a
  `fly.toml`).
- **Byte-equivalent Rust + TypeScript cores** — the serde wire types and the TS
  types are identical on the wire, guarded by wire tests on both sides
  (`crates/sporewright/tests/wire.rs` and `packages/sporewright/src/*.test.ts`).
  Byte-equivalence is bounded to the serialisable wire types (cost, options,
  chains, protests, the resolve trace, the protocol envelope, telemetry events);
  the orchestrator `Hub` and its persistence are Rust-only.

### Notes

- Pre-1.0 (`0.1.0`): the API may change before a stable release.
- Not yet published to crates.io or npm. Build and use from source / git.
- Licensed under Apache-2.0.

[0.2.0]: https://github.com/corbet-foss/sporewright/releases/tag/v0.2.0
[0.1.0]: https://github.com/corbet-foss/sporewright/releases/tag/v0.1.0
