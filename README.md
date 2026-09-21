<p align="center">
  <img src="assets/logo.svg" alt="sporewright" width="380">
</p>

# sporewright

[![CI](https://github.com/corbet-foss/sporewright/actions/workflows/ci.yml/badge.svg)](https://github.com/corbet-foss/sporewright/actions/workflows/ci.yml)
[![License: LGPL-3.0-only WITH exception](https://img.shields.io/badge/license-LGPL--3.0--only-blue.svg)](LICENSE.md)
[![Docs](https://img.shields.io/badge/docs-the%20model-brightgreen.svg)](docs/MODEL.md)
[![Cores](https://img.shields.io/badge/cores-Rust%20%2B%20TypeScript-orange.svg)](#the-object)

A framework for **participant-relative, multi-objective routing**. It keeps the
available possibilities in a sparse tensor until a participant combines shared
knowledge with its own circumstances and realizes the locally appropriate order.
It answers one question for any kind of work (LLM calls, scraping, rendering,
embedding…) across providers or execution environments (browser tabs, shells,
containers, servers, reachable peers):

> Given everything that *could* do this, which options are viable here and now, in
> what order should this participant try them, and how should the outcome reshape
> later decisions?

The tensor is shared conceptually, not necessarily as one byte-identical global
object. Participants share its coordinate language, operators, and exchangeable
facts, while local measurements, reachability, resources, policy, and task context
may shape different materializations and therefore different correct results. Facts
flow both ways: shared policy influences local realization; local outcomes become
observations that can influence other participants later.

The engine has **no opinions about your domain or network topology**. The same core
ships in **Rust and TypeScript, decision-equivalent**, so participants apply the same
mathematics even when their current facts differ. Persistence and exchange are
integration choices: CareerVector can use peers and collaborative state; JobCache can
put an orchestrator at the prominent edge.

## The local materialization

A participant evaluates one sparse materialization of the tensor:

```
T[ level ][ instance ][ option ][ dimension ]  →  value
```

- **level** — a fixed, ordered hierarchy, coarse→fine (`global ≺ workspace ≺ … ≺ device`).
- **instance** — the horizontal axis a level spawns: `global` is one (`""`); `workspace`/`device` branch one-per-id.
- **option** — the candidates; the resolved **queue is a vector along this axis**.
- **dimension** — the per-option decision parameters (`latency`, `financial`, `reliability`, a gate…). A dimension's **weight** lives one axis coarser.

You read with a **cursor** (one instance per axis — `{workspace: acme, device: A}`).
The realization is an ordered viable set, not an executed side effect: the product
tries an option, reports the outcome, and decides when to stop. Three core operators:

- **fold** — collapse the level axis; the deepest set cell wins (most-specific inheritance, the CSS cascade).
- **resolve** — collapse the dimension axis; `Σ w·v`, a `+∞` gate drops the option, order ascending → the queue.
- **reduce** — the one path *up*: the median over a level's instances, the orchestrator's trusted aggregation.

Writes obey **write-down** (a device sets its own slice, never the shared levels), and
robustness is the orchestrator reading where evidence is thin — not metadata in the
cells. The full model, maths, and design choices are in **[docs/MODEL.md](docs/MODEL.md)**;
for *why* one `resolve` becomes a self-learning, multi-objective, resource-aware router
— the three-role dimension vector, the Lagrangian-dual maths, and eight worked stories —
read **[docs/ROUTING-MODEL.md](docs/ROUTING-MODEL.md)**.

## Quickstart

**Rust** (`crates/sporewright`)

```rust
use sporewright::tensor::{Tensor, Cursor};

let mut t = Tensor::new(["global", "workspace", "device"]);
let mut w = t.writer("global").unwrap();
w.set_value("global", "", "groq", "latency", 0.4).unwrap();
w.set_value("global", "", "cerebras", "latency", 0.3).unwrap();
w.set_weight("global", "", "latency", 1.0).unwrap();

// a premium workspace loosens a weight on its own slice; a device records its own value
w.set_value("device", "A", "groq", "latency", 0.05).unwrap();

let mut cursor = Cursor::new();
cursor.insert("device".into(), "A".into());
assert_eq!(t.resolve(&cursor), vec!["groq".to_string(), "cerebras".to_string()]);
```

**TypeScript** (`packages/sporewright`) — same model, decision-equivalent.

```ts
import { Tensor } from "sporewright";

const t = new Tensor(["global", "workspace", "device"]);
const w = t.writer("global")!;
w.setValue("global", "", "groq", "latency", 0.4);
w.setValue("global", "", "cerebras", "latency", 0.3);
w.setWeight("global", "", "latency", 1.0);
w.setValue("device", "A", "groq", "latency", 0.05);

t.resolve({ device: "A" }); // -> ["groq", "cerebras"]
```

## Build

```
cargo test -p sporewright          # the Rust core
cd packages/sporewright && bun test # the TypeScript core
```

The two cores are kept decision-equivalent (the wire round-trips); see
[CONTRIBUTING.md](CONTRIBUTING.md) and [SCOPE.md](SCOPE.md) (what lives in the
engine vs the product).
