<!-- SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception -->
# The model — participant-relative realization from a sparse tensor

Sporewright models a field of possible actions as a sparse, high-dimensional tensor.
Each participant combines the shared facts it has received with local observations,
resources, reachability, policy, and task context. The engine **slices** that local
materialization into an ordered viable set. The product performs the action; its
outcome becomes another observation that may reshape later materializations.

“Shared tensor” therefore does not mean one universally identical in-memory object.
What participants share is the coordinate schema, deterministic operators, numeric
semantics, and exchangeable facts. Their materializations may legitimately differ.
The same possibility field can realize a different correct order on every participant.
This document is the algebraic contract; persistence and distribution are adapters.

The governing intuition: **the tensor is hyperdimensional but almost entirely
curled.** Every axis and every knob *exists*; sparsity means nearly none are
populated, so an axis you aren't using crumbles to a point and costs nothing. You
pay only for the knobs you turn. Generality is free when it's sparse — which is why
we never compact. (Humans compact dimensions because >4 makes the head dizzy; a
machine doesn't care, so we don't.)

---

## 1. A participant's materialization

A cell is a **fact at a coordinate**. The coordinate names a point on every axis:

| Axis | What it is |
|---|---|
| **context tiers** (`workspace`, `device`, `l0/l1/l2`, …) | each tier is *its own axis*, ordered coarse→fine for precedence. A coordinate gives each a value or leaves it curled (`""` = wildcard / "applies to all"). There is no separate "level + instance" — a tier *is* an axis and its id *is* the coordinate on it. |
| **option** | the candidates (a method, a `(provider,model)`, a peer). The resolved **queue is a vector along this axis**; `""` is the shared slot ("all options"). |
| **dimension** | the per-option objectives: `latency, reliability, financial, quality, reach, …` (the neutral word for "cost"). |

At each coordinate sit **two co-located scalar layers** — not a "role" axis (they
coexist, they don't exclude), but two factors that multiply where they line up:

- **value** — an observation or declaration available to this participant about an
  option on a dimension. `+∞` is a **gate** (drops the option); negative is allowed;
  no clamp. Opaque application documents are not decision values and remain outside
  the core algebra.
- **weight** — the *governing will*: how much that value matters.

Versioning is **not** on the cell — the persistence port (the DB) stamps each write's
version (§6); a cell carries only its `value` and `weight`. A **cursor** is a point in
the context space — `{workspace: acme, device: A}` — a *slice* that says where you are.

> `instance`, "levels", a bolted-on "context" coordinate, a value/weight "role" —
> all gone. They were compactions: bundling N context axes into one, or two layers
> into one axis. Uncompacted, it is just axes and coordinates.

---

## 2. The two layers, and the directions

**value** is filled from both ends, by *which coordinate you write* (never a stored
"kind" field): **measured** up the context axes via `reduce` (latency, reliability);
**static** at one coordinate (financial cost); **declared** down (model quality).

**weight** is the orchestrator's lever, written top-down (write-down, §4):

- **99% of the time it is one weight per dimension** — set at `option=""` (the
  shared slot), inheriting across every contender. That's the sparsity pattern, not
  a structural rule.
- **The grid also lets you pin a weight to a single cell** — one option, one
  dimension. That is the **gas pedal**: the orchestrator watches quota and cranks an
  over-budget model's weight so it sinks to last-resort, or drops it to promote one
  with idle capacity. `reach = +∞` is the hard kill switch; weight is the soft steer.

So **value = truth, weight = will; both per-cell, both sparse**, value leaning into
the option axis, weight mostly sitting at `option=""`. Same mechanism, different
fill — sparsity carries the generality.

---

## 3. The operators are slices

The stored object is the truth; every decision is a slice computed on demand and
**never stored**:

**fold** — collapse the context axes at a cursor. The **most-specific compatible
cell wins**: a cell is compatible if every context axis it pins matches the cursor
(or is `""`); "most specific" is decided by the tier precedence (finer out-specifies
coarser). This is CSS-cascade / git-config last-wins — *not* a deep-merge. Applies
to value and to weight independently (weight folds option-specific over `option=""`).

**resolve** — *what's available, and how it fares.* Enumerate the option axis at the
cursor; for each option contract the dimension axis:

$$s_c(o)=\sum_{k}\,V_c(o,k)\cdot W_c(o,k)\quad(\text{drop }o\text{ if any }V_c(o,k)=+\infty)$$

Order ascending by $s_c$ (ties by id). The vector along the **option** axis is the
**queue**. (A pure-document option, with no scalar dimension, is not a candidate.)

**reduce** — the one path *up*: $\operatorname{median}\{\,\text{value at each instance of a context axis}\,\}$, written to a coarser coordinate — a robust order-statistic (50% breakdown). `roll_up` is `reduce` over every reported `(option, dimension)`.

**support** — the count of instances backing a cell: the **evidence map**, emergent
from sparsity, read not stored (§5).

So a realization is a *pure function of a participant's slice* — reproducible and
explainable down to the contributing cells. The result is an ordered viable set. It
does not execute work or claim that every participant must have received the same
facts.

---

## 4. Write-down (integrity, as correctness)

The context tiers are an **integrity lattice**, coarse = high (the orchestrator),
fine = low (an untrusted device). The rule is **Biba ⋆-integrity, "no write up"**: a
writer may set its tier **or finer**, never coarser. The **only** path up is
`reduce` — a single trusted aggregation (Clark-Wilson transformation procedure)
whose median depends on no single low input, which is what *makes* the upgrade
legitimate. Enforced **structurally** via a [`Writer`] capability handle (write-up is
*unrepresentable*, not merely denied; `attenuate` only ever mints a stricter
handle). A device that could write `global` would corrupt every workspace's routing
— this is correctness, not security theater.

---

## 5. Dumb data, active orchestrator

The tensor stays **dumb**: facts at coordinates, sparse inheritance, the write rule.
No provenance, no quorum gate, no epoch stamps. Robustness is the orchestrator's
**active** job, and the data already gives it everything, because **sparsity *is*
the evidence map**: a cell with `support == 1` is *visibly* thin; one with fifty is
well-attested. The orchestrator reads `support`, finds the thin corners, and spends
its exploration budget there. It also drives the weights — the gas pedal of §2 — to
govern quota and cost. None of this is passive tensor behaviour; the tensor scores
what's in front of it, the orchestrator decides what to put there. `min_reporters`
is effectively **1**.

---

## 6. Distribution and persistence are adapters

The core carries **no clock, database, CRDT, transport, or network topology**. An
integration decides how facts are persisted, identified, exchanged, expired, merged,
and authorized. Those choices are not interchangeable implementation details: they
follow the use case.

- CareerVector can exchange collaborative state and observations directly between
  active peers, with Yjs and a durable service providing the convergence guarantees
  that product needs.
- JobCache can make an orchestrator the prominent edge, use a database for durable
  facts and leases, and send only the relevant slice to a worker.
- A standalone process can keep its tensor entirely in memory.

Sporewright may provide serialization and adapter interfaces so these integrations
exchange facts consistently. It does not declare one of them to be the universal
topology. Decision-equivalence is scoped to applying the algebra to the same
materialization; participants with different facts may correctly realize different
orders.

---

## 7. Facts vs lens, and auditability

The split that makes the whole thing legible:

- **Facts = the participant's current materialization.** Values and weights at
  coordinates may come from shared policy, received observations, or local
  circumstances. An adapter may version, persist, and audit them. The algebra does
  not mistake one participant's current view for universal truth.
- **Lens = the engine.** The axis set, the tier precedence, and the operator
  definitions (`fold`/`resolve`/`reduce`). This is *code, not data* — tiny, pure, and
  **byte-identical on every node**. You audit the facts; you trust the lens because
  it's small enough to read in one sitting and provably the same everywhere.

Auditability follows from addressable facts plus adapter-supplied identity: a
realization can list every contributing coordinate and its source. Replaying the same
materialization through the same lens must reproduce the same result.

---

## 8. Design choices

| Choice | Why |
|---|---|
| **everything is an axis; never compact** | context tiers, option, dimension are all axes; value and weight are two co-located layers. Sparsity makes the full grid free, so we never bundle axes (`instance`) or layers (a "role") — those were the footguns. |
| **value & weight co-located, both per-cell** | two factors that multiply where they share a coordinate. Weight is *usually* option-shared (one per dimension) and *occasionally* per-option (the gas pedal) — by sparsity, not by a structural cap. |
| **most-specific-wins fold**, not deep-merge | cells stay atomic; merge collisions (Kustomize/Helm's trap) can't happen. |
| **write-down via a capability handle** | makes write-up *unrepresentable*, not merely denied — no TOCTOU, no ambient authority. |
| **median as the one up-guard**, `min_reporters=1` | the only certified integrity upgrade; robustness beyond it is the orchestrator's call. |
| **store the facts, derive the decisions** | the object is the source of truth; every view is a slice, recomputed, never persisted — so no derived store can drift. |
| **byte-equivalence scoped to the *decision*** | the cores must agree on `resolve`/`reduce`/`fold` or the mesh is incoherent — that is the one load-bearing equivalence. The wire only **round-trips** (no byte-identity, no IEEE-bits, no content-addressing — the clock dedups). The dual-core tax is paid only where decisions are made. |

---

## 9. What's deliberately absent

- **provenance / who-wrote-what beyond the clock** — the write rule already bounds
  who *could* have written a cell; the orchestrator reads `support`, not history.
- **quorum / epoch / tranquility stamps** — orchestrator behaviour, not cell state.
- **confidentiality** (could device-B read device-A's slice?) — a *separate*,
  orthogonal lattice; not needed while a device only orders options it was granted.

---

## 10. Next: the products are thin backends

The engine is general — no product nouns. Products declare their axes and write
facts, then **route by `resolve` on the tensor** — no bespoke routing logic.

- A **dimension-heavy** product (e.g. routing work across heterogeneous methods):
  context tiers `source / device`, options the methods, dimensions
  `latency / reliability / cost` + a `reach` gate, values measured at the device and
  reduced up — a textbook `resolve` fit.
- A **precedence-heavy** product (e.g. an LLM provider/model cascade): override
  precedence (instance ≻ consumer ≻ stage) becomes context **tiers** (the fold:
  most-specific wins); each `(provider, model)` is an **option**; the ordered
  failover chain is exactly `resolve`'s queue (preference as values/weights), with
  any "top-N chain length" a thin app trim on the returned queue. Collaborative
  documents remain in the product's document substrate.

Same framework and operators, different participant materializations and axes.

**Options are self-announced, not provisioned, and may span tensors.** An option (a
device, a peer) enters a tensor by **registering itself at runtime** — a process that
started: a browser tab a user opened, a desktop app they installed, a container that
booted. The library does not stand it up; *how* its host is deployed is a separate
infrastructure concern entirely, and the number of options is not known in advance. One
option may also register in **more than one tensor instance at once** — the same browser
scraper can serve two products' tensors in parallel. Registration (runtime, self-driven)
and deployment (provisioning a host) are **orthogonal** — do not conflate them.

---

## 11. Discovery, liveness, and trust

Five additions from the design dialogue. None change the math; they name policies
the system leans on.

**`corroborate` — the option-axis twin of `reduce`.** `reduce` rolls a value's
*magnitude* up the levels (median, liar-resistant). `corroborate` rolls a fact's
*existence* up the levels (consensus, faker-resistant). A report is a plain cell —
`(level=device, instance=R, option=O, dimension=C)` = reporter **R** attests subject
**O** has capability **C**. The roll-up, for each `(O, C)`, counts independent
reporters `R ≠ O` (self-report excluded — "A,B,C about D louder than D about D"),
and materializes `O` into the higher level's options vector if the count clears a
quorum. It is `support` with self-exclusion + a quorum + a write-up. Population, not
evaluation.

**Boolean gate vs continuous score.** Work is atomic — you cannot half-send it — so
*population* is binary and *evaluation* is graded. A boolean gate answers **can I
use it at all?** (existence + capability + reachability; `can`/`can't`; the `±∞`/`0`
gate). A continuous score answers **how good, among the usable?** (`value·weight`).
`resolve` drops everything that fails a gate, then orders the rest by the score. The
two never mix; "boolean" is the *attestation/gate*, never the *value*.

**Live-scoping (an optional lifecycle policy).** An integration may retain a writer's
ephemeral cells only while that writer is present or leased. This can provide GC,
freshness, and a cost for disposable identities, but it is not part of the tensor
algebra and need not be enforced by an orchestrator.

**Topology follows the product.** Peer-to-peer exchange is important when active
CareerVector participants collaborate and shape one another's local view. A hub is
prominent when JobCache assigns and accounts for scraping work. Both can exchange
Sporewright facts; neither topology belongs in the core contract.

**Trust is a value, not a vote.** Device trust is a learned value about the
device-as-subject — `(option=device:A, dimension=trust)` — computed by the
orchestrator from A's corroboration/liveness record and read top-down to set its
**lease length, scrutiny, and work allocation** (a proven device earns an
ever-longer lease; the network grows quieter as it matures). The hard rule:
**trust must never weight a consensus attestation.** The moment a trusted vote
counts for more, trust becomes the attack target and the count-of-independent-live
property collapses. Consensus stays unweighted; trust may raise the **quorum the
orchestrator demands**, never the worth of a single vote. (`corroborate` is
implemented in both cores — `tensor.rs` / `tensor.ts`, guarded by the cross-core
decision-equivalence vectors; the trust *learning* that derives a device's trust value
from its corroboration + liveness record is still orchestrator policy, not yet in code.)
