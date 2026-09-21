<!-- SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception -->
# The routing model — one `resolve`, a myriad of purposes

This is the re-readable companion to [`MODEL.md`](MODEL.md). `MODEL.md` is the
**contract** — the object, the operators, write-down, sync. This document is the
**why**: it shows how the same tiny `resolve` becomes a self-learning,
multi-objective, resource-aware router for *any* kind of heterogeneous work, and it
states honestly which parts are implemented today and which are the milestone ahead.

If you read one section, read [the dimension vector](#3-the-dimension-vector-the-heart)
— that the cost of an option is **one entry in a long vector**, never the objective,
is the whole idea.

---

## 1. Why this exists

Real work has to be placed on real machines, and the machines are not alike. An LLM
call can go to a dozen providers, each with a different free-tier ceiling, latency,
and answer quality. A web page can be fetched by a 256 MB `curl` container, a
single-process headless Chromium, a residential-IP full browser, or a paid remote
renderer. An embedding job wants a GPU; a JSON scrape would *waste* one. Some targets
are hostile and demand a residential IP in the right country; some are trivial and
demand nothing.

The naive answer is a pile of `if` statements: *if the board is hostile use the
browser; if we're rate-limited switch providers; if it's a GPU task wait for the
window.* Every one of those `if`s is a routing decision dressed as control flow, and
they rot — a new provider, a new board, a new quota, and the pile grows another
branch.

sporewright replaces the pile with **one decision procedure over one object**. You
do not write routing logic; you write down *facts* about what each option is and what
each objective is worth, and `resolve` computes the order. New provider, new board,
new quota — that is new **data**, not new code. The router is the same three lines it
always was.

---

## 2. The primitive

Everything below is reads and writes on the sparse tensor from `MODEL.md`:

```
T[ level ][ instance ][ option ][ dim ]  →  { value, weight }
```

- **`resolve`** — compute the optimum. For each option, score it by
  `Σ value·weight` across the dim vector; **drop any option that has a `+∞` cell**;
  order ascending. The result is the ordered queue (a vector along the option axis).
- **`fold`** — most-specific-level-wins. When the same `(option, dim)` is set at
  several levels, the finest level that matches the cursor supplies the cell. (CSS
  cascade, git-config last-wins — not a deep merge.)

That pair lives in sporewright — the Rust crate `sporewright` and the
decision-equivalent TypeScript `@sporewright/core` — and it is **product-agnostic**.
It knows nothing about providers, browsers, GPUs, or job boards. It knows axes,
coordinates, two scalar layers, and how to add and sort.

---

## 3. The dimension vector — the heart

The `dim` axis is **long and heterogeneous**, and that is deliberate. It is not "the
cost of the option." It is a vector that plays **three different roles at once**, and
keeping them on one axis is what lets one `resolve` serve every purpose in this
document.

### (A) Judgement dims — *per-decision, mostly non-financial*

The objectives you are trading off for **this one task** — each stored as a **cost**:

| dim | stored value (always a cost — **lower is better**) |
|---|---|
| `financial` | money per call (one entry — **not** the objective) |
| `latency` | expected time to a result |
| `quality` | the quality **deficit** — distance from a perfect answer, so a *better* model carries a *lower* value |
| `reliability` | the observed **error rate**, so a *more* reliable option carries a *lower* value |
| `freshness` | **staleness** — age of the option's last measurement |
| … | any objective the product names, encoded as a cost |

**Direction — read this twice.** `resolve` *minimises*: it orders **ascending** by
`Σ value·weight` (§3 of `MODEL.md`). So **every value is a cost; lower is better.** An
objective where "more is better" — quality, reliability, freshness — is stored as its
**deficit** (quality-gap, error-rate, staleness) so the better option carries the lower
value and sorts to the front. (You *could* instead carry a negative weight, but
cost-encoding the value keeps the whole sum one "total regret" with positive weights —
the live cascade's `route` dim is exactly a lower-is-better cost.)

These costs are **scalarized by the weight vector** into the per-cell score
`Σ value·weight`. A weight is the **marginal rate of substitution** between objectives —
"I will accept 100 ms more latency to save one cent" *is* a ratio of two weights.
Choosing the weights chooses a **point on the Pareto frontier**. Crank `quality`'s
weight and the queue favours the lowest-deficit (best) answer regardless of price; crank
`financial`'s and it favours the cheapest option that still clears the gates.

Cost is dim number one of N. It has no special status. That is the sentence the whole
design turns on.

### (B) Budget / resource dims — *cumulative, time-windowed, shared across tasks*

Some quantities are not judgements about a single task; they are **shared, finite
pools that many tasks draw down together**:

| dim | meaning |
|---|---|
| `rate:<provider>` | requests-per-window allowed by a provider |
| `quota:<provider>` | daily/monthly token or request ceiling |
| `gpu_minutes` | GPU-seconds available per window |
| `egress` | bytes out per window |
| `concurrency` | simultaneous in-flight calls a provider tolerates |

These are **coupling constraints**: the decision for task A depends on what tasks B, C,
D have already consumed this window. You cannot price them per-task in isolation. The
machinery for coupling constraints is a **shadow price** — a Lagrangian multiplier
`λ` carried as the *weight* on the budget dim. As a resource saturates, its `λ` rises,
its contribution to every option's score rises, and load **self-rebalances** away from
the scarce resource without anyone writing a rebalancing rule. (Section 6 makes this
precise; its "Live vs. design" note states which half is built.)

### (C) Gate dims `priv:<cap>` — *boolean feasibility*

Some dims are not graded at all; they answer **can this option do the job, yes or
no?** They are encoded as a `+∞` **value** (never a weight — an infinite *weight* is a
silent no-op and cannot gate) — the convex-analysis **indicator function of the feasible
set**: `0` if the option is allowed, `+∞` if it is not. `resolve` drops any option
carrying a `+∞` value *before* it scores the survivors, so the queue contains only
feasible options, ordered by judgement.

Two kinds of gate matter:

- **Capability gates** — the option lacks a required capability (no browser, wrong
  IP type, no GPU window open). The capability vector in Section 5 is exactly this.
- **Trust is NOT a gate — it is a soft modulation** (settled 2026-06; supersedes the
  earlier `+∞` trust-gate). A low-trust device is **never excluded** from work. Trust
  enters routing only as (i) a soft **cost** on a `trust` judgement dim (§3A) — lower
  trust ranks a device lower, never *out* — and (ii) higher **verify-dispatch** intensity
  (the orchestrator sends more peers to re-check its results). A device must always be
  able to climb out of a hole **reasonably fast**; an *involuntary* malfunction (a glitch,
  a network blip, a page that genuinely changed) must not condemn it, and a *voluntary*
  liar is the rare case whose only tell is a **sustained corroborated-disagreement
  pattern** over time, never one event. The Sybil defense lives in **corroboration, not
  exclusion**: trust never *weights a corroboration vote* (Section 4 of `MODEL.md`),
  self-excluded quorum + aggregation drown a fake-identity swarm's lies when the honest
  majority is large, and the heavy verification a low-trust device attracts keeps its
  trust-cost high until it earns its way back. Capability is the only hard gate on a
  device; trust is a price, not a wall.

> One axis, three roles. Judgement dims set the order; budget dims bend the order as
> shared resources deplete; gate dims decide who is even allowed in the room. Because
> they share an axis, one `Σ value·weight` with one `+∞` drop expresses all three.

---

## 4. Levels — the per-product hierarchy of concerns

`level` is an **ordered hierarchy of concerns, coarse → fine**. As you descend the
levels you encode more about *the nature of this specific task*, and each level can
**clamp** the option set (write a `+∞` gate, narrow what is feasible) or **tilt** it
(adjust weights). The deepest matching cell wins (`fold`).

Crucially, **levels differ per product** — they are config, not engine. Two
archetypes that the engine serves today:

- **A workspace / matrix product** (humans collaborate, bring their own keys):
  `workspace (BYOK) ≺ stage ≺ consumer ≺ instance ≺ device`. The workspace level holds
  the user-supplied keys and preferences; finer levels narrow to a pipeline stage, a
  specific consumer of a capability, a single job/section, and finally the device that
  will run it.

- **A corpus / ingestion product** (no humans, no BYOK, privilege-scoped sessions):
  `session (privilege-scoped) ≺ source ≺ job ≺ device`. The session level carries the
  privilege grant (and explicitly **no** BYOK — the operator owns the keys); finer
  levels narrow to a source/board, a specific fetch job, and the device.

Same engine, different level vocabulary. **Each product is its own tensor instance**;
sporewright is the shared lens over both. A product's level list is a few strings it
declares at startup — the cascade resolver in the `router` package, for example,
declares its tiers as `l0 ≺ l1 ≺ l2` and writes the product's preferences into them.

---

## 5. The device fleet — capability vector and container archetypes

**The worker population is unbounded and weak-dominated — but it is NOT the option axis.**
The option axis (below) is the execution **archetypes** — *kinds* of capability — and, more
broadly, the services/providers being chosen among; it is **never a roster of individual
devices**. The devices that *instantiate* those archetypes are overwhelmingly weak client
instances (every running product tab does *something* — a plain fetch, and above all
**corroboration/verification** of others' results). That weak-but-numerous *worker* mass is
the engine of the trust loop — the verification capacity, and the honest majority whose
aggregation drowns a liar's votes — but knowing *which* devices are connected is an
**ops/observability** concern (telemetry → dashboard), not a routing input. A device declares
which archetype-capability it has (a hard gate when it lacks one); routing then chooses among
archetypes/providers, cheap-per-option, behind an unbounded churny fleet of weak instances.

For products that route *work onto a fleet of execution containers* (the corpus
archetype above is the driver), the option axis is populated with **container
archetypes**, and the gate dims are exactly the boolean feasibility tests of Section
3(C). The archetypes, smallest to heaviest:

| archetype | footprint | what it can do |
|---|---|---|
| `nano-curl` | 256 MB, no browser | plain HTTP GET + parse (JSON-LD, sitemaps) |
| `micro-html` | 384 MB | fetch + lightweight HTML parse, still no JS |
| `slim-browser` | single-process Chromium | renders JS, datacenter IP |
| `full-browser` | full Chromium, residential IP | renders JS behind anti-bot, residential egress |
| `remote-render` | Browserbase (paid) | hosted stealth rendering — last resort, costs money |
| `gpu-burst` | GPU window | embeddings / local inference during a GPU window |

The gate dims that decide which archetype is feasible for a given target:

| gate dim | values / meaning |
|---|---|
| `render_weight` | does the target need JS rendering at all? |
| `ip_type` | `datacenter < residential < mobile` (ascending stealth) |
| `antibot` | does the target deploy anti-bot defenses? |
| `geo` | required egress country (ISO code) |
| `gpu_window` | is a GPU window currently open? |

A target that demands residential IP + JS rendering writes `+∞` onto every option that
is datacenter-only or browser-less; `resolve` drops them and the queue is left with
exactly the archetypes that can do the job, ordered by cost and latency. A target that
needs none of this leaves the cheap `nano-curl` sink ungated and at the front of the
queue because its `financial` and `latency` are lowest.

---

## 6. The math — multi-objective, resource-constrained assignment

Routing here is a textbook shape: **assign a set of tasks to a set of options to
minimise total weighted cost, subject to per-resource budget ceilings and per-option
feasibility gates.** Written out:

> minimise   `Σ_tasks Σ_dims value(option, dim) · w(dim)`
> subject to `Σ_tasks usage(option, resource) ≤ cap(resource)`   for each budget dim
> and        `option is gate-feasible`                            (no `+∞` cell)

The budget constraints **couple** the tasks (they share the pools), which is what
makes a naive per-task greedy choice wrong. The standard, beautiful way to decouple
them is **Lagrangian dual decomposition**:

1. **Relax** each budget constraint into the objective with a price `λ_resource ≥ 0`.
   The objective gains a term `λ_resource · (usage − cap)`. In tensor terms, `λ` is
   simply the **weight on that budget dim** — the shadow price of the resource.

2. **The primal step** — with the prices held fixed, the problem **separates per
   task**: each task independently runs `resolve` at the current weights (= prices) and
   gates, and picks its best feasible option. This is exactly `resolve(cursor)`. No
   task needs to know about any other task; the price is the only channel between them.

3. **The dual step (dual ascent)** — the orchestrator's slow tick measures actual
   usage and nudges each price toward clearing its resource:

   ```
   λ ← [ λ + α · (usage − cap) ]₊
   ```

   When a resource is over-consumed (`usage > cap`), its price rises; when it is idle,
   its price falls back toward zero. A `429 Too Many Requests` from a provider is a
   direct observation that `usage > cap` for its `rate` dim, so a 429 **raises that
   provider's rate price** — and the next round of `resolve` calls naturally route
   around it.

   Four things this update must get right (each is a place an earlier draft was wrong):
   - **Prices are per resource *pool*** `λ_r`, where a pool is a *cap's owner* — a
     provider account's shared quota, a target's per-IP rate — **not** per option. A
     single fleet-wide price over-couples unrelated providers; a per-option price
     under-prices a shared account.
   - **It is *online* projected subgradient ascent** against stale, drifting aggregates
     (`[·]₊` enforces `λ≥0`). Quotas refill, so there is no static optimum — target
     **dynamic regret** with a small *constant/floored* step, not the textbook
     diminishing step.
   - **The feeder is additive `Σ usage`** — distinct from `reduce` (a median) and
     `corroborate` (a count). The core provides it as **`reduce_sum`** (`tensor.rs` /
     `tensor.ts`), the additive value-path up; `Budget::aggregate_usage` rolls each
     window's per-option draws up with it (in a canonical, cross-core-identical order)
     and feeds the totals to the price tick.
   - **The price loop is single-writer.** A price is a non-idempotent accumulator and
     the substrate is last-write-wins (no clock, no CRDT), which would clobber concurrent
     increments — so the price update is a **centralised hub** step on the slow tick
     (devices only *read* prices). Only the **primal** (per-task `resolve`) is distributed.

4. **There is no clean convergence.** The primal is integer, so the relaxation yields
   only an **LP lower bound with a non-zero integrality gap**; per-task argmins can even
   *violate* a cap when tasks tie on the cheapest pool. This is a **price-coordinated
   heuristic that needs a feasibility-repair step**, not a solver reaching a KKT point —
   the earlier "convergence = KKT / complementary slackness" claim was too strong.

> **A known trap in the headline story.** The static `Σ_task usage ≤ cap` models a
> *cumulative (daily) quota*, **not** a per-rolling-window *rate*. For a purely
> rate-limited free tier the daily quota is *slack*, so its price stays zero and
> "reallocate before the wall" **does not fire** for the rate case. Rate and concurrency
> are handled by a **second, edge-local, sub-second token-bucket admission gate** —
> `TokenBucket` in `budget.rs`/`budget.ts`, distinct from the central route-time `+∞`
> gate — with the slow price only shaping the steady state.

The elegance is now concrete: the per-task choice and the global resource balance are the
**same `Σ value·weight`** at two timescales — tasks resolve fast against frozen prices, a
single hub adjusts prices slowly against measured usage. The primitives that do this — the
additive usage feeder, the per-pool price, the single-writer tick, the local admission
gate — are **built** (`budget.rs` / `budget.ts`, see below); none of it was a new engine,
just data, one accumulator, and a slow tick. What remains is *wiring* them into a live
orchestrator loop, plus the feasibility-repair the integer primal needs.

### Live vs. design — stated honestly

The gap is a real milestone, not a flourish:

- **Live today.** The **gated weighted-sum ranker** is real and shipping. `resolve`
  (`crates/sporewright/src/tensor.rs`) computes `Σ value·weight`, drops `+∞`-**value**
  gates, and orders the queue; the `+∞` capability and trust gates are fully wired; and
  two consumers already score **two or more cost dims** — so the *multi-objective
  scalarization itself is implemented*, not just a single static preference. Failover
  walks the resolved queue.

  One honest caveat remains on the live ranker, and one fix has shipped. *Per default*
  a consumer's score still sums **incommensurable** scales, so at the scrape router's
  seeded weights one dim (`financial`) dominates the others by ~100× and the order is
  effectively **lexicographic** — which is *deliberate* there (prefer free). *The
  capability to fix it now exists*: the **per-dimension normalizer**
  (`crates/sporewright/src/normalize.rs` / `packages/sporewright/src/normalize.ts`) gives
  each dim a numéraire, so a consumer that wants a balanced multi-objective trade-off
  divides each cost by its scale (`Norm::effective_weight` folds `1/scale` into the
  weight via the identity `(v/scale)·w == v·(w/scale)`) and the weights become honest
  marginal-rates-of-substitution. The engine stays **pure** — `resolve` is untouched, still
  `Σ value·weight`; normalization is a thin write-side opt-in (the scales are product
  data). This turns the §3 Pareto reading from aspirational into literal *for any
  consumer that opts in* — the scrape router keeps its intended financial-first order,
  unchanged.

- **Live today — the budget/price *primitives*.** The whole budget layer is **built and
  cross-core golden-pinned** in `crates/sporewright/src/budget.rs` +
  `packages/sporewright/src/budget.ts` (`tests/budget-vectors/`): `Budget{cap, λ}` with the
  option→pool map; usage rides as the **value** on `budget:<pool>` and the price `λ` as the
  **shared weight**, so `resolve`'s `Σ value·weight` already contributes `λ·usage` (the
  relaxed term, no engine change); `Budget::tick` is the single-writer
  projected-subgradient step `λ ← max(0, λ + α·(usage−cap))` (constant step, clamped finite
  at `MAX_LAMBDA`); `aggregate_usage` feeds it via `reduce_sum`; and `TokenBucket` is the
  separate edge-local admission gate for the rate-limit trap. The
  `saturation_simulation_load_sheds_to_the_idle_pool` test demonstrates the self-rebalance
  end to end.

- **Not yet wired — the product loop.** No *live consumer* drives that loop yet: the
  orchestrator's slow tick does not yet aggregate real usage, advance `λ`, and publish
  prices, so today a free option (`financial = 0`) is still ranked first **unconditionally**
  and abandoned only by **reactive failover after it errors**. The mechanism is built; the
  remaining step is wiring it into the running orchestrator (plus the feasibility-repair for
  the integer primal). That is a product/deployment milestone, not a missing primitive.

The honest one-liner: **the gates, the multi-cost-dim ranker, the per-dimension normalizer,
and the budget/price primitives (per-pool `λ`, the single-writer tick, the `reduce_sum`
feeder, the token-bucket gate) are all built and golden-pinned; what is not yet wired is a
live orchestrator loop that drives the price tick — the model was designed so that step adds
data and a slow tick, not a new engine.**

---

## 7. The library boundary

What is engine, and what is product:

- **sporewright (the agnostic engine)** holds: the tensor (cells, value/weight,
  `fold`/`resolve`/`reduce`/`corroborate`/`support`), capability routing (the `+∞`
  gates), the AI-provider connection / cascade layer (the `router` package — adapters,
  failover, classification, the cascade `resolve`), and a **store-agnostic broker** /
  persistence port. It is product-blind and open-source destined.

- **The products hold only config + product-specific code.** A product supplies its
  *levels*, its *options*, its *dims*, and its *providers* (all data the engine routes
  over), plus the code that is genuinely its own:

  - the **corpus / ingestion** product brings its source adapters and its fact-tree /
    reconciliation logic;
  - the **workspace / matrix** product brings its workspace model, its CRDT document
    trees, and its UI.

The seam is **facts in → decision out → product acts**. The engine decides; the
product does the side-effecting work. If you find routing logic accreting in a product,
it almost certainly belongs in the engine as data over `resolve` (see `SCOPE.md`).

---

## 8. Stories — the same `resolve`, a myriad of purposes

Each vignette is the *same* `Σ value·weight` drop-the-`+∞` mechanism serving a
different purpose. That breadth is the design earning its keep.

### Story 1 — A free tier self-shards off the 429 wall *(core primitives live; the orchestrator loop that drives them is the remaining product step)*

A provider's `financial` dim is `0`, so on judgement alone `resolve` says **always use
it** — it is free, it leads every queue. Early in the window it does. But every call
draws down its shared `rate` budget. As usage approaches the cap, the dual step raises
`λ_rate` for that provider; its score climbs; it slides down the queue. When it 429s,
that is a hard observation of `usage > cap` and the price jumps. Traffic spills onto
the *next* free tier, whose own price is still low — and the load spreads itself across
the free tiers, riding each one up to its wall and no further. No "if 429 then rotate"
branch exists; the rotation is the shadow price doing its job.

> **The primitives are live; the wiring is the remaining step.** `Budget::tick` /
> `publish_prices` / `aggregate_usage` and the `TokenBucket` all ship and are golden-pinned
> (the `saturation_simulation_load_sheds_to_the_idle_pool` test walks this exact shed). What
> is **not** yet wired is the product-side orchestrator loop that aggregates real usage and
> advances `λ` each tick — so *today* a free tier is still ridden *to* its wall by reactive
> failover, not pre-empted. Mind the §6 trap when wiring it: the daily-quota price is slack
> for a purely *rate*-limited tier, so the proactive reshard must ride the `TokenBucket`
> admission gate, not the slow price.

### Story 2 — The GPU-burst window *(aspirational for the opportunity-cost price; the gate is live today)*

Embedding tasks need `gpu-burst`, which is only feasible while a GPU window is open:
outside the window, `gpu_window` writes `+∞` and `resolve` makes the embed tasks
**pending** with `available_after = next window`. They wait. When the window opens the
gate clears and they **drain in a few GPU-minutes**. The opportunity-cost price keeps
the GPU honest: a plain `curl` scrape never routes to the GPU box, because the GPU's
`gpu_minutes` price makes it far more expensive than a `nano-curl` sink for work that
does not need it. The scarce GPU is spent only on tasks that actually need a GPU.

### Story 3 — A hostile board gates down to residential + full Chromium *(live today)*

A geofenced, anti-bot board (think a LinkedIn-class target) writes gates: `antibot`
and `render_weight` demand JS rendering behind real anti-bot defenses, and `ip_type`
demands residential. Every datacenter `curl` sink and browser-less archetype gets a
`+∞` and is **dropped**. The queue is left with `full-browser` (residential) at the
front — and, behind it, the paid `remote-render` as a costly fallback. The hostile
board gets exactly the heavy, expensive option it requires, and nothing cheaper is even
considered, because nothing cheaper is feasible.

### Story 4 — A trivial JSON scrape routes to the cheapest sink *(live today)*

A friendly board that emits clean schema.org JSON-LD (a Greenhouse-class page) needs no
rendering, no special IP, no GPU. It writes **no gates**. So every archetype is
feasible — and `resolve` orders them by `financial` + `latency`, which puts
`nano-curl` (256 MB, cheapest, fastest) at the front. The GPU box and the fat
residential browser are *technically feasible* but sort to the back on cost and never
get used. The cheap path wins precisely because nothing forced the expensive one.

### Story 5 — The same tensor, a different instance *(live today; this is the product-agnostic proof)*

Drop the device fleet entirely. An LLM cascade routes `(provider, model)` options on
exactly the same engine: the judgement dims are `quality` + `financial`, the budget dim
is the provider's `rate`/`quota`, the levels are the override tiers (stage ≺ consumer ≺
instance). `resolve` returns the ordered fallback chain — best-quality affordable model
first, cheaper or alternate models behind it, a finer-tier override always sorting ahead
of a coarser default. This is a **different tensor instance with different axes**, and
it is the same `Σ value·weight` over the same code. The engine never learned the word
"provider." (This is what `router/cascade-tensor.ts` does today.)

### Story 6 — A brand-new device is priced high, not gated out *(gate live; trust learning aspirational)*

A device the orchestrator has never seen reports in. Its `trust` value is at the
cold-start prior — low. That makes its `trust` **cost** high, so on high-value work it
ranks *behind* proven devices (never *dropped*) and attracts heavy verify-dispatch — it
is naturally left the low-value, easily-corroborated work (fetching a public page that
several other devices can independently confirm). By doing that work and being
corroborated, it **earns** trust (Section 11 of `MODEL.md`: trust is a value computed
from its corroboration and liveness record), its `trust` cost falls, and better work
opens up. This is the Sybil defense *without a wall*: a fake identity is never locked
out, but it cannot leap straight to high-value work either — its lies are out-voted by
the honest majority and its trust-cost stays high until it does verifiable low-value
work under a live, renewed lease.

### Story 7 — Geo routes to the right egress *(live today)*

A US job board sets `geo = US`. Every option whose egress is not US gets `+∞`; the
queue is left with US-egress options. A Swiss board sets `geo = CH` and *prefers* (via
weight, not gate) a CH or EU residential egress — feasible options exist elsewhere but
the geo-appropriate one scores best and leads. Same `resolve`, the country is just
another gate-or-weight on the dim vector.

### Story 8 — A provider quality regression, no code change *(the measurement-fold `reduce` is live; the orchestrator loop that folds measured quality into cells is the remaining product step)*

A provider quietly gets worse — its answers degrade. The product *measures* answer
quality and folds the measurement into the provider's `quality` value (`reduce`, the
median up the instances). That single value drop lowers the provider's score; it slides
down every queue that weights `quality`; traffic shifts to better providers. **No code
changed** — no ranked list was re-edited, no `if provider == X` was added. A measured
fact moved, and the order followed. This is the same mechanism as Story 1, but on a
judgement dim instead of a budget dim.

---

## 9. Glossary

| term | meaning |
|---|---|
| **level** | An ordered tier in a product's coarse→fine hierarchy of concerns. Differs per product. Descending encodes more about the task and can clamp the option set. |
| **instance** | The coordinate on a level axis — *which* workspace, *which* job, *which* device. Each product is its own tensor **instance** (a separate populated tensor over the shared engine). |
| **option** | Any routable candidate on the option axis: a device, an API provider, a render SaaS, a p2p peer. A bare identity; the judgements live in its `(option, dim)` cells. |
| **dim** | A coordinate on the heterogeneous judgement axis, in one of three roles below. The cost of an option is **one dim among many**. |
| **dim — judgement** | (A) Per-decision objective, mostly non-financial: `financial`, `latency`, `quality`, `reliability`, `freshness`. Scalarized by weights into the score. |
| **dim — budget** | (B) A cumulative, time-windowed, **shared** resource pool: `rate`, `quota`, `gpu_minutes`, `egress`, `concurrency`. A coupling constraint, shadow-priced by `λ`. |
| **dim — gate** | (C) A boolean feasibility test `priv:<cap>`, encoded `+∞` (infeasible) / `0` (feasible) — the indicator of the feasible set. Capabilities are gates; trust is NOT a gate — it is a soft cost + verify-dispatch (see §3C). |
| **value** | The learned/declared truth in a cell on a `(option, dim)`. `+∞` is a gate. |
| **weight** | How much a value matters. A judgement weight is a marginal rate of substitution (a point on the Pareto frontier); a budget weight is a shadow price `λ`. |
| **resolve** | Compute the optimum: `Σ value·weight` per option, drop any option with a `+∞` cell, order ascending → the queue. |
| **fold** | Most-specific-level-wins: when an `(option, dim)` is set at several levels, the finest level matching the cursor supplies the cell. |
| **gate** | A boolean feasibility constraint expressed as a `+∞` cell that `resolve` drops before scoring. The convex-analysis indicator of the feasible set. |
| **shadow price** | The Lagrangian multiplier `λ` on a budget dim — the marginal value of one more unit of a scarce resource. Rises as the resource saturates (and on a `429`), falls when idle; carried as the budget dim's weight. **Implemented** as the shared weight on `budget:<pool>` in `budget.rs`/`budget.ts`, advanced by `Budget::tick`. |

---

## See also

- [`MODEL.md`](MODEL.md) — the engine contract (object, operators, write-down, sync,
  trust/corroborate).
- [`../SCOPE.md`](../SCOPE.md) — the mechanism-vs-semantics boundary: what is engine,
  what is product config + data.
- [`../packages/router/README.md`](../packages/router/README.md) — the live cascade
  `resolve` + failover layer described in Story 5.
