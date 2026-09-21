# Routing with an addressed residual field

This guide explains how the generic model in [MODEL.md](MODEL.md) becomes a useful
router without turning product policy into hardcoded dispatch.

## One decision, many objectives

For every candidate option, a product supplies normalized lower-is-better dimensions
such as latency, financial cost, quality deficit, failure probability, or staleness.
Each address prefix may add a value residual and a weight residual. At address `a`:

```text
value(a, o, q)  = sum(value_residual(prefix, o, q))
weight(a, o, q) = sum(weight_residual(prefix, o, q))
score(a, o)     = sum(weight(a, o, q) * value(a, o, q))
```

Unknown is not impossible. An explicit boolean gate is the feasibility plane and is
combined by conjunction. A hard policy or budget limit is never approximated by a
large finite penalty.

## Why hierarchy matters

One observation rarely belongs either to the entire system or only to one attempt.
The address determines its useful statistical neighbourhood.

An LLM product might use:

```text
root -> workspace -> capability -> stage -> consumer -> optional instance
```

A provider outage can be observed broadly; a consumer-specific quality failure can
remain on that branch; and a stable configured instance can specialize recurring
work. Product identity is field metadata, not a fake `system` branch. Execution
environment is omitted when it cannot change LLM reachability or capability.

A job extraction product might use:

```text
corpus -> operation -> source -> binding -> task -> environment
```

The source branch learns that a portal often rejects lean containers, a binding
branch learns the cost of one adapter version, and a local browser can still veto an
offer because its current network cannot reach the target. A participant pulls only
work it can realize now. The orchestrator learns from the returned outcome instead of
pretending it can predict every browser, network, portal, and policy combination.

## Belief propagation rather than a neural network

The standard learner is a Gaussian hierarchical model with an independent residual
at each node. It borrows strength across related branches but updates only one
leaf-to-root path per observation. Cached likelihood messages are sufficient; no
training corpus scan or dense parameter matrix is required.

This deliberately takes useful ideas from neural networks and hierarchical bandits—
forward composition, an execution trace, uncertainty, exploration, shared priors,
and backward credit flow—without requiring a differentiable end-to-end model or a
large optimizer.

The routing receipt is the causal boundary. It freezes the exact field revision,
address, ranked alternatives, contributing prefixes, uncertainty, and selected
options. Feedback uses that receipt rather than reconstructing an address from mutable
runtime state.

## Curiosity is controlled expenditure

The router ranks by a lower confidence bound:

```text
decision_score = expected_score - temperature * sqrt(score_variance)
```

At temperature zero, the best posterior mean wins. A positive temperature lets
uncertain options compete so the system remains sensitive to change. A separate pure
planner admits additional executions only within explicit count and total-cost caps.
This makes the same setting useful for inexpensive distributed scrape probes and for
deliberately spending two or three LLM calls in a workspace that authorizes it.

For repeated distributed batches, lower confidence bounds are paired with a bounded
rotating exploration floor. `BatchExplorer` accrues token-bucket credit at
`temperature / (1 + temperature)` per live option and spends it round-robin. It keeps
only a cursor and fractional credit, so curiosity cannot starve behind deterministic
ties and its scheduling state does not grow with devices or completed tasks.

Exploration changes beliefs, not human policy. The application still decides which
result becomes user-visible.

## Scaling rules

The win case is thousands of unreliable actors, so the implementation has explicit
negative requirements:

- never enumerate the address Cartesian product;
- never scan unrelated workspaces, sources, tasks, actors, or history to decide;
- never broadcast every task to every actor;
- use actor-initiated pull, bounded leases, idempotent outcomes, and message dedup;
- replicate work selectively when uncertainty or validation warrants it;
- validate evidence before it shapes shared branches;
- store sufficient statistics and derived caches, not an unbounded replay scan in
  the decision path.

Retired task subtrees can be collapsed into exact Gaussian likelihood messages on
their parent. Products retain raw outcome events for audit or model replay, but the
hot field need not retain one permanent node per historical task. Clock-derived
discount factors reduce stale precision and reopen exploration without adding a
clock or background scan to a decision.

For address depth `D`, candidates `A`, dimensions `Q`, and path-local declarations
`C`, resolution is `O(C + D*A*Q + A log A)` and one outcome update is `O(D)`.
Structural work counters and cross-runtime golden vectors make that contract testable.

## What remains product-specific

The product chooses normalization, variances, expiry and drift policy, evidence
validation, hard budgets, candidate discovery, leases, execution, persistence, and
transport. Sporewright makes those choices compose and learn consistently; it does
not pretend that LLM quality and browser reachability have the same units or lifecycle.

See [LLM-REFERENCE.md](LLM-REFERENCE.md) for a complete host architecture using
semantic route addresses, legacy policy selectors, durable feedback, evidence aging,
bounded aggregation, and hard-budgeted curiosity.
