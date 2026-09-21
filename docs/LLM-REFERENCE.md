# LLM routing reference host

This is the prime example for Sporewright's addressed field: resilient routing
over non-deterministic model providers whose quality, availability, latency, and
price change independently. It describes a complete host architecture without
making provider keys, persistence, billing, or product prompts part of the
generic library.

## Address and policy are separate inputs

The field root is implicit. A route is read at:

```text
root -> workspace -> capability -> stage -> consumer -> optional instance
```

- `workspace` owns one person's policy and learned branch.
- `capability` says what must be produced, such as `structured-generation`.
- `stage` locates the product workflow, such as `evaluate` or `tailor`.
- `consumer` names the stable operation, such as `column-evaluation`.
- `instance` is present only for a stable recurring configured object whose
  identity should specialize policy or learning, such as a score column or a
  document section. Request IDs and provider attempts are not instances.

Persisted configuration can predate this vocabulary. `resolveLlmRoute` therefore
accepts semantic coordinates and one or more policy selectors separately. A batch
evaluation can read the policy branches for several columns while producing one
decision at `evaluate.column-evaluation`; the column branches vote additively.
This is a migration seam, not a second routing model.

Representative routes:

| Capability | Stage | Consumer | Optional instance |
| --- | --- | --- | --- |
| structured-generation | extract | job-ad-extraction | — |
| structured-generation | enrich | salary-estimation | — |
| structured-generation | evaluate | column-evaluation | column ID for a focused call |
| structured-generation | tailor | section-generation | `document:section` |
| structured-generation | tailor | section-condensation | `document:section` |
| structured-generation | tailor | section-repair | `document:section` |
| structured-generation | tailor | document-refinement | `cv` or `cl` |
| structured-generation | tailor | document-critique | `cv` or `cl` |

There is no `system` coordinate when the product name is constant, and no
`environment` coordinate when browser and server calls have the same model
capability. A product should add either only when changing it selects a real
statistical branch.

## Objectives and observations

One option is a stable `(provider, model)` identifier. Credentials are resolved
only after the option is selected and never enter the field or receipt.

The reference objective uses normalized lower-is-better dimensions:

| Dimension | Example observation | Role |
| --- | --- | --- |
| route | bounded declared rank residual | top-down user policy |
| failure | 0 or 1 | reliability belief |
| quality-loss | 0 through 1 | critic or product completeness loss |
| latency-cost | bounded `log1p(latency)` | responsiveness belief |
| token-use | bounded `log1p(tokens)` | resource belief |

Weights live at the root and may be specialized by a host. Hard feasibility and
hard budgets are not score dimensions. A failed call produces reliability,
latency, and any measurable usage evidence. A successful call produces quality
evidence only when the host measured product quality; parse success alone leaves
quality unknown. Scheduling skips are not model evidence.

## Causal execution cycle

1. The host loads the sparse learned snapshot and current declared policy.
2. Sporewright resolves the semantic address among the route's currently declared
   executable options and returns ranked alternatives, uncertainty, contribution
   traces, and structural work counters. Learned retired models remain inherited
   evidence but cannot enter the decision receipt or its work bound.
3. The host binds credentials only for viable declared options.
4. `runAttemptChain` executes timeout, error-classification, critic, and fallback
   logic. The chain is operational behavior, not the learning state.
5. Every actual attempt is normalized into an outcome carrying a routing receipt.
6. The host validates and deduplicates the outcome, projects it at the receipt's
   exact address, and advances one monotonic workspace revision.
7. Cached Gaussian messages update along that address-to-root path. The next call
   immediately sees the new local posterior.

The tensor is not a log. An append-only outcome table is useful for idempotency,
audit, and future replay; the hot state is the compact sufficient-statistics
snapshot derived from it.

## Persistence and cross-workspace learning

A production host can keep five logical stores:

- one learned snapshot and projection cursor per workspace;
- idempotent append-only outcome envelopes;
- immutable workspace-to-root Gaussian messages by workspace revision;
- a bounded set of root aggregation buckets;
- hard budget accounts and short-lived reservations.

Policy is not copied into shared root evidence. A workspace message is computed
from learned belief only. Each workspace/option/dimension contribution is capped
before aggregation so one prolific or compromised workspace cannot dominate the
root merely by submitting more samples. Root projection selects only revisions
made visible by the workspace's monotonic state pointer, which makes chunked D1 or
SQLite writes interruption-safe.

The requester composes the aggregate root message minus its own included message
with its current local snapshot. This avoids double counting. Root reads aggregate
in SQL to one row per option and dimension; they do not transfer one row per
workspace into the application process.

Stronger hostile-network deployments should add workspace trust, sybil resistance,
and a robust bucket estimator. These are host policies because identity and trust
do not belong to the field algebra.

## Drift and evidence aging

Changing providers require uncertainty to reopen. The host applies an exponential
precision discount derived from its clock:

```text
factor = exp(-ln(2) * elapsed / half_life)
precision'   = factor * precision
information' = factor * information
```

The reference uses an exact factor for local snapshots and for root state at read
time, plus a bounded SQL approximation while aggregating inactive workspace
messages. Priors and declared route policy do not decay. Outcomes remain in the
audit log for its retention period, so a different half-life can be replayed into a
new field revision.

## Curiosity requires two independent permissions

Temperature changes the uncertainty-aware ordering:

```text
decision_score = expected_score - temperature * sqrt(score_variance)
```

It does not authorize money. Extra calls require a separate host budget controller:

1. `planExploration` chooses at most two additional high-information options.
2. The host atomically reserves a conservative per-call ceiling.
3. Only a successful reservation permits execution.
4. Measured usage settles the reservation; failures settle conservatively.
5. Expired reservations are reconciled from source rows by a scheduled job.

At temperature zero there is one deterministic exploitation path. A positive
temperature may make an uncertain option user-visible. A sufficiently high
temperature plus an explicit budget may execute up to three models. Exploratory
results can be discarded while their classified outcomes still improve the field.

For BYOK, a host may denominate the workspace budget in tokens. Platform-funded
calls normally use price-versioned currency micro-units. The generic controller
uses abstract units so the router cannot silently equate the two.

## Privacy boundary

The routing state may contain provider/model IDs, semantic addresses, normalized
outcomes, uncertainty, and revisions. It must never contain API keys, prompts,
generated text, CVs, job descriptions, or user profile data. Credentials remain in
the product's private key store and execution plane. A shared scraping corpus or
analytics database is a separate system even when it is packaged in the same app.

This boundary makes a receipt safe to persist and inspect while keeping the actual
LLM workload private.

## Complexity checklist

- Resolve one address; never enumerate workspaces or address products.
- Project one bounded event batch and update one leaf-to-root path.
- Aggregate root messages in storage, not one application statement per actor.
- Bound per-workspace root influence, feedback envelope size, alternatives, and
  observation dimensions.
- Keep raw history off the decision path.
- Use immutable messages plus a monotonic visibility pointer for crash recovery.
- Make every extra execution contingent on an atomic hard-budget reservation.
- Preserve the same algorithm at one workspace and at thousands; only batching,
  retention, and projection cadence should change.
