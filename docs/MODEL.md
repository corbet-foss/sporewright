# The Sporewright model

> **Sporewright provides resilient pull-based work distribution over
> heterogeneous, transient, and non-deterministic actors.**

Sporewright is a sparse hierarchical decision field. A product gives names and
meaning to an address, the candidate options, the quantities being optimized,
and the observations produced by execution. Sporewright provides the common
forward composition, uncertainty, routing receipt, and leaf-to-root learning
machinery.

The same model is used in a browser, a desktop process, and an orchestrator.
Small deployments do not use a reduced algorithm. Scale is a correctness
property of the model and its implementation.

“Pull-based” means the participating actor realizes offered work against its
current local field. An orchestrator may express demand and a preferred route,
but the participant can specialize or veto that route using facts that only it
can observe. An LLM endpoint is non-deterministic and changes over time just as
a browser's network and policies do; execution feedback from either travels back
through the same addressed field.

## 1. Addresses are paths

A product declares an ordered address schema

```text
L = (l0, l1, ..., lD-1)
```

and one decision is made at an address

```text
a = (l0=x0, l1=x1, ..., lk=xk),  k < D.
```

The field has one implicit root, written `{}`. It is not the first named layer.
Every prefix of `a` is a node on the realized path after that root. A broader address contributes
an inherited message; a longer address adds a specialization. Missing Cartesian
combinations do not exist as objects and consume no memory.

Examples of product-owned schemas are:

```text
LLM routing:
root -> workspace -> capability -> stage -> consumer -> optional instance

Job extraction:
corpus -> operation -> source -> binding -> task -> environment
```

An address segment exists when changing it selects another branch of the field.
It is not a database table, a transport hop, or an identity registry. Volatile
execution-attempt IDs do not belong in a durable address.

Product constants are not coordinates. If every field belongs to the same system,
`system=product-name` does not select a branch and belongs in field identity or
metadata, not the address. Likewise, an execution environment belongs in an LLM
address only when it changes what can execute; a browser-local BYOK call and a
server fallback with the same unconstrained model capability need no synthetic
environment segment.

## 2. Each node adds a residual

For option `o` and scalar quantity `q`, a node `p` on the address path carries a
local residual `delta[p,o,q]`. The realized latent quantity is

```text
Q(a,o,q) = sum(delta[p,o,q] for p in prefixes(a)).
```

This is the load-bearing algebra. In the toy path `A.B1.C11`, values `1`, `2`,
and `3` produce `Q = 6`. A lower residual can strengthen, neutralize, or invert
a broader message. There is no most-specific-wins cascade.

Products may define other field algebras, links, norms, and losses. Addition is
the implemented standard scalar algebra because it supports cheap sparse evaluation,
interpretable attribution, and leaf-to-root differentiation. Hard feasibility uses
conjunction: one explicit veto cannot be cancelled by a score.

## 3. Policy and belief are different planes

A scalar cell may contain two kinds of residual:

- `policy`: a declared top-down residual, such as a workspace's cost posture;
- `belief`: a learned residual with uncertainty, such as source-specific browser
  reliability.

Both contribute to the forward value, but observations update only belief.
Execution is not allowed to silently rewrite a human or operator policy. A hard
budget and a hard feasibility veto are constraints, not very large score weights.

For a multi-objective route, products normalize each raw measurement into a
dimension value. The standard expected cost is

```text
S(a,o) = sum(weight(a,o,q) * Q(a,o,q) for q in dimensions(o)).
```

Lower is better. Weights are themselves path-composed residual fields, so a
workspace default can be strengthened, revoked, or inverted at a finer address.
Products are responsible for choosing meaningful units and bounds.

## 4. Learned residuals form a Gaussian hierarchy

The standard belief field is a scalar Gaussian tree. It is written in residual
form to preserve the additive model:

```text
theta[root] = delta[root]
theta[child] = theta[parent] + delta[child]
delta[node] ~ Normal(policy_delta[node], process_variance[layer(node)])
observation ~ Normal(theta[address], observation_variance)
```

Thus `theta[address]` is exactly the sum of the path residuals. Observations make
the residuals statistically correlated, so Sporewright does not pretend that a
collection of independent EWMAs is an exact posterior.

Each node caches a Gaussian likelihood message from its whole observed subtree.
A new outcome changes local sufficient statistics and recomputes messages only
from that node to the root. A query propagates one cavity distribution from the
root down the requested path. In scalar natural parameters, independent evidence
combines by addition:

```text
precision = sum(precision_i)
information = sum(precision_i * mean_i)
```

If a node's subtree likelihood is `Normal(m, v)` and the residual variance from
its parent is `r`, its message to the parent is `Normal(m - policy_delta, v + r)`.
This is closed-form Gaussian belief propagation, not a neural-network training
loop and not a corpus-wide reducer.

Products can normalize Bernoulli reliability into log-odds, positive latency or
cost into log space, or supply another learner. The framework owns the address
traversal, cached message protocol, receipts, and complexity contract; the
product owns what an observation means.

## 5. Forward and backward are one causal cycle

```text
declared policy + cached belief messages
    -> resolve at one address
    -> choose under exploration and budget policy
    -> execute outside Sporewright
    -> normalize observed outcome
    -> attach it to the routing receipt
    -> update sufficient statistics at the observed node
    -> refresh messages on that node-to-root path
```

The forward pass records the implicit root and every contributing prefix. The receipt is the analogue
of an autodiff tape: feedback never guesses which branch produced a decision.
Bottom-up evidence changes shared ancestors, so sibling and unseen descendants
can learn. Top-down policy changes ancestors, so every descendant immediately
sees the new message. Neither direction is privileged.

## 6. Curiosity is uncertainty under a budget

Curiosity is not random traffic and is not hidden outside the field. A resolved
option has posterior expected score `S` and score variance `V`. The standard
lower-confidence decision score is

```text
curious_score = S - temperature * sqrt(V).
```

At temperature zero, routing is deterministic exploitation. Higher temperature
values make uncertain options competitive. A separate exploration budget says
how many options may be executed and how much additional resource may be spent.
For LLM routing this may intentionally authorize two or three model calls; for
JobCache it normally distributes different tasks across execution classes.

A confidence bound alone does not guarantee that a previously poor option is
ever sampled again: fixed evidence can keep it below the decision boundary.
Repeated batch schedulers can therefore use `BatchExplorer`, a constant-memory
token bucket with a round-robin cursor. It turns temperature into the bounded
rate `temperature / (1 + temperature)`, reserves one bootstrap probe when a
second option first appears, and rotates subsequent probes across live options.
Present-tense task vetoes may defer a reservation, but must not silently consume
it. This supplies explicit starvation resistance without a per-actor table or an
unbounded event scan.

Temperature and budget are addressed policy fields. A workspace can set them to
zero, or explicitly accept additional money and occasionally worse results so
the system continues to learn. A non-negotiable product or operator budget is a
hard cap outside the additive score.

The field receipt records posterior uncertainty and the selection policy. The
execution plan records expected cost, additional exploration cost, and the hard
budget that admitted it. Outcomes from exploratory executions update beliefs, not
declared policy.

## 7. The scaling contract

Let `D` be address depth, `A` the number of candidate options, `Q` their routed
dimensions, `C` the number of stored declarations in the exact-prefix buckets,
and `R` the number of ranked options. Sporewright requires:

```text
resolve:  O(C + D*A*Q + R log R)
feedback: O(D)
storage:  O(P)
```

where `P` is the number of explicitly declared or observed sparse parameters.
`D` counts named address layers; the implicit root adds one visited node but never
adds a Cartesian dimension.
No decision or feedback operation may be proportional to all workspaces, ads,
devices, historical outcomes, or possible address combinations.

`decide` discovers every candidate declared on the path and therefore includes
`C`. A host that already knows the currently executable option set uses
`decideAmong`/`decide_among`; candidate discovery then performs bounded indexed
lookups for that allow-list and never scans unrelated historical root options.

The implementation indexes exact address prefixes. Resolution reads only the
buckets on one path. Learning stores bounded sufficient statistics and replaces
one cached child message at each ancestor. Receipts expose visited node and cell
counts so the structural bound can be tested without relying on timing tests.

For persistence or gossip, raw evidence statistics are the mergeable facts and
posterior messages are derived caches. A product must deduplicate observations
before adding sufficient statistics. A retired non-root subtree can be compacted
into its exact Gaussian likelihood message on the parent. This preserves ancestor
and sibling posteriors while releasing descendant state. Because that likelihood
depends on the process variances inside the removed subtree, compaction freezes the
field's process-variance model; a changed model requires replaying uncompacted source
evidence into a new field revision.

The core also exposes a clock-free evidence discount. A product computes a factor
from its clock, source volatility, or detected drift; Sporewright scales natural
precision and information at the exact address and refreshes only that path. Old
observations can therefore become uncertain again without putting time semantics in
the generic engine.

## 8. Product boundary

Sporewright owns:

- ordered address schemas and exact-prefix sparse indexing;
- additive residual composition and hard feasibility conjunction;
- the standard Gaussian hierarchy and cached message passing;
- curiosity-aware resolution and deterministic routing receipts;
- capability-limited writers and deterministic serialization;
- structural work counters and Rust/TypeScript decision equivalence.

Products own:

- address, option, dimension, and unit vocabulary;
- normalization, observation noise, layer process variance, and hard budgets;
- candidate discovery, execution, retries, and user-visible result selection;
- clocks, evidence expiry, change detection, deduplication, persistence, CRDTs,
  gossip, authentication, and transport;
- deciding which outcomes are valid evidence.

JobCache may use a central orchestrator while CareerVector exchanges state among
collaborative participants. Topology does not change the field algebra.

## 9. Failure modes this contract forbids

- materializing the address Cartesian product;
- scanning all tensor cells for one decision;
- rescanning all descendants after one outcome;
- implementing specialization as replacement instead of addition;
- representing a hard constraint as a huge finite penalty;
- overwriting policy with execution feedback;
- treating an exploration temperature as an unbounded spending authority;
- reporting a learned decision without the receipt and observation that caused
  its update;
- allowing stale certainty to suppress all future curiosity forever.
