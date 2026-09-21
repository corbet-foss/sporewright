# Scope — mathematical core and product world

Sporewright owns the reusable mathematics and protocol shape. A product owns the
meaning of the world around it.

> **If a rule can be a deterministic transformation of explicit, domain-neutral
> values, it belongs in Sporewright. If it decides what those values mean or causes
> an external effect, it belongs in the product.**

## Sporewright owns

| Concern | Contract |
|---|---|
| Addressed sparse residual field | Exact-prefix indexing; additive path composition; no Cartesian materialization |
| Standard learner | Gaussian residual hierarchy, natural sufficient statistics, cached subtree messages |
| Decision | Multi-objective weighted score, uncertainty, absolute feasibility gates, deterministic ranking |
| Curiosity | Temperature-aware score and a pure execution planner bounded by explicit count and cost limits |
| Causal trace | Routing receipt containing the exact address, revision, alternatives, path contributions, and selected options |
| Integrity | A writer capability may write only its address depth or deeper |
| Portability | Deterministic serialization and Rust/TypeScript golden decision vectors |
| Scale evidence | Structural work counters and path-local update guarantees |

Sporewright may also contain pure helpers for normalization, budgets, freshness,
trust, or scheduling when they operate only on explicit inputs. It does not decide
which helper a product should use.

## The product owns

| Concern | Product decision |
|---|---|
| Vocabulary | Address layers, option classes, dimensions, units, and process/observation variances |
| Population | Candidate discovery, registration, liveness, capabilities, and current local context |
| Evidence semantics | Normalization, validation, deduplication, choosing discount factors, drift detection, and responsibility assignment |
| Hard authority | Money, token, execution, latency, privacy, and policy limits supplied to the pure planner |
| Effects | Pulling, leasing, executing, retrying, protesting, selecting a user-visible result, and failover |
| State and topology | Clocks, durable stores, snapshots, CRDTs, gossip, queues, P2P, authentication, and transport |
| Security | Enrollment, credentials, authorization, abuse controls, and observability |

An option is a candidate class for a decision: an execution archetype, method,
provider/model, or—only when the product intentionally chooses peers—a peer. Large,
changing participant populations normally sit behind stable option classes. Stable
classes let the field learn useful generalizations without creating an unbounded
option axis.

The host loop stays compact:

```text
declare address and candidates
  -> supply current policy and evidence
  -> resolve and budget curiosity
  -> offer work for pull
  -> execute outside the library
  -> validate and deduplicate the outcome
  -> update through the receipt's address
```

Sporewright does not prescribe a central or peer-to-peer topology. JobCache can
retain a central corpus and orchestrator; a collaborative application can exchange
the same field facts through P2P links. The algebra is identical.
