# Scope — the framework boundary

Sporewright is a framework, not a deployment architecture or a ready-made routing
dish. Its public promise is deliberately smaller than “all reusable mechanism”:

> **Sporewright defines how a participant represents possibilities, combines facts,
> realizes a locally viable order, explains that realization, and incorporates
> outcomes. Products define the world in which those operations run.**

The framework may offer optional reusable policies and adapters, but no product is
required to adopt one topology, database, transport, trust model, or vocabulary.

## The boundary

| Concern | Owner |
|---|---|
| Sparse coordinates, values, weights, constraints, and participant context | **core** |
| Deterministic composition, inheritance, feasibility, ordering, and explanation | **core** |
| Domain-neutral aggregation and update primitives | **core** when their semantics are explicit |
| Scheduling, budgets, exploration, trust, and normalization | **optional policy modules** |
| Serialization and fact-exchange contracts | **adapter interfaces** |
| Persistence, clocks, deletion, CRDTs, gossip, relays, and network topology | **integration** |
| Axes, dimensions, units, normalizers, constraints, and the meaning of success | **product** |
| Execution, retries with side effects, leases, queues, credentials, and portal policy | **product** |
| Measurements and accumulated facts | **participant/product data** |

“Generic” is not sufficient reason to move code into the core. A mechanism belongs
in the public library only when it preserves the participant-relative model, has a
small explicit contract, and is independently useful. Product adapters can be
published beside the core without becoming part of its algebra.

## What is shared

Participants need not hold identical tensors. They share:

- the coordinate schema used by an integration;
- deterministic operators and numeric semantics;
- fact/update identities that adapters can exchange;
- enough explanation data to reproduce a realization from the same materialization.

They may differ in local observations, cached remote facts, reachability, resources,
policy, and current task context. That difference is a feature: it is what lets the
same possibility field realize the right answer for each participant.

## What a product supplies

A product:

1. declares its axes, dimensions, units, and feasibility constraints;
2. combines shared inputs with local circumstances;
3. chooses persistence and exchange adapters appropriate to its topology;
4. consumes the ordered viable set and performs the side effect;
5. turns outcomes into observations and decides what to propagate.

CareerVector can therefore emphasize peer exchange and collaboration while JobCache
emphasizes an orchestrator. Both use the same framework principle without pretending
to be the same distributed system.
