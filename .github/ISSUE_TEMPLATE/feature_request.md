---
name: Feature request
about: Propose a capability or improvement
title: ""
labels: enhancement
assignees: ""
---

**The problem**
What are you trying to route / resolve / fail over that sporewright doesn't handle well today?

**Proposed solution**
What you'd like the engine, an operator (`fold`/`resolve`/`reduce`/`corroborate`), or the docs to do. If it touches a serialisable cell/wire type, note that both cores must stay decision-equivalent.

**Alternatives considered**
Other ways you've tried or thought about (a custom `Layer`, composing existing ones, handling it outside the engine).

**Scope**
- [ ] Pure composition of the existing API (a new recipe / example / doc)
- [ ] New engine capability
- [ ] Protocol / wire change (Rust + TS)
