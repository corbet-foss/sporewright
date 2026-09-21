---
name: Bug report
about: Something doesn't work the way the docs say it should
title: ""
labels: bug
assignees: ""
---

**What happened**
A clear description of the bug.

**Which core**
- [ ] Rust crate (`sporewright`)
- [ ] TypeScript package (`sporewright`)

**To reproduce**
A minimal example — the smallest `Tensor` (the cells you write), the cursor you read at, and the operator you call (`fold` / `resolve` / `reduce` / `corroborate`) that shows the problem. Paste code, not screenshots.

```rust
// or ```ts
```

**Expected vs actual**
What you expected the operator to return for that cursor (the queue, the folded value, the reduced/corroborated cell), and what it returned instead.

**Environment**
- sporewright version / commit:
- `rustc --version` (Rust) or `bun --version` (TS):
- OS:
