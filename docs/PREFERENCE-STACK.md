# Addressed residual fields

This document previously described Sporewright as a most-specific-wins stack of
preference owners. That model has been replaced.

The current primitive is an ordered **address path**. Each prefix contributes an
additive residual to the realized value:

```text
Q(A.B.C, option, dimension)
  = delta(A) + delta(A.B) + delta(A.B.C)
```

Broader prefixes therefore provide defaults without being overwritten. A deeper
residual may strengthen, neutralize, or invert their contribution. Observations at a
leaf update cached Gaussian messages back toward the root, while root policy flows
forward to every descendant. The address is both the composition path and the causal
feedback path.

See [MODEL.md](MODEL.md) for the normative algebra, uncertainty model, curiosity
policy, complexity bounds, and product boundary.
