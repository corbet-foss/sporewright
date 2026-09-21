# sporewright documentation

| Guide | What it covers |
|---|---|
| [MODEL.md](MODEL.md) | **The contract.** The whole engine as one sparse tensor `T[level][instance][option][dimension]` — the object, the maths, the operators (`fold` / `resolve` / `reduce`), write-down/integrity, the dumb-data / active-orchestrator split, and the design choices. |
| [ROUTING-MODEL.md](ROUTING-MODEL.md) | **The why, re-readable.** How one `resolve` becomes a self-learning, multi-objective, resource-aware router: the three-role dimension vector (judgement / budget / gate), the Lagrangian-dual maths (with the honest implemented-vs-aspirational line), the device-fleet capability archetypes, eight worked stories, and a glossary. Read this to *get* the model; read `MODEL.md` for the exact contract. |
| [openssf-best-practices.md](openssf-best-practices.md) | OpenSSF Best Practices badge evidence. |

For the project overview see the [top-level README](../README.md); for the
mechanism-vs-semantics boundary, [SCOPE.md](../SCOPE.md).
