<p align="center">
  <img src="assets/logo.svg" alt="sporewright" width="380">
</p>

# sporewright

[![CI](https://github.com/corbet-foss/sporewright/actions/workflows/ci.yml/badge.svg)](https://github.com/corbet-foss/sporewright/actions/workflows/ci.yml)
[![License: LGPL-3.0-only WITH exception](https://img.shields.io/badge/license-LGPL--3.0--only-blue.svg)](LICENSE.md)
[![Docs](https://img.shields.io/badge/docs-the%20model-brightgreen.svg)](docs/MODEL.md)
[![Cores](https://img.shields.io/badge/cores-Rust%20%2B%20TypeScript-orange.svg)](#quickstart)

Sporewright is a framework for **resilient pull-based work distribution over
heterogeneous, transient, and non-deterministic actors**.

It provides a sparse addressed decision field shared by an orchestrator and its
participants. Broader addresses express inherited policy and belief; every longer
address adds a local residual. Outcomes flow back along the same realized path by
cached Gaussian message passing. This gives one mechanism for both directions:
top-down intent immediately shapes descendants, while bottom-up evidence informs
ancestors and related branches.

The same model can route scraping work across changing browser and container
environments or LLM work across changing providers and models. A product supplies
the vocabulary and measurements. Sporewright supplies the composition, uncertainty,
curiosity, receipts, and bounded-cost update protocol.

```text
policy + belief
      |
      v
resolve one address -> participant pulls/realizes work -> outcome
      ^                                                   |
      +--------- cached leaf-to-root messages <-----------+
```

The address Cartesian product is never materialized. Resolution touches only the
selected path and its candidate parameters; feedback touches only that path back to
the root. Work therefore does not grow with unrelated workspaces, tasks, actors, or
historical observations.

Read [the model](docs/MODEL.md) for the algebra and complexity contract and
[the scope](SCOPE.md) for the library/product boundary.

## Quickstart

Rust:

```rust
use sporewright::field::{address, Address, AddressedField, DecisionPolicy};

let mut field = AddressedField::new(["workspace", "capability", "stage", "consumer"]);
let local = address([
    ("workspace", "research"),
    ("capability", "structured-generation"),
    ("stage", "extract"),
    ("consumer", "job-ad-extraction"),
]);

let root = Address::new();
field.root_writer().set_prior(&root, "fast", "cost", 1.0).unwrap();
field.root_writer().set_prior(&root, "careful", "cost", 2.0).unwrap();
// Local evidence specializes the shared prior without replacing it.
field.writer("workspace").unwrap().observe(&local, "fast", "cost", 4.0, 0.1).unwrap();

let decision = field
    .decide(&local, DecisionPolicy { temperature: 0.0 })
    .unwrap();
assert_eq!(decision.alternatives[0].option, "careful");
```

TypeScript uses the decision-equivalent API:

```ts
import { address, AddressedField } from "sporewright";

const field = new AddressedField(["workspace", "capability", "stage", "consumer"]);
const local = address({
  workspace: "research",
  capability: "structured-generation",
  stage: "extract",
  consumer: "job-ad-extraction",
});

field.rootWriter().setPrior({}, "fast", "cost", 1);
field.rootWriter().setPrior({}, "careful", "cost", 2);
field.writer("workspace")!.observe(local, "fast", "cost", 4, 0.1);

const decision = field.decide(local);
if (typeof decision !== "string") decision.alternatives[0].option; // "careful"
```

## Build

```text
cargo test --workspace
cd packages/sporewright && bun test && bunx tsc --noEmit
```

The repository currently retains the pre-0.2 scoped-cascade modules while its two
adopters migrate. New integrations should use `field` and `explore`; the old
`tensor` surface will be removed before the next release.

## License and contributing

[LGPL-3.0-only WITH LGPL-3.0-linking-exception](LICENSE.md): use, modify, and
redistribute Sporewright under the LGPL with a static/dynamic linking exception
(no relinking or installation-information duties for combined works; library
modifications stay LGPL). See `LICENSE`, `LICENSES/` and `NOTICE`.

Contributions are welcome. Fork Sporewright, send a focused pull request, and
agree to the organization-wide [Individual Contributor License Agreement](https://github.com/corbet-foss/.github/blob/cla-v1.0/CLA.md).
Contributors retain copyright and grant the project the rights needed to license
accepted contributions coherently. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the equivalence and verification rules.
