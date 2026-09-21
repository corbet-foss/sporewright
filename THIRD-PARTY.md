# Third-party components

No third-party source is vendored in this repository. All dependencies are
external packages that retain their own licenses; the library's LGPL grant
does not relicense them, and their notices must be preserved when
distributing them.

All listed dependencies are permissive (MIT / Apache-2.0) and
LGPL-compatible: linking them from an LGPL-3.0-only library — statically or
dynamically — imposes no additional duties beyond preserving their notices.
No GPL/AGPL implementation code is included. The exact locked inventory is
`Cargo.lock` (Rust) and `packages/*/bun.lock` (TypeScript).

## Rust (`crates/sporewright`)

| Crate | License |
| --- | --- |
| `base64 0.22` | MIT / Apache-2.0 |
| `serde 1` (+ `serde_core`, `serde_derive`) | MIT / Apache-2.0 |
| `serde_json 1` (+ `itoa`, `memchr`, `zmij`) | MIT / Apache-2.0 |
| `proc-macro2`, `quote`, `syn`, `unicode-ident` (build) | MIT / Apache-2.0 (+ Unicode-3.0 for `unicode-ident`) |

## TypeScript (`packages/sporewright`)

| Package | License |
| --- | --- |
| `zod 3` | MIT |
| `typescript`, `@types/bun` (dev) | Apache-2.0 / MIT |

## TypeScript (`packages/router`, LLM adopter)

| Package | License |
| --- | --- |
| Vercel AI SDK `ai 6` + `@ai-sdk/*` providers, `@openrouter/ai-sdk-provider` | Apache-2.0 |
| `zod 3` | MIT |

External model providers reached through those adapters have separate
account/service terms; the library's license grants no rights to a provider
account or subscription.

## Relocated own code (not third-party)

`packages/sporewright/src/trust.ts` documents a relocation of Julian Corbet's
own pairwise trust arithmetic from the jobcache `fact-tree.ts` adapter. Same
author, no third-party grant involved; the numbers are pinned by
`packages/sporewright/src/trust.test.ts`.
