# router

Provider-fallback LLM execution and tensor-cascade resolution over a
[sporewright](../sporewright) tensor.

`router` is the generic LLM routing layer: it decides **which** `(provider, model)`
option to try and in **what order** (the cascade, resolved over the override-tier
tensor), then **executes** each attempt with timeout/abort handling, error
classification, rate-limit cooldown, and critic acceptance — falling through to the
next option on failure. It ships the Vercel AI SDK adapter registry, the provider
catalog, and a host-supplied `KeyStore` seam so the engine never reads keys, tiers,
or persistence itself.

## What it is

- **`runAttemptChain`** — the dependency-free fallback engine. Iterates an
  already-resolved chain of attempts, executes each under a per-attempt timeout /
  parent-abort race, applies a critic, classifies thrown errors, and emits a trace
  per attempt.
- **`callWithChain`** — the LLM orchestrator over `runAttemptChain`. Adds AI-SDK
  error classification (429 / retry-after / CORS / provider-level), per-router
  cooldown + dedup state, and `RateLimitError` / `ChainExhaustionError` outcomes.
- **`resolveCascadeTensor` / `resolveChain`** — the cascade: the fallback order
  *emerges* from a `resolve` over the override-tier sporewright tensor (l0 stage
  ≺ l1 consumer ≺ l2 instance). Options are `(provider, model)`; the objective is
  a single declared `route` preference; a per-tier precedence weight makes a finer
  override always sort ahead.
- **`createModel` + the adapter registry** — the Vercel AI SDK wrappers for every
  supported provider, lazy-loaded behind a dynamic import so consumers that never
  make LLM calls don't pay the SDK parse cost.
- **`KeyStore`** — the seam. The host supplies `get(scopeId): Promise<ChainConfig>`;
  the router supplies the mechanism. It never reads env vars, persistence, tiers,
  or grants.

## Injectable seams (host-supplied, all defaulted)

- **`onTrace` / `onAttempt`** — observers the host wires for telemetry. The engine
  emits every `AttemptTrace`; the router itself writes nothing.
- **`logger`** — an injected `{ log, warn }` (default: no-op). The router carries
  no build-tool assumption (no `import.meta.env`), so it compiles under bun,
  wrangler/esbuild, and Vite alike.
- **`RouterState`** — the cooldown / dedup / balanced-key-rotation state. Defaults
  to a module singleton (process-scoped behaviour) but can be a per-router instance.

## Usage

```ts
import { resolveChain, callWithChain, createModel } from 'router';

const chain = resolveChain(config, 'evaluate', 'fit', 'chat'); // ProviderSlot[]
const { result, providerId } = await callWithChain(chain, (model, signal) =>
  generateText({ model, abortSignal: signal, maxRetries: 0, /* … */ }),
  { label: 'evaluate:fit' },
);
```

`router` `file:`-depends on the sibling sporewright tensor and uses its
`Tensor` / `Writer` / `resolve` public API. The dependency is one-way: `router`
depends on the tensor, never the reverse.

## License

LGPL-3.0-only WITH LGPL-3.0-linking-exception. See [LICENSE](./LICENSE),
[NOTICE](./NOTICE) and the repository-root [LICENSE.md](../../LICENSE.md).
