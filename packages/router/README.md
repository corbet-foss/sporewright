# router

Provider-fallback LLM execution and addressed-field resolution over
[sporewright](../sporewright).

`router` decides which `(provider, model)` option to try and in what order, then
executes attempts with timeout and abort handling, error classification, cooldown,
and critic acceptance. It ships Vercel AI SDK adapters and a host-supplied `KeyStore`
seam so routing never reads credentials or persistence itself.

## Routing address

LLM decisions use this complete branch:

```text
root -> workspace -> capability -> stage -> consumer -> optional instance
```

Declared stage, consumer, and instance chains become bounded additive `route`
residuals. Repeated options accumulate those contributions. No layer replaces a
broader layer and no large lexicographic weight is used. Normalized quality, latency,
reliability, and financial outcomes can be learned on the same branch without
overwriting route policy.

Hosts may attach human authority to a declaration with `routing: "auto"`,
`"prefer"`, `"fixed"`, or `"never"`. Preferences remain additive. Fixed order and
hard exclusions are evaluated as policy, outside learned belief, so feedback cannot
silently rewrite a person's choice. An explicit declaration at a more specific
address can specialize or revoke a broader gate.

The root is implicit (`{}`), not a named `system` layer. `resolveLlmRoute` separates
the semantic tensor address from persisted policy keys, allowing a host to replace an
old cascade without losing user configuration. The compatibility helper
`resolveCascadeTensor` instantiates this field and returns
the ordered fallback slots. Feedback-aware hosts use `resolveCascadeDecision`, retain
its receipt, persist the field evidence, and revise it after execution. Resolution
uses the field's bounded allow-list primitive, so retired or unavailable learned
models cannot leak into a live receipt or make work scale with global history.

## Execution

- **`runAttemptChain`** executes an already-resolved chain, applies a critic,
  classifies errors, and emits a trace per attempt.
- **`callWithChain`** adds AI SDK error classification, cooldown and dedup state,
  and structured exhaustion outcomes.
- **`resolveLlmRoute`** resolves a semantic address from one or more policy branches.
- **`resolveChain`** preserves the legacy same-address convenience API and binds keys.
- **`createModel`** and the adapter registry lazily load supported provider SDKs.
- **`modelFactory`** lets a host execute approved credentialless routes, such as a
  consented WebLLM model on the participant, without pretending that a local route
  owns an API key.
- **`KeyStore`** is the host seam for scoped chain configuration and credentials.

Injectable `onTrace`, `onAttempt`, `onLearningEvent`, `explorationBudget`, `logger`,
and `RouterState` seams keep telemetry, persistence, hard spending authority, and
process isolation under host control.

## Usage

```ts
import { callWithChain, resolveLlmRoute } from "router";

const chain = resolveLlmRoute(config, {
  capability: "structured-generation",
  stage: "evaluate",
  consumer: "column-evaluation",
  instance: "culture-fit",
  policyCapability: "chat",
  policy: [{ stageId: "evaluate", consumerId: "culture_fit" }],
});
const { result, providerId } = await callWithChain(
  chain,
  (model, signal) => generateText({ model, abortSignal: signal, maxRetries: 0 }),
  { label: "evaluate:fit" },
);
```

## License

LGPL-3.0-only WITH LGPL-3.0-linking-exception. See [LICENSE](./LICENSE),
[NOTICE](./NOTICE) and the repository-root [LICENSE.md](../../LICENSE.md).
