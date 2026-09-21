// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * router — provider-fallback LLM execution + tensor-cascade resolution over a
 * sporewright tensor. The cascade decides which (provider, model) to try and in
 * what order; the engine executes each attempt with fallback, classification,
 * cooldown, and critic acceptance. Keys/persistence/tiers stay host-side behind
 * the KeyStore seam.
 *
 * Single flat root export — every consumer imports `from 'router'`.
 */

// --- local generic value types ------------------------------------------------
export type { ChainSlot, ChainValue, ProviderKeyUsage, ChainConfig } from './types';

// --- generic fallback engine --------------------------------------------------
export {
    runAttemptChain,
} from './attempt-chain';
export type {
    AttemptStatus,
    AttemptFailure,
    AttemptSkip,
    AttemptTrace,
    AttemptChainState,
    AttemptExecution,
    AttemptCritic,
    AttemptChainOptions,
    AttemptChainResult,
} from './attempt-chain';

// --- execution types + error classes ------------------------------------------
export {
    RateLimitError,
    ChainExhaustionError,
} from './errors';
export type {
    ProviderSlot,
    ProviderFailure,
    UsageInfo,
    ChainAttemptStatus,
    ChainAttempt,
    CallResult,
    RoutingReasonCode,
    RoutingExplanation,
    ChainAttemptObserver,
    LearningEventObserver,
    ExplorationReservation,
    ExplorationBudgetController,
    RouterLogger,
    CallWithChainOptions,
} from './errors';

export { learningEventForTrace } from './learning';
export type { LlmLearningEvent, LlmLearningObservation } from './learning';

// --- AI-SDK error classifiers -------------------------------------------------
export {
    extractStatusCode,
    describeError,
    isRateLimitError,
    isCorsError,
    isTransientError,
    isProviderLevelFailure,
    extractRetryAfter,
} from './classify';

// --- cooldown / dedup / rotation state ----------------------------------------
export {
    createRouterState,
    defaultRouterState,
    providerCooldownKey,
    shouldSuppressRepeat,
    activeCooldownSeconds,
    setProviderCooldown,
    clearProviderCooldownForTests,
} from './cooldown';
export type { RouterState } from './cooldown';

// --- model factory + lazy module caches ---------------------------------------
export {
    createModel,
    getAiModule,
    getAdapterModule,
} from './model-factory';

// --- the orchestrator ---------------------------------------------------------
export { callWithChain } from './execute';

// --- provider catalog + creator normalization ---------------------------------
export {
    PROVIDERS,
    CHAT_PROVIDER_IDS,
    isChatProvider,
    providersWithCapability,
    normalizeCreator,
} from './catalog';
export type { Capability, ProviderMeta } from './catalog';

// --- SDK-free adapter contract ------------------------------------------------
export {
    ProviderAdapter,
    LlmAdapter,
    PASSIVE_PROVIDERS,
} from './adapter-constants';
export type {
    DiscoveredModel,
    HealthCheckStrategy,
} from './adapter-constants';

// --- adapter registry (heavy @ai-sdk module) ----------------------------------
export {
    ADAPTERS,
    getAdapter,
    getLlmAdapter,
    isLlmAdapter,
} from './adapters';

// --- cascade address helpers --------------------------------------------------
export {
    stagePrefix,
    chainKeyL1,
    chainKeyL2,
    isStageChainKey,
    slotKey,
    flattenAllChainValues,
    usedSlotKeys,
    l2KeysForStage,
    instanceIdFromKey,
} from './cascade-keys';

// --- tensor cascade resolver --------------------------------------------------
// `cascadeConfig` is the thin-config declaration (the cascade tensor as data);
// `resolveCascadeTensor` instantiates + resolves it into the fallback chain.
export {
    cascadeConfig,
    compileCascadeField,
    cascadeReceipt,
    instantiateCascade,
    reviseCascadePreference,
    resolveCascadeDecision,
    resolveCascadeTensor,
    LLM_LAYERS,
    WORKSPACE,
    CAPABILITY,
    STAGE,
    CONSUMER,
    INSTANCE,
    type CascadeContext,
    type CascadeConfig,
    type CascadeDecision,
    type CascadeDeclaration,
    type CascadeFeedback,
    type CascadePolicySelector,
    type CascadeState,
    type LlmAddressLayer,
} from './cascade-tensor';

// --- chain resolution ---------------------------------------------------------
export { resolveChain, resolveLlmRoute, flattenCascade, routingPlanOf } from './resolve';
export type { LlmRouteRequest, LlmRoutingPlan, ResolvedProviderChain } from './resolve';

// --- the host-supplied key seam -----------------------------------------------
export type { KeyStore } from './keystore';
