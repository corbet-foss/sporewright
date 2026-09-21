// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Chain resolution — "who do I call, in what order?"
 *
 * `resolveChain` turns a {@link ChainConfig} into an ordered {@link ProviderSlot}[]
 * with API keys for a specific consumer call. The cascade ORDER emerges from a
 * `resolve` over the override-tier tensor (see `cascade-tensor.ts`): every L2
 * (instance) option, then every L1 (consumer), then every L0 (stage), each tier
 * by declared preference. A `(provider, model)` that recurs across tiers folds to
 * its finest tier and appears once, at its best position.
 *
 * Multi-key fan-out per provider is governed by `keyUsage`: `'ordered'` keeps the
 * entered order, `'balanced'` rotates the leading key per call via the
 * {@link RouterState} cursor (default: the module singleton, so rotation is
 * process-scoped exactly as before).
 */

import { resolveCascadeTensor } from './cascade-tensor';
import type { ChainSlot, ChainValue, ChainConfig, ProviderKeyUsage } from './types';
import type { ProviderSlot } from './errors';
import { type RouterState, defaultRouterState } from './cooldown';

export type { ChainConfig } from './types';

function orderedKeysForProvider(
    state: RouterState,
    providerId: string,
    keys: readonly string[],
    usage: ProviderKeyUsage | undefined,
): string[] {
    const clean = keys.map(k => k.trim()).filter(Boolean);
    if (clean.length <= 1 || usage !== 'balanced') return clean;

    const cursor = state.balancedKeyCursor.get(providerId) ?? 0;
    state.balancedKeyCursor.set(providerId, (cursor + 1) % clean.length);
    const start = cursor % clean.length;
    return [...clean.slice(start), ...clean.slice(0, start)];
}

/**
 * Resolve the cascade into the ordered ChainSlot[] for a specific context — the
 * LLM provider/model fallback chain. Thin alias over {@link resolveCascadeTensor}
 * kept for call-site stability.
 */
export function flattenCascade(
    chains: Record<string, ChainValue>,
    stageId: string,
    consumerId: string,
    capability: string,
    instanceId?: string,
    l2Override?: ChainSlot[],
): ChainSlot[] {
    return resolveCascadeTensor(chains, stageId, consumerId, capability, instanceId, l2Override);
}

/**
 * Resolve a provider chain for a specific consumer call.
 *
 * Chains are complete from the host (seeds materialized on first access).
 * Resolution walks L2 → L1 → L0, reading the requested capability at each level.
 *
 * @param config      - Cached key material (chains already seeded)
 * @param stageId     - Pipeline stage ('extract', 'evaluate', 'tailor', etc.)
 * @param consumerId  - Consumer ID ('analysis', 'timelessness', 'text', etc.)
 * @param capability  - Capability to resolve ('chat', 'route', 'scrape', etc.)
 * @param instanceId  - Instance ID for L2 override
 * @param l2Override  - Job-scoped chain override (flat, not capability-keyed)
 * @param state       - Router state holding the balanced-key cursor (default singleton)
 */
export function resolveChain(
    config: ChainConfig,
    stageId: string,
    consumerId: string,
    capability: string,
    instanceId?: string,
    l2Override?: ChainSlot[],
    state: RouterState = defaultRouterState,
): ProviderSlot[] {
    const slots = flattenCascade(config.chains, stageId, consumerId, capability, instanceId, l2Override);
    const result: ProviderSlot[] = [];
    for (const slot of slots) {
        const keys = config.providerKeys[slot.provider];
        if (!keys?.length) continue;
        for (const key of orderedKeysForProvider(state, slot.provider, keys, config.keyUsage?.[slot.provider])) {
            result.push({ providerId: slot.provider, apiKey: key, model: slot.model });
        }
    }
    return result;
}
