// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Chain resolution — "who do I call, in what order?"
 *
 * `resolveChain` turns a {@link ChainConfig} into an ordered {@link ProviderSlot}[]
 * with API keys for a specific consumer call. The order emerges from additive
 * residuals along the complete address in `cascade-tensor.ts`. Stage, consumer,
 * and instance declarations all contribute; a repeated `(provider, model)`
 * accumulates those residuals and appears once.
 *
 * Multi-key fan-out per provider is governed by `keyUsage`: `'ordered'` keeps the
 * entered order, `'balanced'` rotates the leading key per call via the
 * {@link RouterState} cursor (default: the module singleton, so rotation is
 * process-scoped exactly as before).
 */

import {
    resolveCascadeDecision,
    type CascadeDecision,
    type CascadePolicySelector,
} from './cascade-tensor';
import type { Address, FieldDecision } from 'sporewright';
import type { ChainSlot, ChainValue, ChainConfig, ProviderKeyUsage } from './types';
import type { ProviderSlot } from './errors';
import { type RouterState, defaultRouterState } from './cooldown';

export type { ChainConfig } from './types';

export interface LlmRoutingPlan {
    schema: 'sporewright.llm-routing/v1';
    workspaceId: string;
    address: Address;
    tensorRevision: string;
    decision: FieldDecision;
    policyMode: 'auto' | 'prefer' | 'fixed';
    recommended: ChainSlot[];
    exploration?: {
        maxExecutions: number;
        reservationUnits: number;
        accountId: string;
        fundingKind: 'byok' | 'platform';
    };
}

/** Semantic tensor address plus the persisted policy branches that feed it. */
export interface LlmRouteRequest {
    capability: string;
    stage: string;
    consumer: string;
    instance?: string;
    policyCapability?: string;
    policy: readonly CascadePolicySelector[];
}

export type ResolvedProviderChain = ProviderSlot[] & { readonly routingPlan?: LlmRoutingPlan };

export function routingPlanOf(chain: readonly ProviderSlot[]): LlmRoutingPlan | undefined {
    return (chain as ResolvedProviderChain).routingPlan;
}

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
    workspaceId = 'local',
): ChainSlot[] {
    return resolveCascadeDecision(chains, {
        workspaceId,
        stageId,
        consumerId,
        capability,
        instanceId,
        instanceOverride: l2Override,
    }).slots;
}

function resolveDecision(config: ChainConfig, stageId: string, consumerId: string, capability: string, instanceId?: string, l2Override?: ChainSlot[]): CascadeDecision {
    return resolveCascadeDecision(
        config.chains,
        {
            workspaceId: routingWorkspaceId(config),
            stageId,
            consumerId,
            capability,
            instanceId,
            instanceOverride: l2Override,
            explorationTemperature: config.routingState?.explorationTemperature,
        },
        config.routingState,
    );
}

function resolveRouteDecision(config: ChainConfig, route: LlmRouteRequest): CascadeDecision {
    return resolveCascadeDecision(
        config.chains,
        {
            workspaceId: routingWorkspaceId(config),
            stageId: route.stage,
            consumerId: route.consumer,
            capability: route.capability,
            ...(route.instance === undefined ? {} : { instanceId: route.instance }),
            ...(route.policyCapability === undefined ? {} : { policyCapability: route.policyCapability }),
            policySelectors: route.policy,
            explorationTemperature: config.routingState?.explorationTemperature,
        },
        config.routingState,
    );
}

function routingWorkspaceId(config: ChainConfig): string {
    const workspaceId = config.workspaceId?.trim();
    if (workspaceId) return workspaceId;
    if (config.routingState !== undefined) {
        throw new Error('workspaceId is required for persisted LLM tensor routing');
    }
    return 'local';
}

function providerChainFromDecision(
    config: ChainConfig,
    resolution: CascadeDecision,
    state: RouterState,
): ResolvedProviderChain {
    const result: ProviderSlot[] = [];
    const credentialless = new Set(config.credentiallessProviders ?? []);
    for (const slot of resolution.slots) {
        const keys = config.providerKeys[slot.provider];
        if (keys?.length) {
            for (const key of orderedKeysForProvider(state, slot.provider, keys, config.keyUsage?.[slot.provider])) {
                result.push({ providerId: slot.provider, apiKey: key, model: slot.model });
            }
        } else if (credentialless.has(slot.provider)) {
            result.push({ providerId: slot.provider, model: slot.model });
        }
    }
    const resolved = result as ResolvedProviderChain;
    Object.defineProperty(resolved, 'routingPlan', {
        configurable: false,
        enumerable: false,
        writable: false,
        value: {
            schema: 'sporewright.llm-routing/v1',
            workspaceId: String(resolution.address.workspace ?? 'local'),
            address: resolution.address,
            tensorRevision: config.routingState?.tensorRevision ?? 'ephemeral',
            decision: resolution.decision,
            policyMode: resolution.policyMode,
            recommended: resolution.recommendedSlots.map((slot) => ({ ...slot })),
            ...(config.routingState?.exploration === undefined
                ? {}
                : { exploration: { ...config.routingState.exploration } }),
        } satisfies LlmRoutingPlan,
    });
    return resolved;
}

/** Resolve a semantic route while reading one or more legacy/persisted policy branches. */
export function resolveLlmRoute(
    config: ChainConfig,
    route: LlmRouteRequest,
    state: RouterState = defaultRouterState,
): ResolvedProviderChain {
    return providerChainFromDecision(config, resolveRouteDecision(config, route), state);
}

/**
 * Resolve a provider chain for a specific consumer call.
 *
 * Chains are complete from the host (seeds materialized on first access).
 * Resolution composes stage, consumer, and instance declarations at the requested
 * workspace, capability, stage, consumer, and optional instance address.
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
): ResolvedProviderChain {
    const resolution = resolveDecision(config, stageId, consumerId, capability, instanceId, l2Override);
    return providerChainFromDecision(config, resolution, state);
}
