// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * CareerVector LLM routing over an addressed Sporewright field.
 *
 * Address:
 * `root → workspace → capability → stage → consumer → optional instance`
 *
 * Ordered chain declarations become bounded additive route residuals. Measured
 * quality, latency, reliability, and cost can then be observed on the same path
 * without replacing the declared route policy.
 */

import {
    address,
    AddressedField,
    type Address,
    type DecisionPolicy,
    type FeedbackTrace,
    type FieldDecision,
    type FieldError,
    type RoutingReceipt,
} from 'sporewright';
import { stagePrefix } from './cascade-keys';
import type { ChainSlot, ChainValue } from './types';

export const WORKSPACE = 'workspace';
export const CAPABILITY = 'capability';
export const STAGE = 'stage';
export const CONSUMER = 'consumer';
export const INSTANCE = 'instance';
export const LLM_LAYERS = [
    WORKSPACE,
    CAPABILITY,
    STAGE,
    CONSUMER,
    INSTANCE,
] as const;

export type LlmAddressLayer = typeof LLM_LAYERS[number];

const ROUTE = 'route';
// Product mapping from an ordered list to additive residuals. Values stay
// bounded; they are not the old 1e9/1e6/1e3 lexicographic weight hack.
const STAGE_STRENGTH = 1;
const CONSUMER_STRENGTH = 3;
const INSTANCE_STRENGTH = 9;
const EXPLICIT_PREFERENCE_STRENGTH = 4;

const ROOT_WEIGHTS = {
    route: 1,
    // One workspace failure may specialize its own branch immediately, but a
    // single bounded root message must not overturn declared policy globally.
    failure: 2.5,
    'quality-loss': 5,
    'latency-cost': 0.75,
    'token-use': 0.25,
} as const;

export interface CascadeContext {
    workspaceId: string;
    /** Semantic address capability (for example `structured-generation`). */
    stageId: string;
    consumerId: string;
    capability: string;
    instanceId?: string;
    instanceOverride?: ChainSlot[];
    explorationTemperature?: number;
    /** Persisted chain capability when it differs from the semantic address. */
    policyCapability?: string;
    /** One or more persisted policy branches feeding this semantic route. */
    policySelectors?: readonly CascadePolicySelector[];
}

export interface CascadePolicySelector {
    stageId: string;
    consumerId?: string;
    instanceId?: string;
    instanceOverride?: readonly ChainSlot[];
}

export interface CascadeDeclaration {
    address: Address;
    slot: ChainSlot;
    dimension: typeof ROUTE;
    residual: number;
}

export interface CascadeConfig {
    layers: readonly string[];
    address: Address;
    declarations: CascadeDeclaration[];
    blockedOptions: Set<string>;
    fixedOrder?: ChainSlot[];
    policyMode: 'auto' | 'prefer' | 'fixed';
}

export interface CascadeFeedback {
    address: Address;
    slot: ChainSlot;
    dimension: string;
    observation: number;
    variance: number;
}

export interface CascadeDecision {
    field: AddressedField;
    address: Address;
    decision: FieldDecision;
    recommendedSlots: ChainSlot[];
    slots: ChainSlot[];
    policyMode: 'auto' | 'prefer' | 'fixed';
}

export interface CascadeState {
    tensorRevision: string;
    snapshot: string;
}

/** Compile every persisted chain declaration over a learned evidence snapshot. */
export function compileCascadeField(
    _chains: Record<string, ChainValue>,
    workspaceId: string,
    learnedSnapshot?: string,
): AddressedField {
    if (!workspaceId?.trim()) {
        throw new Error('workspaceId is required for LLM tensor routing');
    }
    const field = learnedSnapshot === undefined
        ? new AddressedField([...LLM_LAYERS])
        : AddressedField.fromJson(learnedSnapshot);
    if (field === undefined || field.layers.join('\0') !== LLM_LAYERS.join('\0')) {
        throw new Error('learned cascade state does not match the LLM address schema');
    }

    field.setDefaultRootProcessVariance(0.75);
    for (const [layer, variance] of [
        [WORKSPACE, 0.75],
        [CAPABILITY, 0.5],
        [STAGE, 0.4],
        [CONSUMER, 0.3],
        [INSTANCE, 0.25],
    ] as const) {
        field.setDefaultProcessVariance(layer, variance);
    }

    const root = field.rootWriter();
    for (const [dimension, weight] of Object.entries(ROOT_WEIGHTS)) {
        const error = root.setWeight({}, dimension, weight);
        if (error !== undefined) throw new Error(`root weight declaration failed: ${error}`);
    }

    // Persisted route policy is deliberately not materialized here. A snapshot
    // is learned belief, not a copy of every workspace policy branch. The
    // requested route's declarations are applied by `cascadeConfig` below.
    return field;
}

function optId(provider: string, model: string): string {
    return `${provider}\0${model}`;
}

function slotFromOpt(id: string): ChainSlot {
    const separator = id.indexOf('\0');
    return separator < 0
        ? { provider: id, model: '' }
        : { provider: id.slice(0, separator), model: id.slice(separator + 1) };
}

function chainSlots(
    chains: Record<string, ChainValue>,
    key: string | undefined,
    capability: string,
): ChainSlot[] {
    return key === undefined ? [] : chains[key]?.[capability] ?? [];
}

function firstDeclaredSlots(
    chains: Record<string, ChainValue>,
    keys: readonly (string | undefined)[],
    capability: string,
): ChainSlot[] {
    for (const key of keys) {
        const slots = chainSlots(chains, key, capability);
        if (slots.length > 0) return slots;
    }
    return [];
}

function uniqueSlots(slots: readonly ChainSlot[]): ChainSlot[] {
    const seen = new Set<string>();
    return slots.filter((slot) => {
        const option = optId(slot.provider, slot.model);
        if (seen.has(option)) return false;
        seen.add(option);
        return true;
    });
}

function declarations(
    at: Address,
    slots: readonly ChainSlot[],
    strength: number,
    broadBaseline: boolean,
): CascadeDeclaration[] {
    const unique = uniqueSlots(slots.filter((slot) => slot.routing !== 'never'));
    const count = Math.max(1, unique.length);
    return unique.map((slot, rank) => ({
        address: at,
        slot,
        dimension: ROUTE,
        // A broad list declares its ordinary rank in [0, 1). Finer lists add
        // a negative bounded preference residual. Every layer therefore adds;
        // no layer replaces another layer's declaration.
        residual: (broadBaseline
            ? rank / count
            : -strength * ((count - rank) / count))
            - (slot.routing === 'prefer' ? EXPLICIT_PREFERENCE_STRENGTH : 0),
    }));
}

function routePolicy(slots: readonly ChainSlot[]): {
    gates: Map<string, 'blocked' | 'allowed'>;
    fixed?: ChainSlot[];
    mode?: 'auto' | 'prefer' | 'fixed';
} {
    const unique = uniqueSlots(slots);
    const active = unique.filter((slot) => slot.routing !== 'never');
    const mode = active.some((slot) => slot.routing === 'fixed')
        ? 'fixed'
        : active.some((slot) => slot.routing === 'prefer')
            ? 'prefer'
            : active.some((slot) => slot.routing === 'auto')
                ? 'auto'
                : undefined;
    return {
        // A more specific explicit declaration can re-allow a broadly blocked
        // option. Missing declarations inherit the broader gate unchanged.
        gates: new Map(unique
            .filter((slot) => slot.routing !== undefined)
            .map((slot) => [
                optId(slot.provider, slot.model),
                slot.routing === 'never' ? 'blocked' : 'allowed',
            ])),
        ...(mode === 'fixed' ? { fixed: active } : {}),
        ...(mode === undefined ? {} : { mode }),
    };
}

function routeAddresses(context: CascadeContext): {
    stage: Address;
    consumer: Address;
    instance: Address;
    complete: Address;
} {
    const prefix = stagePrefix(context.stageId);
    const base = {
        workspace: context.workspaceId,
        capability: context.capability,
        stage: prefix,
    };
    const stage = address(base);
    const consumer = address({ ...base, consumer: context.consumerId });
    const instance = address({
        ...base,
        consumer: context.consumerId,
        ...(context.instanceId === undefined ? {} : { instance: context.instanceId }),
    });
    const complete = instance;
    return { stage, consumer, instance, complete };
}

/** Declare one LLM route as addressed additive residuals. */
export function cascadeConfig(
    chains: Record<string, ChainValue>,
    context: CascadeContext,
): CascadeConfig {
    if (!context.workspaceId?.trim()) {
        throw new Error('workspaceId is required for LLM tensor routing');
    }
    const addresses = routeAddresses(context);
    const policyCapability = context.policyCapability ?? context.capability;
    const selectors = context.policySelectors ?? [{
        stageId: context.stageId,
        consumerId: context.consumerId,
        instanceId: context.instanceId,
        instanceOverride: context.instanceOverride,
    }];
    const stageKeys = new Set<string>();
    const routeDeclarations: CascadeDeclaration[] = [];
    const blockedOptions = new Set<string>();
    let fixedOrder: ChainSlot[] | undefined;
    let policyMode: 'auto' | 'prefer' | 'fixed' = 'auto';
    const applyPolicy = (policy: ReturnType<typeof routePolicy>) => {
        for (const [option, gate] of policy.gates) {
            if (gate === 'blocked') blockedOptions.add(option);
            else blockedOptions.delete(option);
        }
        if (policy.mode !== undefined) {
            policyMode = policy.mode;
            fixedOrder = policy.mode === 'fixed' ? policy.fixed : undefined;
        }
    };
    for (const selector of selectors) {
        const prefix = stagePrefix(selector.stageId);
        const prefixes = prefix === selector.stageId ? [prefix] : [prefix, selector.stageId];
        if (!stageKeys.has(prefix)) {
            stageKeys.add(prefix);
            const stageSlots = firstDeclaredSlots(chains, prefixes, policyCapability);
            const stagePolicy = routePolicy(stageSlots);
            applyPolicy(stagePolicy);
            routeDeclarations.push(...declarations(
                addresses.stage,
                stageSlots,
                STAGE_STRENGTH,
                true,
            ));
        }
        if (selector.consumerId === undefined) continue;
        const consumerKeys = prefixes.map((candidate) => `${candidate}:${selector.consumerId}`);
        const consumerSlots = firstDeclaredSlots(chains, consumerKeys, policyCapability);
        const consumerPolicy = routePolicy(consumerSlots);
        applyPolicy(consumerPolicy);
        routeDeclarations.push(...declarations(
            addresses.consumer,
            consumerSlots,
            CONSUMER_STRENGTH,
            false,
        ));
        const instanceKeys = selector.instanceId === undefined
            ? []
            : consumerKeys.map((candidate) => `${candidate}:${selector.instanceId}`);
        const instanceSlots = [
            ...(selector.instanceOverride ?? []),
            ...firstDeclaredSlots(chains, instanceKeys, policyCapability),
        ];
        const instancePolicy = routePolicy(instanceSlots);
        applyPolicy(instancePolicy);
        routeDeclarations.push(...declarations(
            addresses.instance,
            instanceSlots,
            INSTANCE_STRENGTH,
            false,
        ));
    }
    return {
        layers: LLM_LAYERS,
        address: addresses.complete,
        declarations: coalesceDeclarations(routeDeclarations),
        blockedOptions,
        ...(fixedOrder === undefined ? {} : { fixedOrder }),
        policyMode,
    };
}

function coalesceDeclarations(input: readonly CascadeDeclaration[]): CascadeDeclaration[] {
    const combined = new Map<string, CascadeDeclaration>();
    for (const declaration of input) {
        const key = JSON.stringify([
            declaration.address,
            declaration.slot.provider,
            declaration.slot.model,
            declaration.dimension,
        ]);
        const previous = combined.get(key);
        combined.set(key, previous === undefined
            ? { ...declaration, address: { ...declaration.address }, slot: { ...declaration.slot } }
            : { ...previous, residual: previous.residual + declaration.residual });
    }
    return [...combined.values()];
}

/** Instantiate a mutable addressed field for feedback-aware callers. */
export function instantiateCascade(
    chains: Record<string, ChainValue>,
    context: CascadeContext,
    state?: CascadeState,
): { field: AddressedField; address: Address } {
    const config = cascadeConfig(chains, context);
    const field = compileCascadeField(chains, context.workspaceId, state?.snapshot);
    const writer = field.writer(WORKSPACE)!;
    for (const declaration of config.declarations) {
        const error = writer.setPrior(
            declaration.address,
            optId(declaration.slot.provider, declaration.slot.model),
            declaration.dimension,
            declaration.residual,
        );
        if (error !== undefined) throw new Error(`cascade declaration failed: ${error}`);
    }
    return { field, address: config.address };
}

/** Observe one normalized outcome at the exact address that produced it. */
export function reviseCascadePreference(
    field: AddressedField,
    feedback: CascadeFeedback,
): FeedbackTrace | FieldError {
    const writer = field.writer(WORKSPACE)!;
    return writer.observe(
        feedback.address,
        optId(feedback.slot.provider, feedback.slot.model),
        feedback.dimension,
        feedback.observation,
        feedback.variance,
    );
}

/** Resolve the complete LLM decision, retaining uncertainty and path traces. */
export function resolveCascadeDecision(
    chains: Record<string, ChainValue>,
    context: CascadeContext,
    state?: CascadeState,
): CascadeDecision {
    const config = cascadeConfig(chains, context);
    const allowed = new Set(config.declarations
        .map(({ slot }) => optId(slot.provider, slot.model))
        .filter((option) => !config.blockedOptions.has(option)));
    const { field, address: routeAddress } = instantiateCascade(chains, context, state);
    const policy: DecisionPolicy = {
        temperature: context.explorationTemperature ?? 0,
    };
    const decision = field.decideAmong(routeAddress, allowed, policy);
    if (typeof decision === 'string') throw new Error(`cascade decision failed: ${decision}`);
    const recommendedSlots = decision.alternatives
        .filter(({ viable, option }) => viable && allowed.has(option))
        .map(({ option }) => slotFromOpt(option));
    const fixedSlots = config.fixedOrder?.filter((slot) => allowed.has(optId(slot.provider, slot.model)));
    return {
        field,
        address: routeAddress,
        decision,
        recommendedSlots,
        slots: fixedSlots ?? recommendedSlots,
        policyMode: config.policyMode,
    };
}

/** Resolve the complete objective-ordered fallback chain. */
export function resolveCascadeTensor(
    chains: Record<string, ChainValue>,
    context: CascadeContext,
    state?: CascadeState,
): ChainSlot[] {
    return resolveCascadeDecision(chains, context, state).slots;
}

/** Freeze a deterministic receipt after the host chooses one or more routes. */
export function cascadeReceipt(
    resolution: CascadeDecision,
    tensorRevision: string,
    selected: readonly ChainSlot[],
): RoutingReceipt {
    return AddressedField.receipt(
        resolution.decision,
        tensorRevision,
        selected.map((slot) => optId(slot.provider, slot.model)),
    );
}
