// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Tensor-native cascade — the LLM provider/model fallback chain as a `resolve`
 * over one sparse sporewright tensor.
 *
 * --- The mapping (see sporewright docs/MODEL.md) ---
 *
 * The product's purpose is a fallback chain with override tiers: a stage default
 * (L0), a consumer override (L1), and a per-instance override (L2 — a persisted
 * key and/or a flat per-job override). That purpose maps onto the tensor as:
 *
 *   context tiers (levels, coarse→fine)  l0 ≺ l1 ≺ l2
 *       l0  inst=SHARED        — the stage default, applies to every consumer
 *       l1  inst=consumerId    — this consumer's override
 *       l2  inst=instanceId    — this instance's override
 *   option                    (provider, model), encoded as `provider\0model`
 *   dimension  `route`        the single routing objective (lower value = better)
 *
 * Two co-located layers at each option's `route` cell:
 *   - value  — the option's *declared preference*: a real per-option objective,
 *              lower = more preferred. The author of a tier declares the order of
 *              its slots; that declared order IS the value (NOT a borda rank — no
 *              votes, no positions-as-points, just the author's stated preference
 *              written straight into the cell). When the product measures
 *              reliability / quality / cost it overwrites this value with the
 *              measured truth and the chain re-orders itself.
 *   - weight — the tier-precedence gas-pedal: written per tier (the orchestrator's
 *              will), heavier for a coarser tier so a finer override always sorts
 *              ahead. `resolve` = Σ value·weight ordered ascending, so the queue
 *              is: every L2 option, then every L1, then every L0 — and within a
 *              tier, by declared preference. The order *emerges* from resolve; it
 *              is never a hand-tuned positional rank.
 *
 * The returned chain is the FULL objective-ordered set — there is no per-tier cap
 * and no dedup-without-budget. A duplicate `(provider, model)` across tiers folds
 * to its strongest (finest-tier) cell because that is the same option coordinate,
 * so it appears once, at its best position. A consumer that genuinely wants a
 * shortest-N prefix slices `resolve`'s output itself; the tensor never trims.
 */

import {
    instantiate,
    optionWeightCell,
    SHARED,
    tensorConfig,
    Tensor,
    valueCell,
    type Cursor,
    type SeedCell,
    type TensorConfig,
} from 'sporewright';
import type { ChainSlot, ChainValue } from './types';
import { stagePrefix } from './cascade-keys';

/** The override tiers as context levels, coarse → fine (precedence order). */
const L0 = 'l0';
const L1 = 'l1';
const L2 = 'l2';
const LEVELS = [L0, L1, L2] as const;

/** The single routing objective dimension. */
const ROUTE = 'route';

/**
 * Tier-precedence weights (the gas pedal). `resolve` sorts ascending by Σ v·w, so
 * a heavier weight pushes a tier's options later in the queue. Coarser = heavier,
 * so the finer override tier always leads. The spread is wide enough that no
 * intra-tier declared preference (a small non-negative value) can let a coarser
 * option overtake a finer one — the tier separation dominates the order, which is
 * exactly "an override tier re-orders the chain."
 */
const TIER_WEIGHT: Record<string, number> = {
    [L2]: 1,
    [L1]: 1_000,
    [L0]: 1_000_000,
};

/** Encode a (provider, model) pair as the option coordinate. */
function optId(provider: string, model: string): string {
    return `${provider}\0${model}`;
}

/** Decode an option coordinate back to a ChainSlot. */
function slotFromOpt(id: string): ChainSlot {
    const i = id.indexOf('\0');
    return i < 0 ? { provider: id, model: '' } : { provider: id.slice(0, i), model: id.slice(i + 1) };
}

/** Read a level's capability-keyed slot list out of the chain map. */
function levelSlots(
    chains: Record<string, ChainValue>,
    chainKey: string | undefined,
    capability: string,
): ChainSlot[] {
    if (!chainKey) return [];
    return chains[chainKey]?.[capability] ?? [];
}

/**
 * Emit one tier's slots as declarative `route` {@link SeedCell}s at `(level, inst)`
 * — the thin-config twin of the old imperative `writeTier`. The cells are appended
 * to `seeds` **in declared order**, so `instantiate` applies them exactly as the
 * old per-slot `Writer` calls did (load-bearing for overwrite + cross-core parity).
 *
 * Each present slot declares, on its own `route` cell:
 *   - a value = its position in the author's declared order for this tier
 *     (1 = most preferred). This is a *declared preference value*, not a borda
 *     rank: it is the author's stated order, written once, with no cross-ballot
 *     scoring. The earliest declaration of an option wins its value (a slot
 *     repeated within one list keeps its strongest preference).
 *   - an option-specific weight = the tier-precedence gas-pedal, heavier for a
 *     coarser tier. Writing it per option (not at the shared slot) means that
 *     when an option recurs across tiers, the weight folds to the option's own
 *     finest tier — so a duplicate `(provider, model)` resolves once, at its best
 *     position, with no budget bookkeeping.
 *
 * The cell `floor` is the tier itself (`level`): the old `t.writer(level)` minted a
 * writer floored at `level` and wrote at `level` — a write-at-floor, which a
 * `SeedCell { floor: level, level }` reproduces exactly.
 */
function tierCells(level: string, inst: string, slots: readonly ChainSlot[]): SeedCell[] {
    const cells: SeedCell[] = [];
    const tierWeight = TIER_WEIGHT[level] ?? 1;
    // 1-based declared preference: the leading slot is value 1, not 0, so the
    // tier-weight multiplier is never annihilated (0·w would collapse every
    // tier's head to the same score and break precedence).
    let rank = 1;
    const declared = new Set<string>();
    for (const slot of slots) {
        const id = optId(slot.provider, slot.model);
        if (declared.has(id)) continue; // earliest declaration is the strongest preference
        declared.add(id);
        cells.push(valueCell(level, level, inst, id, ROUTE, rank));
        cells.push(optionWeightCell(level, level, inst, id, ROUTE, tierWeight));
        rank++;
    }
    return cells;
}

/**
 * Declare the cascade tensor for a context as **data** — the SCOPE split made
 * literal. The three override tiers (L2 flat-override + persisted, L1 consumer, L0
 * stage default) become a {@link TensorConfig}; {@link resolveCascadeTensor}
 * instantiates and resolves it. Cells are appended finest-tier first (matching the
 * old imperative write order), but tier *order* is decision-irrelevant here because
 * every cell lands at its own distinct `(level, inst, option)` coordinate.
 */
export function cascadeConfig(
    chains: Record<string, ChainValue>,
    stageId: string,
    consumerId: string,
    capability: string,
    instanceId?: string,
    l2Override?: ChainSlot[],
): { config: TensorConfig; cursor: Cursor } {
    const prefix = stagePrefix(stageId);
    const l0Key = prefix;
    const l1Key = `${prefix}:${consumerId}`;
    const l2Key = instanceId ? `${prefix}:${consumerId}:${instanceId}` : undefined;

    const seeds: SeedCell[] = [];

    // L2 (finest): the flat job override leads, then any persisted L2 slots. Both
    // share the l2 coordinate `(l2, instanceId)`; the override's slots are
    // declared first so they take the lower (stronger) declared preference.
    const l2Inst = instanceId ?? SHARED;
    const l2Slots: ChainSlot[] = [
        ...(l2Override ?? []),
        ...(l2Key ? levelSlots(chains, l2Key, capability) : []),
    ];
    if (l2Slots.length > 0) seeds.push(...tierCells(L2, l2Inst, l2Slots));

    // L1: this consumer's override.
    seeds.push(...tierCells(L1, consumerId, levelSlots(chains, l1Key, capability)));

    // L0: the stage default, shared across consumers (inst = SHARED).
    seeds.push(...tierCells(L0, SHARED, levelSlots(chains, l0Key, capability)));

    // The cursor pins where we are on each tier axis. resolve enumerates every
    // visible option and orders by Σ value·weight — the full fallback chain.
    const cursor: Cursor = { [L1]: consumerId, [L2]: l2Inst };
    return { config: tensorConfig([...LEVELS], seeds), cursor };
}

/**
 * Resolve the ordered ChainSlot[] for a context — the LLM fallback chain.
 *
 * Builds the cascade tensor via {@link cascadeConfig} + {@link instantiate} (the
 * stage default L0, consumer override L1, instance/flat override L2), then returns
 * `resolve(cursor)`. The result is the full objective-ordered set of distinct
 * `(provider, model)` options, most-preferred first. The product is now thin config:
 * it declares the tiers as data; the engine seeds and resolves them.
 *
 * @param chains      capability-keyed chain map (e.g. CloudKeys.chains)
 * @param stageId     pipeline stage ('evaluate' → 'eval', else verbatim)
 * @param consumerId  consumer within the stage
 * @param capability  capability to route ('chat', 'route', 'scrape', …)
 * @param instanceId  optional L2 instance (job id, section id)
 * @param l2Override  optional flat per-job override slots (share the L2 coordinate)
 */
export function resolveCascadeTensor(
    chains: Record<string, ChainValue>,
    stageId: string,
    consumerId: string,
    capability: string,
    instanceId?: string,
    l2Override?: ChainSlot[],
): ChainSlot[] {
    const { config, cursor } = cascadeConfig(chains, stageId, consumerId, capability, instanceId, l2Override);
    const t = instantiate(config);
    if (!(t instanceof Tensor)) {
        // Unreachable for a well-formed cascade config (every cell writes at its own
        // tier floor, all tiers are in LEVELS) — surface it loudly rather than route
        // on a half-built tensor.
        throw new Error(`cascade config failed to instantiate: ${t}`);
    }
    return t.resolve(cursor).map(slotFromOpt);
}
