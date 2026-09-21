// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Cascade — chain-key address scheme.
 *
 * Chain keys address declarations at increasing path depth:
 *   prefix                        stage residuals
 *   prefix:consumerId             consumer residuals
 *   prefix:consumerId:instanceId  instance residuals
 *
 * Stage prefix = stageId, except evaluate → 'eval'.
 *
 * These are pure address helpers. The cascade *order* — the (provider, model)
 * fallback chain — is not computed here: it emerges from additive residuals in
 * the addressed Sporewright field (see `cascade-tensor.ts`). This module is the
 * single canonical home of `stagePrefix` (cascade-tensor.ts imports it from here).
 */

import type { ChainSlot, ChainValue } from './types';

// =============================================================================
// Chain key helpers
// =============================================================================

/** Stage prefix used in chain keys. evaluate → 'eval', everything else → ledKey. */
export function stagePrefix(stageId: string): string {
    return stageId === 'evaluate' ? 'eval' : stageId;
}

/** Chain key for L1 (consumer level). Always 2 segments. */
export function chainKeyL1(stageId: string, consumerId: string): string {
    return `${stagePrefix(stageId)}:${consumerId}`;
}

/** Chain key for L2 (instance override). Always 3 segments. */
export function chainKeyL2(stageId: string, consumerId: string, instanceId: string): string {
    return `${stagePrefix(stageId)}:${consumerId}:${instanceId}`;
}

/** Check if a chain key belongs to a specific stage. */
export function isStageChainKey(key: string, stageId: string): boolean {
    return key.startsWith(`${stagePrefix(stageId)}:`);
}

// =============================================================================
// Slot helpers
// =============================================================================

/** Slot identity key for deduplication. */
export function slotKey(s: ChainSlot): string {
    return `${s.provider}\0${s.model}`;
}

/** Flatten all capability-keyed chains to flat slot arrays. */
export function flattenAllChainValues(chains: Record<string, ChainValue>): Record<string, ChainSlot[]> {
    const flat: Record<string, ChainSlot[]> = {};
    for (const [key, capMap] of Object.entries(chains)) {
        const slots = Object.values(capMap).flat();
        if (slots.length > 0) flat[key] = slots;
    }
    return flat;
}

/** Collect all (provider, model) pairs across a stage's cascade. */
export function usedSlotKeys(chains: Record<string, ChainValue>, stageId: string): Set<string> {
    const keys = new Set<string>();
    for (const [k, capMap] of Object.entries(chains)) {
        if (!isStageChainKey(k, stageId)) continue;
        for (const slots of Object.values(capMap)) {
            for (const s of slots) keys.add(slotKey(s));
        }
    }
    return keys;
}

/** Extract L2 keys (3+ segments) for a stage. */
export function l2KeysForStage(chains: Record<string, ChainValue>, stageId: string): string[] {
    return Object.keys(chains)
        .filter(k => isStageChainKey(k, stageId) && k.split(':').length >= 3)
        .sort();
}

/** Extract the instance ID from a chain key (last segment). */
export function instanceIdFromKey(key: string): string {
    return key.split(':').pop()!;
}
