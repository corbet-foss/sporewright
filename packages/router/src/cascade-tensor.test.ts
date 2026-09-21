// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { describe, expect, it } from 'bun:test';
import { instantiate, SHARED, Tensor } from 'sporewright';
import { cascadeConfig, resolveCascadeTensor } from './cascade-tensor';
import { stagePrefix } from './cascade-keys';
import type { ChainSlot, ChainValue } from './types';

type Chains = Record<string, ChainValue>;
const s = (provider: string, model = ''): ChainSlot => ({ provider, model });
const providers = (slots: ChainSlot[]) => slots.map((x) => x.provider);

describe('resolveCascadeTensor — the chain emerges from resolve over the tensor', () => {
    it('orders by tier: every L2, then every L1, then every L0', () => {
        const chains: Chains = {
            tailor: { chat: [s('l0-a'), s('l0-b'), s('l0-c')] },
            'tailor:text': { chat: [s('l1-a'), s('l1-b')] },
            'tailor:text:sec-1': { chat: [s('l2-a')] },
        };
        const chain = resolveCascadeTensor(chains, 'tailor', 'text', 'chat', 'sec-1');
        expect(providers(chain)).toEqual(['l2-a', 'l1-a', 'l1-b', 'l0-a', 'l0-b', 'l0-c']);
    });

    it('an override tier RE-ORDERS the chain — adding an L1 lifts it above L0', () => {
        const base: Chains = { eval: { chat: [s('g0'), s('g1')] } };
        const withoutOverride = resolveCascadeTensor(base, 'evaluate', 'fit', 'chat');
        expect(providers(withoutOverride)).toEqual(['g0', 'g1']);

        const withOverride: Chains = {
            ...base,
            'eval:fit': { chat: [s('c0')] },
        };
        const reordered = resolveCascadeTensor(withOverride, 'evaluate', 'fit', 'chat');
        // c0 was declared at the finer (consumer) tier — it now leads the chain.
        expect(providers(reordered)).toEqual(['c0', 'g0', 'g1']);
    });

    it('within a tier, declared preference order is preserved (the lower value leads)', () => {
        const chains: Chains = { eval: { chat: [s('first'), s('second'), s('third')] } };
        const chain = resolveCascadeTensor(chains, 'evaluate', 'x', 'chat');
        expect(providers(chain)).toEqual(['first', 'second', 'third']);
    });

    it('returns the FULL resolved set — no per-tier cap (L0 of 4 keeps all 4)', () => {
        const chains: Chains = {
            tailor: { chat: [s('a'), s('b'), s('c'), s('d')] },
            'tailor:text': { chat: [s('e'), s('f'), s('g')] },
            'tailor:text:sec': { chat: [s('h'), s('i')] },
        };
        const chain = resolveCascadeTensor(chains, 'tailor', 'text', 'chat', 'sec');
        // 2 (L2) + 3 (L1) + 4 (L0) = 9, all present, tier-ordered.
        expect(providers(chain)).toEqual(['h', 'i', 'e', 'f', 'g', 'a', 'b', 'c', 'd']);
    });

    it('a cross-tier duplicate folds to its FINEST tier — appears once, at its best position', () => {
        const chains: Chains = {
            eval: { chat: [s('shared'), s('only-l0')] },
            'eval:fit': { chat: [s('shared')] }, // same option declared finer
        };
        const chain = resolveCascadeTensor(chains, 'evaluate', 'fit', 'chat');
        // `shared` is declared at L1 and L0; it folds to L1 and leads — no duplicate.
        expect(providers(chain)).toEqual(['shared', 'only-l0']);
        expect(providers(chain).filter((p) => p === 'shared')).toHaveLength(1);
    });

    it('a flat job override leads its persisted L2 (both share the instance coordinate)', () => {
        const chains: Chains = {
            'eval:fit': { chat: [s('l1') ] },
            'eval:fit:job-1': { chat: [s('persisted-l2')] },
        };
        const chain = resolveCascadeTensor(
            chains, 'evaluate', 'fit', 'chat', 'job-1', [s('override')],
        );
        expect(providers(chain)).toEqual(['override', 'persisted-l2', 'l1']);
    });

    it('a consumer with no override gets only the stage default (L0)', () => {
        const chains: Chains = { eval: { chat: [s('g0'), s('g1')] } };
        const chain = resolveCascadeTensor(chains, 'evaluate', 'unknown', 'chat');
        expect(providers(chain)).toEqual(['g0', 'g1']);
    });

    it('reads only the requested capability', () => {
        const chains: Chains = {
            eval: { chat: [s('chat-p')], route: [s('route-p')] },
        };
        expect(providers(resolveCascadeTensor(chains, 'evaluate', 'x', 'chat'))).toEqual(['chat-p']);
        expect(providers(resolveCascadeTensor(chains, 'evaluate', 'x', 'route'))).toEqual(['route-p']);
        expect(resolveCascadeTensor(chains, 'evaluate', 'x', 'scrape')).toEqual([]);
    });

    it('empty chains resolve to an empty chain; a lone override still resolves', () => {
        expect(resolveCascadeTensor({}, 'evaluate', 'x', 'chat')).toEqual([]);
        expect(providers(resolveCascadeTensor({}, 'evaluate', 'x', 'chat', undefined, [s('only')]))).toEqual(['only']);
    });

    it('preserves (provider, model) identity through resolve', () => {
        const chains: Chains = { tailor: { chat: [s('groq', 'llama-4-scout'), s('mistral', 'medium')] } };
        const chain = resolveCascadeTensor(chains, 'tailor', 'text', 'chat');
        expect(chain).toEqual([
            { provider: 'groq', model: 'llama-4-scout' },
            { provider: 'mistral', model: 'medium' },
        ]);
    });
});

describe('cascadeConfig — the thin-config refactor is byte-identical to the imperative writeTier path', () => {
    // The original (pre-refactor) imperative cascade, reproduced verbatim as the
    // witness. If `cascadeConfig` + `instantiate` ever diverges from these exact
    // `Writer` calls, this fails — the proof that the SCOPE-split refactor changed
    // no routing decision.
    const L0 = 'l0', L1 = 'l1', L2 = 'l2';
    const LEVELS = [L0, L1, L2] as const;
    const ROUTE = 'route';
    const TIER_WEIGHT: Record<string, number> = { [L2]: 1, [L1]: 1_000, [L0]: 1_000_000 };
    const optId = (provider: string, model: string) => `${provider}\0${model}`;

    function writeTierImperative(t: Tensor, level: string, inst: string, slots: readonly ChainSlot[]): void {
        const writer = t.writer(level);
        if (!writer) return;
        const tierWeight = TIER_WEIGHT[level] ?? 1;
        let rank = 1;
        const declared = new Set<string>();
        for (const slot of slots) {
            const id = optId(slot.provider, slot.model);
            if (declared.has(id)) continue;
            declared.add(id);
            writer.setValue(level, inst, id, ROUTE, rank);
            writer.setOptionWeight(level, inst, id, ROUTE, tierWeight);
            rank++;
        }
    }

    function imperativeCascade(
        chains: Chains,
        stageId: string,
        consumerId: string,
        capability: string,
        instanceId?: string,
        l2Override?: ChainSlot[],
    ): Tensor {
        const prefix = stagePrefix(stageId);
        const l1Key = `${prefix}:${consumerId}`;
        const l2Key = instanceId ? `${prefix}:${consumerId}:${instanceId}` : undefined;
        const t = new Tensor([...LEVELS]);
        const l2Inst = instanceId ?? SHARED;
        const l2Slots: ChainSlot[] = [
            ...(l2Override ?? []),
            ...(l2Key ? chains[l2Key]?.[capability] ?? [] : []),
        ];
        if (l2Slots.length > 0) writeTierImperative(t, L2, l2Inst, l2Slots);
        writeTierImperative(t, L1, consumerId, chains[l1Key]?.[capability] ?? []);
        writeTierImperative(t, L0, SHARED, chains[prefix]?.[capability] ?? []);
        return t;
    }

    // A spread of cases exercising every tier-presence + dedup + override branch.
    const cases: Array<[string, () => readonly [Chains, string, string, string, string?, ChainSlot[]?]]> = [
        ['all three tiers', () => [{
            tailor: { chat: [s('l0-a'), s('l0-b'), s('l0-c')] },
            'tailor:text': { chat: [s('l1-a'), s('l1-b')] },
            'tailor:text:sec-1': { chat: [s('l2-a')] },
        }, 'tailor', 'text', 'chat', 'sec-1', undefined]],
        ['cross-tier duplicate', () => [{
            eval: { chat: [s('shared'), s('only-l0')] },
            'eval:fit': { chat: [s('shared')] },
        }, 'evaluate', 'fit', 'chat', undefined, undefined]],
        ['flat override + persisted L2', () => [{
            'eval:fit': { chat: [s('l1')] },
            'eval:fit:job-1': { chat: [s('persisted-l2')] },
        }, 'evaluate', 'fit', 'chat', 'job-1', [s('override')]]],
        ['L0 only (unknown consumer)', () => [{ eval: { chat: [s('g0'), s('g1')] } }, 'evaluate', 'unknown', 'chat', undefined, undefined]],
        ['empty chains', () => [{}, 'evaluate', 'x', 'chat', undefined, undefined]],
        ['lone flat override, no chains', () => [{}, 'evaluate', 'x', 'chat', undefined, [s('only')]]],
        ['repeated slot within a tier (dedup)', () => [{ eval: { chat: [s('dup'), s('dup'), s('other')] } }, 'evaluate', 'x', 'chat', undefined, undefined]],
        ['(provider, model) identity', () => [{ tailor: { chat: [s('groq', 'llama-4-scout'), s('mistral', 'medium')] } }, 'tailor', 'text', 'chat', undefined, undefined]],
    ];

    for (const [name, mk] of cases) {
        it(`byte-identical tensor for: ${name}`, () => {
            const [chains, stage, consumer, cap, inst, override] = mk();
            const imperative = imperativeCascade(chains, stage, consumer, cap, inst, override);
            const { config } = cascadeConfig(chains, stage, consumer, cap, inst, override);
            const declared = instantiate(config);
            expect(declared).toBeInstanceOf(Tensor);
            // The wire form is the byte witness — identical bytes ⇒ identical tensor.
            expect((declared as Tensor).toJson()).toEqual(imperative.toJson());
        });
    }
});
