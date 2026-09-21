// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { describe, expect, it } from 'bun:test';
import { flattenCascade, resolveChain, createRouterState, type ChainConfig } from './index';

function makeConfig(provider: string, keyUsage?: ChainConfig['keyUsage']): ChainConfig {
    return {
        providerKeys: {
            [provider]: ['key-one', 'key-two', 'key-three'],
        },
        keyUsage,
        chains: {
            tailor: {
                chat: [{ provider, model: 'model-a' }],
            },
        },
    };
}

describe('resolveChain key usage', () => {
    it('preserves entered key order by default', () => {
        const chain = resolveChain(makeConfig('ordered-default'), 'tailor', 'text', 'chat');
        expect(chain.map((s) => s.apiKey)).toEqual(['key-one', 'key-two', 'key-three']);
    });

    it('preserves entered key order when usage is ordered', () => {
        const chain = resolveChain(
            makeConfig('ordered-explicit', { 'ordered-explicit': 'ordered' }),
            'tailor',
            'text',
            'chat',
        );
        expect(chain.map((s) => s.apiKey)).toEqual(['key-one', 'key-two', 'key-three']);
    });

    it('rotates the first attempted key when usage is balanced (per-router state cursor)', () => {
        const provider = 'balanced-rotation';
        const config = makeConfig(provider, { [provider]: 'balanced' });
        const state = createRouterState();

        const first = resolveChain(config, 'tailor', 'text', 'chat', undefined, undefined, state).map((s) => s.apiKey);
        const second = resolveChain(config, 'tailor', 'text', 'chat', undefined, undefined, state).map((s) => s.apiKey);
        const third = resolveChain(config, 'tailor', 'text', 'chat', undefined, undefined, state).map((s) => s.apiKey);

        expect(first).toEqual(['key-one', 'key-two', 'key-three']);
        expect(second).toEqual(['key-two', 'key-three', 'key-one']);
        expect(third).toEqual(['key-three', 'key-one', 'key-two']);
    });

    it('keeps flat job overrides before persisted L2, L1, and L0 slots', () => {
        const config: ChainConfig = {
            providerKeys: {
                override: ['override-key'],
                persisted: ['persisted-key'],
                consumer: ['consumer-key'],
                global: ['global-key'],
            },
            chains: {
                'eval:fit:job-1': {
                    chat: [{ provider: 'persisted', model: 'persisted-model' }],
                },
                'eval:fit': {
                    chat: [{ provider: 'consumer', model: 'consumer-model' }],
                },
                eval: {
                    chat: [{ provider: 'global', model: 'global-model' }],
                },
            },
        };

        const chain = resolveChain(
            config,
            'evaluate',
            'fit',
            'chat',
            'job-1',
            [{ provider: 'override', model: 'override-model' }],
        );

        expect(chain.map((slot) => `${slot.providerId}:${slot.model}:${slot.apiKey}`)).toEqual([
            'override:override-model:override-key',
            'persisted:persisted-model:persisted-key',
            'consumer:consumer-model:consumer-key',
            'global:global-model:global-key',
        ]);
    });

    it('resolves the FULL tier-ordered chain — no per-tier cap', () => {
        const slots = flattenCascade(
            {
                'tailor:text:section-1': {
                    chat: [
                        { provider: 'l2-a', model: 'a' },
                        { provider: 'l2-b', model: 'b' },
                    ],
                },
                'tailor:text': {
                    chat: [
                        { provider: 'l1-a', model: 'a' },
                        { provider: 'l1-b', model: 'b' },
                        { provider: 'l1-c', model: 'c' },
                    ],
                },
                tailor: {
                    chat: [
                        { provider: 'l0-a', model: 'a' },
                        { provider: 'l0-b', model: 'b' },
                        { provider: 'l0-c', model: 'c' },
                        { provider: 'l0-d', model: 'd' },
                    ],
                },
            },
            'tailor',
            'text',
            'chat',
            'section-1',
        );

        expect(slots.map((slot) => slot.provider)).toEqual([
            'l2-a', 'l2-b', 'l1-a', 'l1-b', 'l1-c', 'l0-a', 'l0-b', 'l0-c', 'l0-d',
        ]);
    });
});
