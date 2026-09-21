// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { describe, expect, it } from 'bun:test';
import {
    flattenCascade,
    resolveChain,
    resolveLlmRoute,
    routingPlanOf,
    createRouterState,
    LLM_LAYERS,
    type ChainConfig,
} from './index';
import { AddressedField } from 'sporewright';

function makeConfig(provider: string, keyUsage?: ChainConfig['keyUsage']): ChainConfig {
    return {
        workspaceId: 'alice',
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

    it('includes a host-approved credentialless route without inventing a key', () => {
        const config: ChainConfig = {
            workspaceId: 'alice',
            providerKeys: {},
            credentiallessProviders: ['webllm'],
            chains: {
                tailor: {
                    chat: [{ provider: 'webllm', model: 'local-model' }],
                },
            },
        };
        expect(resolveChain(config, 'tailor', 'text', 'chat')).toEqual([
            { providerId: 'webllm', model: 'local-model' },
        ]);
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
            workspaceId: 'alice',
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

    it('separates semantic tensor addresses from persisted policy keys', () => {
        const config: ChainConfig = {
            workspaceId: 'alice',
            providerKeys: { old: ['key'] },
            chains: { 'extract:analysis': { chat: [{ provider: 'old', model: 'model' }] } },
        };
        const chain = resolveLlmRoute(config, {
            capability: 'structured-generation',
            stage: 'extract',
            consumer: 'job-ad-extraction',
            policyCapability: 'chat',
            policy: [{ stageId: 'extract', consumerId: 'analysis' }],
        });
        expect(chain).toMatchObject([{ providerId: 'old', model: 'model' }]);
        expect(routingPlanOf(chain)?.address).toEqual({
            workspace: 'alice',
            capability: 'structured-generation',
            stage: 'extract',
            consumer: 'job-ad-extraction',
        });
    });

    it('reads the legacy evaluate prefix only when the canonical eval prefix is absent', () => {
        const legacy: ChainConfig = {
            workspaceId: 'alice',
            providerKeys: { old: ['old-key'] },
            chains: { evaluate: { chat: [{ provider: 'old', model: 'model' }] } },
        };
        expect(resolveLlmRoute(legacy, {
            capability: 'structured-generation',
            stage: 'evaluate',
            consumer: 'column-evaluation',
            policyCapability: 'chat',
            policy: [{ stageId: 'evaluate', consumerId: 'fit' }],
        })).toMatchObject([{ providerId: 'old' }]);

        const canonical: ChainConfig = {
            workspaceId: 'alice',
            providerKeys: { old: ['old-key'], current: ['current-key'] },
            chains: {
                evaluate: { chat: [{ provider: 'old', model: 'model' }] },
                eval: { chat: [{ provider: 'current', model: 'model' }] },
            },
        };
        expect(resolveLlmRoute(canonical, {
            capability: 'structured-generation',
            stage: 'evaluate',
            consumer: 'column-evaluation',
            policyCapability: 'chat',
            policy: [{ stageId: 'evaluate', consumerId: 'fit' }],
        }).map(({ providerId }) => providerId)).toEqual(['current']);
    });

    it('keeps workspace-less legacy hosts ephemeral but rejects persisted state without an owner', () => {
        const { workspaceId: _workspaceId, ...legacyConfig } = makeConfig('legacy');
        const config = legacyConfig as ChainConfig;
        const ephemeral = resolveChain(config, 'tailor', 'text', 'chat');
        expect(routingPlanOf(ephemeral)).toMatchObject({ workspaceId: 'local', tensorRevision: 'ephemeral' });

        config.routingState = {
            tensorRevision: 'persisted',
            snapshot: new AddressedField([...LLM_LAYERS]).toJson(),
        };
        expect(() => resolveChain(config, 'tailor', 'text', 'chat'))
            .toThrow('workspaceId is required for persisted LLM tensor routing');
    });

    it('lets several policy branches vote additively on one batched semantic route', () => {
        const config: ChainConfig = {
            workspaceId: 'alice',
            providerKeys: { shared: ['s'], first: ['f'], second: ['x'] },
            chains: {
                eval: { chat: [{ provider: 'first', model: 'm' }] },
                'eval:a': { chat: [{ provider: 'shared', model: 'm' }] },
                'eval:b': { chat: [{ provider: 'shared', model: 'm' }, { provider: 'second', model: 'm' }] },
            },
        };
        const chain = resolveLlmRoute(config, {
            capability: 'structured-generation',
            stage: 'evaluate',
            consumer: 'column-evaluation',
            policyCapability: 'chat',
            policy: [
                { stageId: 'evaluate', consumerId: 'a' },
                { stageId: 'evaluate', consumerId: 'b' },
            ],
        });
        expect(chain.map(({ providerId }) => providerId)).toEqual(['shared', 'second', 'first']);
    });
});
