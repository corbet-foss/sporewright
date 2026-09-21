// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception

import { describe, expect, it } from 'bun:test';
import { AddressedField } from 'sporewright';
import {
    cascadeConfig,
    cascadeReceipt,
    instantiateCascade,
    LLM_LAYERS,
    resolveCascadeDecision,
    resolveCascadeTensor,
    reviseCascadePreference,
} from './cascade-tensor';
import type { ChainSlot, ChainValue } from './types';

type Chains = Record<string, ChainValue>;
const slot = (provider: string, model = ''): ChainSlot => ({ provider, model });
const providers = (slots: ChainSlot[]) => slots.map(({ provider }) => provider);

describe('LLM addressed field', () => {
    it('uses the complete product address without constant or execution-only coordinates', () => {
        const config = cascadeConfig(
            { eval: { chat: [slot('stage')] } },
            { workspaceId: 'alice', stageId: 'evaluate', consumerId: 'fit', capability: 'chat' },
        );
        expect(config.layers).toEqual([...LLM_LAYERS]);
        expect(config.address).toEqual({
            workspace: 'alice',
            capability: 'chat',
            stage: 'eval',
            consumer: 'fit',
        });
    });

    it('adds instance, consumer and stage residuals without the old huge weights', () => {
        const chains: Chains = {
            tailor: { chat: [slot('stage-a'), slot('stage-b')] },
            'tailor:text': { chat: [slot('consumer-a'), slot('consumer-b')] },
            'tailor:text:sec-1': { chat: [slot('instance-a')] },
        };
        const resolution = resolveCascadeDecision(chains, {
            workspaceId: 'alice',
            stageId: 'tailor',
            consumerId: 'text',
            capability: 'chat',
            instanceId: 'sec-1',
        });
        expect(providers(resolution.slots)).toEqual([
            'instance-a', 'consumer-a', 'consumer-b', 'stage-a', 'stage-b',
        ]);
        const declared = resolution.decision.alternatives
            .flatMap(({ dimensions }) => dimensions)
            .flatMap(({ contributions }) => contributions)
            .map(({ declared_prior }) => Math.abs(declared_prior));
        expect(Math.max(...declared)).toBeLessThanOrEqual(9);
    });

    it('keeps the complete fallback set while repeated options accumulate residuals', () => {
        const chains: Chains = {
            eval: { chat: [slot('shared'), slot('stage-only'), slot('last')] },
            'eval:fit': { chat: [slot('shared'), slot('consumer-only')] },
        };
        expect(providers(resolveCascadeTensor(chains, {
            workspaceId: 'alice',
            stageId: 'evaluate', consumerId: 'fit', capability: 'chat',
        }))).toEqual(['shared', 'consumer-only', 'stage-only', 'last']);
    });

    it('puts a call-local instance preference before the persisted instance preference', () => {
        const chains: Chains = {
            'eval:fit': { chat: [slot('consumer')] },
            'eval:fit:job-1': { chat: [slot('persisted')] },
        };
        expect(providers(resolveCascadeTensor(chains, {
            workspaceId: 'alice',
            stageId: 'evaluate',
            consumerId: 'fit',
            capability: 'chat',
            instanceId: 'job-1',
            instanceOverride: [slot('override')],
        }))).toEqual(['override', 'persisted', 'consumer']);
    });

    it('isolates capability branches and preserves provider/model identity', () => {
        const chains: Chains = {
            tailor: {
                chat: [slot('groq', 'llama-4-scout'), slot('mistral', 'medium')],
                route: [slot('router')],
            },
        };
        expect(resolveCascadeTensor(chains, {
            workspaceId: 'alice',
            stageId: 'tailor', consumerId: 'text', capability: 'chat',
        })).toEqual([
            { provider: 'groq', model: 'llama-4-scout' },
            { provider: 'mistral', model: 'medium' },
        ]);
        expect(providers(resolveCascadeTensor(chains, {
            workspaceId: 'alice',
            stageId: 'tailor', consumerId: 'text', capability: 'route',
        }))).toEqual(['router']);
    });

    it('feeds normalized outcomes upward while specializing the observed consumer', () => {
        const chains: Chains = {
            eval: { chat: [slot('groq'), slot('mistral')] },
            'eval:fit': { chat: [slot('groq'), slot('mistral')] },
        };
        const fit = { workspaceId: 'alice', stageId: 'evaluate', consumerId: 'fit', capability: 'chat' };
        const { field, address } = instantiateCascade(chains, fit);
        const feedback = reviseCascadePreference(field, {
            address,
            slot: slot('groq'),
            dimension: 'failure',
            observation: 20,
            variance: 0.1,
        });
        expect(feedback).not.toBeString();
        const fitDecision = field.decide(address);
        if (typeof fitDecision === 'string') throw new Error(fitDecision);
        expect(fitDecision.alternatives.map(({ option }) => option.split('\0')[0]))
            .toEqual(['mistral', 'groq']);

        const siblingAddress = cascadeConfig(chains, {
            workspaceId: 'alice',
            stageId: 'evaluate', consumerId: 'summary', capability: 'chat',
        }).address;
        const sibling = field.decide(siblingAddress);
        if (typeof sibling === 'string') throw new Error(sibling);
        const siblingGroq = sibling.alternatives.find(({ option }) => option.startsWith('groq\0'))!;
        const learnedFailure = siblingGroq.dimensions.find(({ dimension }) => dimension === 'failure')!;
        expect(learnedFailure.mean).toBeGreaterThan(0);
        expect(learnedFailure.mean).toBeLessThan(20);
    });

    it('serializes the field and freezes a decision receipt', () => {
        const resolution = resolveCascadeDecision(
            { eval: { chat: [slot('groq')] } },
            { workspaceId: 'alice', stageId: 'evaluate', consumerId: 'fit', capability: 'chat' },
        );
        expect(AddressedField.fromJson(resolution.field.toJson())?.toJson())
            .toBe(resolution.field.toJson());
        const receipt = cascadeReceipt(resolution, 'workspace-clock:42', [slot('groq')]);
        expect(receipt.tensor_revision).toBe('workspace-clock:42');
        expect(receipt.selected).toEqual(['groq\0']);
        expect(receipt.decision.work.address_nodes).toBe(Object.keys(receipt.decision.address).length + 1);
    });

    it('excludes learned options that are not executable on the current route', () => {
        const historical = new AddressedField([...LLM_LAYERS]);
        for (let index = 0; index < 100; index++) {
            historical.rootWriter().setPrior({}, `retired-${index}\0model`, 'failure', index + 1);
        }
        const resolution = resolveCascadeDecision(
            { eval: { chat: [slot('current', 'model')] } },
            { workspaceId: 'alice', stageId: 'evaluate', consumerId: 'fit', capability: 'chat' },
            { tensorRevision: 'workspace:7', snapshot: historical.toJson() },
        );
        expect(resolution.slots).toEqual([slot('current', 'model')]);
        expect(resolution.decision.alternatives.map(({ option }) => option))
            .toEqual(['current\0model']);
        expect(resolution.decision.work.ranked_options).toBe(1);
        expect(resolution.decision.work.stored_cells_visited).toBeLessThan(20);
    });

    it('keeps a fixed human order authoritative while retaining the learned recommendation', () => {
        const context = {
            workspaceId: 'alice',
            stageId: 'evaluate',
            consumerId: 'fit',
            capability: 'chat',
        };
        const adaptive: Chains = {
            eval: { chat: [slot('first'), slot('second')] },
        };
        const learned = instantiateCascade(adaptive, context);
        reviseCascadePreference(learned.field, {
            address: learned.address,
            slot: slot('first'),
            dimension: 'failure',
            observation: 20,
            variance: 0.01,
        });
        const fixed: Chains = {
            eval: {
                chat: [
                    { ...slot('first'), routing: 'fixed' },
                    { ...slot('second'), routing: 'fixed' },
                ],
            },
        };
        const resolution = resolveCascadeDecision(fixed, context, {
            tensorRevision: 'workspace:2',
            snapshot: learned.field.toJson(),
        });
        expect(providers(resolution.recommendedSlots)).toEqual(['second', 'first']);
        expect(providers(resolution.slots)).toEqual(['first', 'second']);
        expect(resolution.policyMode).toBe('fixed');
    });

    it('treats never as a hard gate that learned scores cannot cancel', () => {
        const resolution = resolveCascadeDecision({
            eval: { chat: [slot('allowed'), slot('blocked')] },
            'eval:fit': { chat: [{ ...slot('blocked'), routing: 'never' }] },
        }, {
            workspaceId: 'alice',
            stageId: 'evaluate',
            consumerId: 'fit',
            capability: 'chat',
        });
        expect(providers(resolution.slots)).toEqual(['allowed']);
        expect(resolution.decision.alternatives.map(({ option }) => option)).toEqual(['allowed\0']);
    });

    it('lets a more specific explicit policy revoke a broader hard gate', () => {
        const resolution = resolveCascadeDecision({
            eval: { chat: [
                { ...slot('revived'), routing: 'never' },
                slot('fallback'),
            ] },
            'eval:fit': { chat: [
                { ...slot('revived'), routing: 'auto' },
                { ...slot('fallback'), routing: 'auto' },
            ] },
        }, {
            workspaceId: 'alice',
            stageId: 'evaluate',
            consumerId: 'fit',
            capability: 'chat',
        });
        expect(providers(resolution.slots)).toContain('revived');
        expect(resolution.policyMode).toBe('auto');
    });

    it('lets a more specific Smart policy revoke a broader fixed order', () => {
        const resolution = resolveCascadeDecision({
            eval: { chat: [
                { ...slot('stage-first'), routing: 'fixed' },
                { ...slot('stage-second'), routing: 'fixed' },
            ] },
            'eval:fit': { chat: [
                { ...slot('stage-second'), routing: 'auto' },
                { ...slot('stage-first'), routing: 'auto' },
            ] },
        }, {
            workspaceId: 'alice',
            stageId: 'evaluate',
            consumerId: 'fit',
            capability: 'chat',
        });
        expect(resolution.policyMode).toBe('auto');
        expect(providers(resolution.slots)).toEqual(providers(resolution.recommendedSlots));
    });
});
