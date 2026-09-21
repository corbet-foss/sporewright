// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { describe, expect, it } from 'bun:test';
import { AddressedField } from 'sporewright';
import { learningEventForTrace } from './learning';
import type { LlmRoutingPlan } from './resolve';
import type { AttemptTrace } from './attempt-chain';
import type { ProviderSlot } from './errors';

const slot: ProviderSlot = { providerId: 'provider', model: 'model', apiKey: 'secret-key' };

function plan(): LlmRoutingPlan {
    const field = new AddressedField(['workspace', 'capability', 'stage', 'consumer', 'instance']);
    const writer = field.writer('workspace')!;
    writer.setPrior(
        { workspace: 'alice', capability: 'structured-generation', stage: 'extract', consumer: 'job-ad-extraction' },
        'provider\0model',
        'route',
        0,
    );
    const decision = field.decide({
        workspace: 'alice',
        capability: 'structured-generation',
        stage: 'extract',
        consumer: 'job-ad-extraction',
    });
    if (typeof decision === 'string') throw new Error(decision);
    return {
        schema: 'sporewright.llm-routing/v1',
        workspaceId: 'alice',
        address: decision.address,
        tensorRevision: 'workspace:1',
        decision,
        policyMode: 'auto',
        recommended: [{ provider: 'provider', model: 'model' }],
    };
}

describe('LLM learning events', () => {
    it('encodes measured quality and usage without leaking the key', () => {
        const trace: AttemptTrace<ProviderSlot> = {
            attempt: slot,
            attemptNo: 1,
            status: 'selected',
            ok: true,
            latencyMs: 1_000,
            wasFallback: false,
            metadata: {
                qualityLoss: 0.35,
                usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
            },
        };
        const event = learningEventForTrace(plan(), trace, 123)!;
        expect(event.observations.find(({ dimension }) => dimension === 'quality-loss')?.value).toBe(0.35);
        expect(event.observations.some(({ dimension }) => dimension === 'token-use')).toBe(true);
        expect(JSON.stringify(event)).not.toContain(slot.apiKey);
    });

    it('does not mistake a scheduling skip for model evidence', () => {
        const trace: AttemptTrace<ProviderSlot> = {
            attempt: slot,
            attemptNo: 1,
            status: 'skipped',
            ok: false,
            latencyMs: 0,
            wasFallback: false,
            error: 'cooldown',
        };
        expect(learningEventForTrace(plan(), trace, 123)).toBeUndefined();
    });

    it('does not relearn a cached cooldown as another provider failure', () => {
        const trace: AttemptTrace<ProviderSlot> = {
            attempt: slot,
            attemptNo: 1,
            status: 'failed',
            ok: false,
            latencyMs: 0,
            wasFallback: false,
            error: 'Provider cooldown active',
            statusCode: 429,
            metadata: { executed: false },
        };
        expect(learningEventForTrace(plan(), trace, 123)).toBeUndefined();
    });

    it('keeps unmeasured product quality unknown instead of inventing a perfect score', () => {
        const trace: AttemptTrace<ProviderSlot> = {
            attempt: slot,
            attemptNo: 1,
            status: 'selected',
            ok: true,
            latencyMs: 25,
            wasFallback: false,
        };
        const event = learningEventForTrace(plan(), trace, 123)!;
        expect(event.observations.map(({ dimension }) => dimension)).toEqual([
            'failure',
            'latency-cost',
        ]);
    });
});
