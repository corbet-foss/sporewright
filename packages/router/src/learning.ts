// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/** Product-level LLM outcome encoding over Sporewright's generic field math. */

import { AddressedField, type RoutingReceipt } from 'sporewright';
import type { AttemptTrace } from './attempt-chain';
import type { ProviderSlot } from './errors';
import type { LlmRoutingPlan } from './resolve';

export interface LlmLearningObservation {
    dimension: 'failure' | 'quality-loss' | 'latency-cost' | 'token-use';
    value: number;
    variance: number;
}

export interface LlmLearningEvent {
    schema: 'sporewright.llm-outcome/v1';
    eventId: string;
    workspaceId: string;
    receipt: RoutingReceipt;
    attempt: {
        attemptNo: number;
        providerId: string;
        model: string;
        status: AttemptTrace<ProviderSlot>['status'];
        statusCode?: number;
        providerLevelFailure?: boolean;
        criticRejected?: boolean;
    };
    observations: LlmLearningObservation[];
    occurredAtMs: number;
}

function optionId(slot: Pick<ProviderSlot, 'providerId' | 'model'>): string {
    return `${slot.providerId}\0${slot.model}`;
}

function boundedLogCost(value: number, reference: number): number {
    if (!Number.isFinite(value) || value <= 0) return 0;
    return Math.min(2, Math.log1p(value) / Math.log1p(reference));
}

function eventId(): string {
    return globalThis.crypto?.randomUUID?.()
        ?? `llm-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function totalTokens(trace: AttemptTrace<ProviderSlot>): number | undefined {
    const usage = trace.metadata?.usage;
    if (usage === null || typeof usage !== 'object') return undefined;
    const value = (usage as { totalTokens?: unknown }).totalTokens;
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
}

function measuredQualityLoss(trace: AttemptTrace<ProviderSlot>): number | undefined {
    const value = trace.metadata?.qualityLoss;
    return typeof value === 'number' && Number.isFinite(value)
        ? Math.max(0, Math.min(1, value))
        : undefined;
}

/**
 * Turn one actual provider attempt into bounded, dimension-specific evidence.
 * Skips are scheduling facts, not new model evidence, and are therefore ignored.
 */
export function learningEventForTrace(
    plan: LlmRoutingPlan | undefined,
    trace: AttemptTrace<ProviderSlot>,
    now = Date.now(),
): LlmLearningEvent | undefined {
    if (plan === undefined
        || plan.tensorRevision === 'ephemeral'
        || trace.status === 'skipped'
        || trace.metadata?.executed === false) {
        return undefined;
    }

    const observations: LlmLearningObservation[] = [];
    if (trace.status === 'failed') {
        observations.push({ dimension: 'failure', value: 1, variance: 0.1 });
    } else {
        observations.push({ dimension: 'failure', value: 0, variance: 0.1 });
        const qualityLoss = trace.status === 'rejected' || trace.criticRejected
            ? 1
            : measuredQualityLoss(trace);
        // A parseable result is not automatically a good result. When the host
        // has no product-quality measurement, preserve uncertainty rather than
        // teaching the shared field that this option was perfect.
        if (qualityLoss !== undefined) {
            observations.push({
                dimension: 'quality-loss',
                value: qualityLoss,
                variance: 0.15,
            });
        }
    }
    observations.push({
        dimension: 'latency-cost',
        value: boundedLogCost(trace.latencyMs, 60_000),
        variance: 0.25,
    });
    const tokens = totalTokens(trace);
    if (tokens !== undefined) {
        observations.push({
            dimension: 'token-use',
            value: boundedLogCost(tokens, 100_000),
            variance: 0.25,
        });
    }

    return {
        schema: 'sporewright.llm-outcome/v1',
        eventId: eventId(),
        workspaceId: plan.workspaceId,
        receipt: AddressedField.receipt(
            plan.decision,
            plan.tensorRevision,
            [optionId(trace.attempt)],
        ),
        attempt: {
            attemptNo: trace.attemptNo,
            providerId: trace.attempt.providerId,
            model: trace.attempt.model,
            status: trace.status,
            ...(trace.statusCode === undefined ? {} : { statusCode: trace.statusCode }),
            ...(trace.providerLevelFailure === undefined
                ? {}
                : { providerLevelFailure: trace.providerLevelFailure }),
            ...(trace.criticRejected === undefined ? {} : { criticRejected: trace.criticRejected }),
        },
        observations,
        occurredAtMs: now,
    };
}
