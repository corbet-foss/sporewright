// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { describe, it, expect } from 'bun:test';
import { runAttemptChain, type AttemptTrace } from './attempt-chain';

interface Slot {
    providerId: string;
    model: string;
}

const slots: Slot[] = [
    { providerId: 'a', model: 'broken' },
    { providerId: 'b', model: 'working' },
];

function makeClock(values: number[]) {
    let index = 0;
    return () => values[index++] ?? values.at(-1) ?? 0;
}

describe('runAttemptChain', () => {
    it('falls through a failed attempt to the next provider and records latency', async () => {
        let calls = 0;
        const traces: AttemptTrace<Slot>[] = [];
        const result = await runAttemptChain(
            slots,
            { prompt: 'hello' },
            ({ attempt }) => {
                calls += 1;
                if (attempt.providerId === 'a') throw new Error('provider down');
                return 'ok';
            },
            { now: makeClock([100, 107, 110, 123]), onTrace: (t) => { traces.push(t); } },
        );

        expect(result.ok).toBe(true);
        expect(calls).toBe(2);
        if (!result.ok) throw new Error('unreachable');
        expect(result.result).toBe('ok');
        expect(result.attempt.providerId).toBe('b');
        expect(result.trace).toMatchObject([
            { attempt: slots[0], status: 'failed', error: 'provider down', latencyMs: 7, wasFallback: false },
            { attempt: slots[1], status: 'selected', latencyMs: 13, wasFallback: true },
        ]);
        // onTrace fires for every emitted entry, in order.
        expect(traces.map((t) => t.status)).toEqual(['failed', 'selected']);
    });

    it('classifyError maps a thrown error to status code / retry / metadata on the failed trace', async () => {
        const result = await runAttemptChain(
            slots,
            undefined,
            ({ attempt }) => {
                if (attempt.providerId === 'a') throw new Error('429 Too Many Requests');
                return 'ok';
            },
            {
                classifyError: (error) => ({
                    message: error instanceof Error ? error.message : String(error),
                    statusCode: 429,
                    retryAfterSeconds: 30,
                    metadata: { rateLimited: true },
                }),
            },
        );

        expect(result.ok).toBe(true);
        expect(result.trace[0]).toMatchObject({
            status: 'failed',
            statusCode: 429,
            retryAfterSeconds: 30,
            metadata: { rateLimited: true },
        });
    });

    it('records a cooldown skip as a failed trace entry (429-style) and continues', async () => {
        let calls = 0;
        const result = await runAttemptChain(
            slots,
            undefined,
            ({ attempt }) => {
                calls += 1;
                return attempt.providerId;
            },
            {
                shouldSkip: (attempt) =>
                    attempt.providerId === 'a'
                        ? { status: 'failed', reason: 'Provider cooldown active; retry after 30s', statusCode: 429, retryAfterSeconds: 30 }
                        : false,
            },
        );

        expect(result.ok).toBe(true);
        expect(calls).toBe(1); // 'a' never executed — only skipped
        expect(result.trace).toMatchObject([
            { attempt: slots[0], status: 'failed', statusCode: 429, retryAfterSeconds: 30, error: expect.stringContaining('cooldown') },
            { attempt: slots[1], status: 'selected' },
        ]);
    });

    it('drops a silent skip with no trace entry (sibling-key short-circuit)', async () => {
        let calls = 0;
        const result = await runAttemptChain(
            slots,
            undefined,
            ({ attempt }) => {
                calls += 1;
                return attempt.providerId;
            },
            {
                shouldSkip: (attempt) =>
                    attempt.providerId === 'a' ? { status: 'silent', reason: 'provider-model already skipped' } : false,
            },
        );

        expect(result.ok).toBe(true);
        expect(calls).toBe(1);
        expect(result.trace).toMatchObject([{ attempt: slots[1], status: 'selected' }]);
        expect(result.trace).toHaveLength(1);
    });

    it('critic rejection on a non-last attempt falls through to the next provider', async () => {
        let calls = 0;
        const result = await runAttemptChain(
            slots,
            undefined,
            () => {
                calls += 1;
                return calls; // 1 then 2
            },
            { critic: (value) => value === 2 },
        );

        expect(result.ok).toBe(true);
        if (!result.ok) throw new Error('unreachable');
        expect(result.result).toBe(2);
        expect(result.attempt.providerId).toBe('b');
        expect(result.trace).toMatchObject([
            { status: 'rejected', criticRejected: true },
            { status: 'selected' },
        ]);
    });

    it('returns a degraded last result when continueOnLastCriticReject is set', async () => {
        const result = await runAttemptChain(
            [{ providerId: 'only', model: 'mediocre' }],
            undefined,
            () => 'mediocre',
            { critic: () => false, continueOnLastCriticReject: true, describeResult: () => ({ usage: { totalTokens: 7 } }) },
        );

        expect(result.ok).toBe(true);
        if (!result.ok) throw new Error('unreachable');
        expect(result.result).toBe('mediocre');
        expect(result.trace).toMatchObject([
            { status: 'selected', ok: true, criticRejected: true, error: 'critic rejected result', metadata: { usage: { totalTokens: 7 } } },
        ]);
    });

    it('fails the whole chain when every attempt is rejected/failed', async () => {
        const result = await runAttemptChain(
            slots,
            undefined,
            () => 'bad',
            { critic: () => false }, // not last-pass-through → both rejected
        );

        expect(result.ok).toBe(false);
        if (result.ok) throw new Error('unreachable');
        expect(result.error).toBe('no route attempt succeeded');
        expect(result.trace.map((t) => t.status)).toEqual(['rejected', 'rejected']);
    });

    it('attaches describeResult metadata to a selected attempt', async () => {
        const result = await runAttemptChain(
            [{ providerId: 'a', model: 'working' }],
            undefined,
            () => ({ text: 'ok', usage: { totalTokens: 12 } }),
            { describeResult: (output) => ({ usage: output.usage }) },
        );

        expect(result.ok).toBe(true);
        expect(result.trace[0]?.metadata).toEqual({ usage: { totalTokens: 12 } });
    });

    it('times out a stalled attempt and continues to the next provider', async () => {
        let calls = 0;
        const result = await runAttemptChain(
            slots,
            undefined,
            async ({ signal }) => {
                calls += 1;
                if (calls === 1) {
                    await new Promise<never>((_, reject) => {
                        signal.addEventListener('abort', () => reject(signal.reason), { once: true });
                    });
                }
                return 'ok';
            },
            { attemptTimeoutMs: 5 },
        );

        expect(result.ok).toBe(true);
        expect(result.trace).toMatchObject([
            { status: 'failed', error: expect.stringContaining('timed out') },
            { status: 'selected' },
        ]);
    });

    it('throws a caller abort instead of spending a fallback attempt', async () => {
        const controller = new AbortController();
        let calls = 0;

        await expect(
            runAttemptChain(
                slots,
                undefined,
                async () => {
                    calls += 1;
                    controller.abort(new DOMException('caller stopped', 'AbortError'));
                    throw controller.signal.reason;
                },
                { signal: controller.signal },
            ),
        ).rejects.toMatchObject({ name: 'AbortError', message: 'caller stopped' });

        expect(calls).toBe(1);
    });
});
