// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { beforeEach, describe, it, expect } from 'bun:test';
import {
    callWithChain,
    ChainExhaustionError,
    clearProviderCooldownForTests,
    RateLimitError,
} from './index';
import type { ChainAttempt, ProviderFailure, ProviderSlot, UsageInfo } from './index';

function makeSlot(providerId: string): ProviderSlot {
    return { providerId, apiKey: `key-${providerId}`, model: 'test-model' };
}

const USAGE: UsageInfo = { inputTokens: 10, outputTokens: 5, totalTokens: 15 };

describe('RateLimitError', () => {
    it('includes providers and retry timing in message', () => {
        const err = new RateLimitError(['groq', 'together'], 30);
        expect(err.message).toContain('groq');
        expect(err.message).toContain('together');
        expect(err.message).toContain('30s');
        expect(err.providers).toEqual(['groq', 'together']);
        expect(err.retryAfterSeconds).toBe(30);
    });

    it('handles no retry timing', () => {
        const err = new RateLimitError(['groq']);
        expect(err.message).toContain('Try again later');
        expect(err.retryAfterSeconds).toBeUndefined();
    });

    it('is instanceof Error', () => {
        const err = new RateLimitError(['groq']);
        expect(err).toBeInstanceOf(Error);
        expect(err).toBeInstanceOf(RateLimitError);
        expect(err.name).toBe('RateLimitError');
    });
});

describe('callWithChain', () => {
    beforeEach(() => {
        clearProviderCooldownForTests();
    });

    it('throws when chain is empty', async () => {
        await expect(
            callWithChain([], async () => ({ result: 'ok' })),
        ).rejects.toThrow('No LLM providers configured');
    });

    it('returns result from first provider on success', async () => {
        const chain = [makeSlot('groq')];
        const result = await callWithChain(chain, async () => ({
            result: { text: 'hello' },
            usage: USAGE,
        }));
        expect(result.result).toEqual({ text: 'hello' });
        expect(result.providerId).toBe('groq');
        expect(result.usage).toEqual(USAGE);
        expect(result.attempts).toMatchObject([
            { providerId: 'groq', status: 'selected', wasFallback: false },
        ]);
    });

    it('falls through to next provider on error', async () => {
        const chain = [makeSlot('groq'), makeSlot('together')];
        let callCount = 0;
        const result = await callWithChain(chain, async () => {
            callCount++;
            if (callCount === 1) throw new Error('Groq failed');
            return { result: 'ok', usage: USAGE };
        });
        expect(result.providerId).toBe('together');
        expect(result.result).toBe('ok');
        expect(result.attempts.map((attempt) => attempt.status)).toEqual(['failed', 'selected']);
    });

    it('notifies the attempt observer as each provider is classified', async () => {
        const chain = [makeSlot('groq'), makeSlot('together')];
        const observed: string[] = [];
        let callCount = 0;
        const result = await callWithChain(
            chain,
            async () => {
                callCount++;
                if (callCount === 1) throw new Error('Groq failed');
                return { result: 'ok', usage: USAGE };
            },
            {
                onAttempt: (attempt) => {
                    observed.push(`${attempt.providerId}:${attempt.status}`);
                },
            },
        );

        expect(result.result).toBe('ok');
        expect(observed).toEqual(['groq:failed', 'together:selected']);
    });

    it('fires the onTrace observer for every emitted trace (host telemetry seam)', async () => {
        const chain = [makeSlot('groq'), makeSlot('together')];
        const traced: string[] = [];
        let callCount = 0;
        await callWithChain(
            chain,
            async () => {
                callCount++;
                if (callCount === 1) throw new Error('Groq failed');
                return { result: 'ok', usage: USAGE };
            },
            {
                onTrace: (trace) => {
                    traced.push(`${trace.attempt.providerId}:${trace.status}`);
                },
            },
        );
        expect(traced).toEqual(['groq:failed', 'together:selected']);
    });

    it('times out a stalled provider attempt and falls through', async () => {
        const chain = [makeSlot('groq'), makeSlot('together')];
        let callCount = 0;
        const result = await callWithChain(
            chain,
            async (_model, signal) => {
                callCount++;
                if (callCount === 1) {
                    await new Promise<never>((_, reject) => {
                        signal.addEventListener('abort', () => reject(signal.reason), { once: true });
                    });
                }
                return { result: 'ok', usage: USAGE };
            },
            { attemptTimeoutMs: 10 },
        );

        expect(result.providerId).toBe('together');
        expect(result.result).toBe('ok');
        expect(result.attempts).toMatchObject([
            { providerId: 'groq', status: 'failed', error: expect.stringContaining('timed out') },
            { providerId: 'together', status: 'selected' },
        ]);
    });

    it('throws RateLimitError when all providers are rate-limited', async () => {
        const chain = [makeSlot('groq'), makeSlot('together')];
        await expect(
            callWithChain(chain, async () => {
                throw new Error('429 Too Many Requests');
            }),
        ).rejects.toBeInstanceOf(RateLimitError);
    });

    it('records provider cooldown attempts after a rate limit', async () => {
        const chain = [makeSlot('groq')];
        await expect(
            callWithChain(chain, async () => {
                throw new Error('429 Too Many Requests');
            }),
        ).rejects.toBeInstanceOf(RateLimitError);

        try {
            await callWithChain(chain, async () => ({ result: 'should not run' }));
            throw new Error('expected cooldown to throw');
        } catch (err) {
            expect(err).toBeInstanceOf(RateLimitError);
            expect((err as RateLimitError).attempts).toMatchObject([
                {
                    providerId: 'groq',
                    status: 'failed',
                    statusCode: 429,
                    error: expect.stringContaining('Provider cooldown active'),
                },
            ]);
        }
    });

    it('does not replay active cooldown for duplicate provider-model keys', async () => {
        await expect(
            callWithChain([makeSlot('groq')], async () => {
                throw new Error('429 Too Many Requests');
            }),
        ).rejects.toBeInstanceOf(RateLimitError);

        const duplicateKeyChain: ProviderSlot[] = [
            { providerId: 'groq', model: 'test-model', apiKey: 'key-a' },
            { providerId: 'groq', model: 'test-model', apiKey: 'key-b' },
        ];
        let calls = 0;

        try {
            await callWithChain(duplicateKeyChain, async () => {
                calls++;
                return { result: 'should not run' };
            });
            throw new Error('expected cooldown to throw');
        } catch (err) {
            expect(err).toBeInstanceOf(RateLimitError);
            expect(calls).toBe(0);
            expect((err as RateLimitError).attempts).toMatchObject([
                { providerId: 'groq', model: 'test-model', status: 'failed' },
            ]);
            expect((err as RateLimitError).attempts).toHaveLength(1);
        }
    });

    it('throws regular error when some providers fail with non-429 errors', async () => {
        const chain = [makeSlot('groq'), makeSlot('together')];
        let callCount = 0;
        await expect(
            callWithChain(chain, async () => {
                callCount++;
                if (callCount === 1) throw new Error('429 rate limit');
                throw new Error('Server error');
            }),
        ).rejects.toThrow('All 2 providers failed');
    });

    it('critic rejection falls through to next provider', async () => {
        const chain = [makeSlot('groq'), makeSlot('together')];
        let callCount = 0;
        const result = await callWithChain(
            chain,
            async () => {
                callCount++;
                return { result: callCount, usage: USAGE };
            },
            { critic: (r) => r === 2, label: 'test' },
        );
        expect(result.result).toBe(2);
        expect(result.providerId).toBe('together');
        expect(result.attempts.map((attempt) => attempt.status)).toEqual(['rejected', 'selected']);
    });

    it('critic passes on last provider even if critic fails', async () => {
        const chain = [makeSlot('groq')];
        const result = await callWithChain(
            chain,
            async () => ({ result: 'mediocre', usage: USAGE }),
            { critic: () => false },
        );
        expect(result.result).toBe('mediocre');
        expect(result.attempts).toMatchObject([{ status: 'selected', criticRejected: true }]);
    });
});

// =============================================================================
// Provider failure normalization matrix — the same user-visible behaviour holds
// against any provider adapter; the normalized error names the provider and never
// leaks the API key.
// =============================================================================

interface FailureMode {
    name: 'malformed-output' | 'timeout' | 'quota-exhausted' | 'refusal' | 'server-5xx';
    simulate: () => Error;
    expectedClass: typeof RateLimitError | typeof ChainExhaustionError;
}

const FAILURE_MODES: ReadonlyArray<FailureMode> = [
    {
        name: 'malformed-output',
        simulate: () => Object.assign(new Error('Cannot parse output: invalid JSON'), { statusCode: 400 }),
        expectedClass: ChainExhaustionError,
    },
    {
        name: 'timeout',
        simulate: () => Object.assign(new Error('Request timeout exceeded'), { statusCode: 504 }),
        expectedClass: ChainExhaustionError,
    },
    {
        name: 'quota-exhausted',
        simulate: () => Object.assign(new Error('429 Too Many Requests'), { statusCode: 429 }),
        expectedClass: RateLimitError,
    },
    {
        name: 'refusal',
        simulate: () => Object.assign(new Error('content_policy_violation: provider refused to respond'), { statusCode: 400 }),
        expectedClass: ChainExhaustionError,
    },
    {
        name: 'server-5xx',
        simulate: () => Object.assign(new Error('upstream returned 503'), { statusCode: 503 }),
        expectedClass: ChainExhaustionError,
    },
];

const PROVIDERS_UNDER_TEST: ReadonlyArray<string> = ['groq', 'together', 'fireworks', 'openrouter'];
const SECRET_SHAPED_KEY = 'sk-supersecret-pattern-DO-NOT-LEAK';

function makeSlotWithKey(providerId: string, apiKey: string): ProviderSlot {
    return { providerId, apiKey, model: 'test-model' };
}

describe('provider failure normalization matrix', () => {
    beforeEach(() => {
        clearProviderCooldownForTests();
    });

    for (const mode of FAILURE_MODES) {
        for (const providerId of PROVIDERS_UNDER_TEST) {
            it(`normalizes ${mode.name} on ${providerId} into ${mode.expectedClass.name}`, async () => {
                const chain = [makeSlotWithKey(providerId, SECRET_SHAPED_KEY)];
                let captured: unknown;
                try {
                    await callWithChain(chain, async () => {
                        throw mode.simulate();
                    });
                } catch (err) {
                    captured = err;
                }
                expect(captured).toBeInstanceOf(mode.expectedClass);
                const err = captured as Error & { attempts?: ChainAttempt[]; providers?: string[]; failures?: ProviderFailure[] };
                const surface = JSON.stringify({
                    message: err.message,
                    providers: err.providers ?? [],
                    failures: err.failures ?? [],
                    attempts: err.attempts ?? [],
                });
                expect(surface).toContain(providerId);
            });

            it(`${mode.name} on ${providerId} does not leak the API key`, async () => {
                const chain = [makeSlotWithKey(providerId, SECRET_SHAPED_KEY)];
                let captured: unknown;
                try {
                    await callWithChain(chain, async () => {
                        throw mode.simulate();
                    });
                } catch (err) {
                    captured = err;
                }
                const err = captured as Error & { attempts?: ChainAttempt[]; providers?: string[]; failures?: ProviderFailure[] };
                const surface = JSON.stringify({
                    message: err.message,
                    providers: err.providers ?? [],
                    failures: err.failures ?? [],
                    attempts: err.attempts ?? [],
                });
                expect(surface).not.toContain(SECRET_SHAPED_KEY);
            });
        }
    }
});
