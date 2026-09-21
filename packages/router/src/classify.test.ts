// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { describe, expect, it } from 'bun:test';
import {
    isRateLimitError,
    isCorsError,
    isTransientError,
    isProviderLevelFailure,
    describeError,
} from './index';

// A minimal stand-in for the `ai` module's static helpers. The classifiers only
// touch APICallError.isInstance / NoObjectGeneratedError.isInstance for the
// status-code path; message-based detection works without them.
const aiStub = {
    APICallError: { isInstance: (_e: unknown) => false },
    NoObjectGeneratedError: { isInstance: (_e: unknown) => false },
} as unknown as typeof import('ai');

describe('AI-SDK error classifiers (message-path)', () => {
    it('detects rate limits from message content', () => {
        expect(isRateLimitError(new Error('429 Too Many Requests'), aiStub)).toBe(true);
        expect(isRateLimitError(new Error('rate limit exceeded'), aiStub)).toBe(true);
        expect(isRateLimitError(new Error('ordinary failure'), aiStub)).toBe(false);
    });

    it('detects CORS-shaped TypeErrors with no status code', () => {
        expect(isCorsError(new TypeError('Failed to fetch'), aiStub)).toBe(true);
        expect(isCorsError(new Error('Failed to fetch'), aiStub)).toBe(false); // not a TypeError
    });

    it('isTransientError is exported and recognises transient signals (kept dead in the engine)', () => {
        expect(isTransientError(new Error('service unavailable'), aiStub)).toBe(true);
        expect(isTransientError(new Error('overloaded'), aiStub)).toBe(true);
        expect(isTransientError(new Error('nope'), aiStub)).toBe(false);
    });

    it('isProviderLevelFailure is false for message-only errors (needs a status code)', () => {
        expect(isProviderLevelFailure(new Error('bad request'), aiStub)).toBe(false);
    });

    it('describeError falls back to the error message', () => {
        expect(describeError(new Error('boom'), aiStub)).toEqual({ message: 'boom' });
        expect(describeError('plain string', aiStub)).toEqual({ message: 'plain string' });
    });
});
