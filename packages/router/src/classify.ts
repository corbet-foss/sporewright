// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * AI-SDK error classifiers.
 *
 * All helpers are generic to the Vercel AI SDK and take the lazily-loaded `ai`
 * module as a parameter so this file pulls no SDK at parse time.
 */

type AiModule = typeof import('ai');

/** Extract HTTP status code from an AI SDK error. */
export function extractStatusCode(error: unknown, aiMod: AiModule): number | undefined {
    if (error instanceof Error && aiMod.APICallError.isInstance(error)) {
        return (error as import('ai').APICallError).statusCode;
    }
    return undefined;
}

/** Summarize an error for logging / failure records. */
export function describeError(error: unknown, aiMod: AiModule): { message: string; statusCode?: number } {
    const statusCode = extractStatusCode(error, aiMod);
    if (statusCode) return { message: (error as Error).message, statusCode };
    if (error instanceof Error && aiMod.NoObjectGeneratedError.isInstance(error)) {
        return { message: 'Structured output parse failed' };
    }
    return { message: error instanceof Error ? error.message : String(error) };
}

/** Check if an error is a 429 rate limit from any AI provider. */
export function isRateLimitError(error: unknown, aiMod: AiModule): boolean {
    const code = extractStatusCode(error, aiMod);
    if (code === 429) return true;
    if (error instanceof Error) {
        const msg = error.message.toLowerCase();
        return msg.includes('429') || msg.includes('rate limit') || msg.includes('too many requests');
    }
    return false;
}

/**
 * Check if an error is likely a CORS block.
 * Browser-originated CORS failures surface as TypeError with no status code.
 * We can't suppress the browser's own network-error log, but we can tag
 * these so the engine emits one descriptive warning instead of a cryptic
 * "failed" message, and suppresses repeats for the same provider.
 */
export function isCorsError(error: unknown, aiMod: AiModule): boolean {
    if (extractStatusCode(error, aiMod) !== undefined) return false; // has HTTP status → not CORS
    if (error instanceof TypeError) {
        const msg = error.message.toLowerCase();
        return msg.includes('failed to fetch') || msg.includes('network error') ||
               msg.includes('networkerror') || msg.includes('load failed');
    }
    return false;
}

/**
 * Check if an error is transient (worth retrying via fallback).
 *
 * NOTE: this helper is exported but intentionally NOT wired into the engine —
 * `callWithChain` falls through on ANY thrown error, not only transient ones.
 * Wiring it in would silently change routing. Kept available for hosts that
 * want their own transient classification.
 */
export function isTransientError(error: unknown, aiMod: AiModule): boolean {
    const code = extractStatusCode(error, aiMod);
    if (code && [402, 429, 500, 502, 503, 529].includes(code)) return true;
    if (error instanceof Error) {
        // AI SDK retryable errors
        if ('isRetryable' in error && (error as { isRetryable?: boolean }).isRetryable) return true;
        const msg = error.message.toLowerCase();
        if (msg.includes('rate limit') || msg.includes('too many requests') ||
            msg.includes('overloaded') || msg.includes('timeout') ||
            msg.includes('econnreset') || msg.includes('service unavailable')) return true;
    }
    return false;
}

/**
 * Check if a failure is provider+model-level, meaning retrying the same
 * provider+model with a different API key won't help.
 * - 400: bad request / schema incompatibility (SDK-level issue)
 * - 404: model not found on provider
 * - NoObjectGeneratedError: same content + model → same parse failure
 */
export function isProviderLevelFailure(error: unknown, aiMod: AiModule): boolean {
    const code = extractStatusCode(error, aiMod);
    if (code === 400 || code === 404) return true;
    if (error instanceof Error && aiMod.NoObjectGeneratedError.isInstance(error)) return true;
    return false;
}

/** Extract retry-after delay in seconds from an APICallError's response headers. */
export function extractRetryAfter(error: unknown, aiMod: AiModule): number | undefined {
    if (!(error instanceof Error) || !aiMod.APICallError.isInstance(error)) return undefined;
    const headers = (error as import('ai').APICallError).responseHeaders;
    if (!headers) return undefined;

    // retry-after-ms (used by OpenAI, Groq)
    const retryMs = headers['retry-after-ms'];
    if (retryMs) {
        const ms = parseFloat(retryMs);
        if (!Number.isNaN(ms)) return Math.ceil(ms / 1000);
    }

    // retry-after (standard HTTP header, in seconds or HTTP date)
    const retryAfter = headers['retry-after'];
    if (retryAfter) {
        const secs = parseFloat(retryAfter);
        if (!Number.isNaN(secs)) return Math.ceil(secs);
        const dateMs = Date.parse(retryAfter) - Date.now();
        if (!Number.isNaN(dateMs) && dateMs > 0) return Math.ceil(dateMs / 1000);
    }

    return undefined;
}
