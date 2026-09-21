// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Model factory + lazy module caches.
 *
 * The adapter module imports every @ai-sdk/* provider (the heavy module) and the
 * base `ai` package are dynamically imported here, not at module parse time, so
 * consumers that never make an LLM call don't pay the parse cost. The dynamic
 * import is also the bundler code-split boundary (Vite/wrangler/esbuild).
 */

import type { LanguageModel } from 'ai';

// The adapter module imports every @ai-sdk/* provider. Dynamic import here is the
// bundler code-split boundary.
let _adapterModule: typeof import('./adapters') | null = null;
export async function getAdapterModule() {
    if (!_adapterModule) _adapterModule = await import('./adapters');
    return _adapterModule;
}

// The base 'ai' package (APICallError, NoObjectGeneratedError) is only needed
// inside callWithChain(), which is itself only called when LLM calls fire.
let _aiModule: typeof import('ai') | null = null;
export async function getAiModule() {
    if (!_aiModule) _aiModule = await import('ai');
    return _aiModule;
}

/**
 * Create an AI SDK language model instance for any supported provider.
 * Delegates to the ProviderAdapter registered for this provider ID.
 *
 * Async because it lazy-loads the adapter module on first call (code-split boundary).
 */
export async function createModel(
    providerId: string,
    apiKey: string,
    model: string,
): Promise<LanguageModel> {
    if (!model) throw new Error(`No model specified for provider "${providerId}". Configure in Models tab.`);
    const { getLlmAdapter } = await getAdapterModule();
    return getLlmAdapter(providerId).createLanguageModel(apiKey, model);
}
