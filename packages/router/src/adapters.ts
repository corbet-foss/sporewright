// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Provider Adapters — universal base class for ALL providers.
 *
 * Two-tier hierarchy:
 *   ProviderAdapter (base)  — testKey() only. Used by service providers.
 *   LlmAdapter (subclass)   — adds createLanguageModel() + listModels().
 *
 * Every provider in PROVIDERS has an adapter in ADAPTERS.
 *
 * Types, base classes, and PASSIVE_PROVIDERS live in adapter-constants.ts so
 * that lightweight consumers (e.g. health UIs) don't pull @ai-sdk/* into
 * the initial bundle.
 */

import { createGroq } from '@ai-sdk/groq';
import { createOpenAI } from '@ai-sdk/openai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createXai } from '@ai-sdk/xai';
import { createDeepSeek } from '@ai-sdk/deepseek';
import { createMistral } from '@ai-sdk/mistral';
import { createCohere } from '@ai-sdk/cohere';
import { createTogetherAI } from '@ai-sdk/togetherai';
import { createFireworks } from '@ai-sdk/fireworks';
import { createCerebras } from '@ai-sdk/cerebras';
import { createDeepInfra } from '@ai-sdk/deepinfra';
import { createPerplexity } from '@ai-sdk/perplexity';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { createHuggingFace } from '@ai-sdk/huggingface';
import { PROVIDERS, type Capability, normalizeCreator } from './catalog';
import {
    ProviderAdapter,
    LlmAdapter,
    PASSIVE_PROVIDERS,
    type DiscoveredModel,
} from './adapter-constants';
export type {
    DiscoveredModel,
    HealthCheckStrategy,
} from './adapter-constants';
export { ProviderAdapter, LlmAdapter, PASSIVE_PROVIDERS };

// =============================================================================
// OpenAI-compatible model fetcher
// =============================================================================

async function fetchOpenAIModels(
    baseUrl: string,
    apiKey: string,
    filter?: (m: { id: string; owned_by?: string }) => boolean,
): Promise<DiscoveredModel[]> {
    const url = `${baseUrl.replace(/\/+$/, '')}/models`;
    const res = await fetch(url, {
        headers: { Authorization: `Bearer ${apiKey}` },
        signal: AbortSignal.timeout(10_000),
    });

    if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`${res.status}: ${text.slice(0, 200)}`);
    }

    const body = await res.json() as { data?: { id: string; owned_by?: string }[] };
    const models = (body.data || []).map(m => ({
        id: m.id,
        owned_by: m.owned_by,
        creator: m.owned_by ? normalizeCreator(m.owned_by) : undefined,
    }));

    return filter ? models.filter(filter) : models;
}

// =============================================================================
// LLM adapters
// =============================================================================

class GroqAdapter extends LlmAdapter {
    readonly id = 'groq';
    override createLanguageModel(apiKey: string, modelId: string) { return createGroq({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        return fetchOpenAIModels('https://api.groq.com/openai/v1', apiKey, m =>
            !m.id.includes('whisper') && !m.id.includes('guard') && !m.id.includes('embed'),
        );
    }
}

class OpenAIAdapter extends LlmAdapter {
    readonly id = 'openai';
    override createLanguageModel(apiKey: string, modelId: string) { return createOpenAI({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://api.openai.com/v1', apiKey, m =>
            m.id.startsWith('gpt-') || m.id.startsWith('o1') || m.id.startsWith('o3') || m.id.startsWith('o4') || m.id.startsWith('chatgpt-'),
        );
        return models.map(m => ({ ...m, creator: 'openai' }));
    }
}

class AnthropicAdapter extends LlmAdapter {
    readonly id = 'anthropic';
    override createLanguageModel(apiKey: string, modelId: string) { return createAnthropic({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        const models: DiscoveredModel[] = [];
        let url: string | null = 'https://api.anthropic.com/v1/models?limit=100';
        while (url) {
            const res = await fetch(url, {
                headers: { 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
                signal: AbortSignal.timeout(10_000),
            });
            if (!res.ok) {
                const text = await res.text().catch(() => '');
                throw new Error(`${res.status}: ${text.slice(0, 200)}`);
            }
            const body = await res.json() as {
                data?: { id: string; display_name?: string }[];
                has_more?: boolean;
                last_id?: string;
            };
            for (const m of body.data || []) {
                models.push({ id: m.id, name: m.display_name, owned_by: 'anthropic', creator: 'anthropic' });
            }
            if (body.has_more && body.last_id) {
                const base: URL = new URL(url);
                base.searchParams.set('after_id', body.last_id);
                url = base.toString();
            } else {
                url = null;
            }
        }
        return models;
    }
}

class GoogleAdapter extends LlmAdapter {
    readonly id = 'google';
    override createLanguageModel(apiKey: string, modelId: string) { return createGoogleGenerativeAI({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        const res = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}&pageSize=100`,
            { signal: AbortSignal.timeout(10_000) },
        );
        if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new Error(`${res.status}: ${text.slice(0, 200)}`);
        }
        const body = await res.json() as {
            models?: { name: string; displayName?: string; supportedGenerationMethods?: string[] }[];
        };
        return (body.models || [])
            .filter(m => m.supportedGenerationMethods?.includes('generateContent'))
            .map(m => ({
                id: m.name.replace(/^models\//, ''),
                name: m.displayName,
                owned_by: 'google',
                creator: 'google',
            }));
    }
}

class XaiAdapter extends LlmAdapter {
    readonly id = 'xai';
    override createLanguageModel(apiKey: string, modelId: string) { return createXai({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://api.x.ai/v1', apiKey);
        return models.map(m => ({ ...m, creator: 'xai' }));
    }
}

class DeepSeekAdapter extends LlmAdapter {
    readonly id = 'deepseek';
    override createLanguageModel(apiKey: string, modelId: string) { return createDeepSeek({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://api.deepseek.com', apiKey);
        return models.map(m => ({ ...m, creator: 'deepseek' }));
    }
}

class MistralAdapter extends LlmAdapter {
    readonly id = 'mistral';
    override createLanguageModel(apiKey: string, modelId: string) { return createMistral({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://api.mistral.ai/v1', apiKey);
        return models.map(m => ({ ...m, creator: 'mistral' }));
    }
}

class CohereAdapter extends LlmAdapter {
    readonly id = 'cohere';
    override createLanguageModel(apiKey: string, modelId: string) { return createCohere({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        const models: DiscoveredModel[] = [];
        let url: string | null = 'https://api.cohere.com/v2/models?page_size=100';
        while (url) {
            const res = await fetch(url, {
                headers: { Authorization: `Bearer ${apiKey}` },
                signal: AbortSignal.timeout(10_000),
            });
            if (!res.ok) {
                const text = await res.text().catch(() => '');
                throw new Error(`${res.status}: ${text.slice(0, 200)}`);
            }
            const body = await res.json() as {
                models?: { name: string; endpoints?: string[] }[];
                next_page_token?: string;
            };
            for (const m of body.models || []) {
                if (m.endpoints?.includes('chat')) {
                    models.push({ id: m.name, owned_by: 'cohere', creator: 'cohere' });
                }
            }
            if (body.next_page_token) {
                const base: URL = new URL(url);
                base.searchParams.set('page_token', body.next_page_token);
                url = base.toString();
            } else {
                url = null;
            }
        }
        return models;
    }
}

class TogetherAdapter extends LlmAdapter {
    readonly id = 'together';
    override createLanguageModel(apiKey: string, modelId: string) { return createTogetherAI({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://api.together.xyz/v1', apiKey, m =>
            !m.id.includes('embed') && !m.id.includes('rerank'),
        );
        return models.map(m => ({
            ...m,
            creator: m.id.includes('/') ? normalizeCreator(m.id.split('/')[0]!) : m.creator,
        }));
    }
}

class FireworksAdapter extends LlmAdapter {
    readonly id = 'fireworks';
    override createLanguageModel(apiKey: string, modelId: string) { return createFireworks({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://api.fireworks.ai/inference/v1', apiKey);
        return models.map(m => ({ ...m, creator: m.creator || 'fireworks' }));
    }
}

class CerebrasAdapter extends LlmAdapter {
    readonly id = 'cerebras';
    override createLanguageModel(apiKey: string, modelId: string) { return createCerebras({ apiKey })(modelId); }
    override async listModels(apiKey: string) { return fetchOpenAIModels('https://api.cerebras.ai/v1', apiKey); }
}

class DeepInfraAdapter extends LlmAdapter {
    readonly id = 'deepinfra';
    override createLanguageModel(apiKey: string, modelId: string) { return createDeepInfra({ apiKey })(modelId); }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://api.deepinfra.com/v1/openai', apiKey);
        return models.map(m => ({
            ...m,
            creator: m.id.includes('/') ? normalizeCreator(m.id.split('/')[0]!) : m.creator,
        }));
    }
}

class PerplexityAdapter extends LlmAdapter {
    readonly id = 'perplexity';
    override createLanguageModel(apiKey: string, modelId: string) { return createPerplexity({ apiKey })(modelId); }
    override async listModels(_apiKey: string) {
        return [
            { id: 'sonar-pro', owned_by: 'perplexity', creator: 'perplexity' },
            { id: 'sonar', owned_by: 'perplexity', creator: 'perplexity' },
            { id: 'sonar-deep-research', owned_by: 'perplexity', creator: 'perplexity' },
        ];
    }
    override async testKey(apiKey: string): Promise<{ ok: boolean; error?: string; message?: string }> {
        // Direct HTTP probe — keeps this off the `ai` SDK so a key-test surface
        // doesn't pull @ai-sdk/gateway → @vercel/oidc into the SSR bundle (the
        // top-level `createRequire(import.meta.url)` in @vercel/oidc breaks
        // Cloudflare Worker validation, code: 10021).
        try {
            const res = await fetch('https://api.perplexity.ai/chat/completions', {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${apiKey}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: 'sonar',
                    messages: [{ role: 'user', content: 'ok' }],
                    max_tokens: 1,
                }),
                signal: AbortSignal.timeout(10_000),
            });
            if (res.ok) return { ok: true };
            return { ok: false, error: `HTTP ${res.status}` };
        } catch (e) {
            return { ok: false, error: e instanceof Error ? e.message : 'Connection failed' };
        }
    }
}

class OpenRouterAdapter extends LlmAdapter {
    readonly id = 'openrouter';
    override createLanguageModel(apiKey: string, modelId: string) {
        return createOpenRouter({ apiKey })(modelId);
    }
    override async listModels(apiKey: string) {
        const res = await fetch('https://openrouter.ai/api/v1/models', {
            headers: { Authorization: `Bearer ${apiKey}` },
            signal: AbortSignal.timeout(15_000),
        });
        if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new Error(`${res.status}: ${text.slice(0, 200)}`);
        }
        const body = await res.json() as {
            data?: {
                id: string;
                name?: string;
                architecture?: { output_modalities?: string[] };
            }[];
        };
        return (body.data || [])
            .filter(m => m.architecture?.output_modalities?.includes('text'))
            .map(m => ({
                id: m.id,
                name: m.name,
                owned_by: 'openrouter',
                creator: m.id.includes('/') ? normalizeCreator(m.id.split('/')[0]!) : 'openrouter',
            }));
    }
}

// =============================================================================
// OpenAI-compatible adapters (generic base URL)
// =============================================================================

class NvidiaAdapter extends LlmAdapter {
    readonly id = 'nvidia';
    override createLanguageModel(apiKey: string, modelId: string) {
        return createOpenAICompatible({ baseURL: 'https://integrate.api.nvidia.com/v1', apiKey, name: 'nvidia' }).chatModel(modelId);
    }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://integrate.api.nvidia.com/v1', apiKey);
        return models.map(m => ({ ...m, creator: m.id.includes('/') ? normalizeCreator(m.id.split('/')[0]!) : 'nvidia' }));
    }
}

class HuggingFaceAdapter extends LlmAdapter {
    readonly id = 'huggingface';
    override createLanguageModel(apiKey: string, modelId: string) {
        return createHuggingFace({ apiKey })(modelId);
    }
    override async listModels(_apiKey: string) {
        // HuggingFace has thousands of models — return a curated list for the free tier
        return [
            { id: 'Qwen/Qwen3-8B', creator: 'qwen' },
            { id: 'Qwen/Qwen3-4B-Instruct-2507', creator: 'qwen' },
            { id: 'meta-llama/Llama-3.1-8B-Instruct', creator: 'meta' },
            { id: 'google/gemma-3n-E4B-it', creator: 'google' },
        ];
    }
    override async testKey(apiKey: string): Promise<{ ok: boolean; error?: string; message?: string }> {
        try {
            const res = await fetch('https://router.huggingface.co/v1/models', {
                headers: { Authorization: `Bearer ${apiKey}` },
                signal: AbortSignal.timeout(10_000),
            });
            return res.ok ? { ok: true } : { ok: false, error: `HTTP ${res.status}` };
        } catch (e) {
            return { ok: false, error: e instanceof Error ? e.message : 'Connection failed' };
        }
    }
}

class CloudflareAiAdapter extends LlmAdapter {
    readonly id = 'cloudflareai';
    /** Parse "accountId:apiToken" key format. */
    private parseKey(apiKey: string) {
        const sep = apiKey.indexOf(':');
        if (sep <= 0) throw new Error('Cloudflare AI key must be "accountId:apiToken"');
        return { accountId: apiKey.slice(0, sep), token: apiKey.slice(sep + 1) };
    }
    override createLanguageModel(apiKey: string, modelId: string) {
        const { accountId, token } = this.parseKey(apiKey);
        return createOpenAICompatible({
            baseURL: `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1`,
            apiKey: token,
            name: 'cloudflareai',
        }).chatModel(modelId);
    }
    override async listModels(apiKey: string) {
        const { accountId, token } = this.parseKey(apiKey);
        const res = await fetch(`https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/models/search?task=Text Generation`, {
            headers: { Authorization: `Bearer ${token}` },
            signal: AbortSignal.timeout(10_000),
        });
        if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new Error(`${res.status}: ${text.slice(0, 200)}`);
        }
        const body = await res.json() as { result?: { name: string; description?: string }[] };
        return (body.result || []).map(m => ({
            id: m.name,
            name: m.description,
            creator: m.name.includes('/') ? normalizeCreator(m.name.split('/')[1] || m.name.split('/')[0]!) : 'cloudflare',
        }));
    }
    override async testKey(apiKey: string): Promise<{ ok: boolean; error?: string; message?: string }> {
        try {
            const { accountId, token } = this.parseKey(apiKey);
            const res = await fetch(`https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/models/search?task=Text Generation&per_page=1`, {
                headers: { Authorization: `Bearer ${token}` },
                signal: AbortSignal.timeout(10_000),
            });
            return res.ok ? { ok: true } : { ok: false, error: `HTTP ${res.status}` };
        } catch (e) {
            return { ok: false, error: e instanceof Error ? e.message : 'Connection failed' };
        }
    }
}

class CodestralAdapter extends LlmAdapter {
    readonly id = 'codestral';
    override createLanguageModel(apiKey: string, modelId: string) {
        return createMistral({ apiKey, baseURL: 'https://codestral.mistral.ai/v1' })(modelId);
    }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://codestral.mistral.ai/v1', apiKey);
        return models.map(m => ({ ...m, creator: 'mistral' }));
    }
}

class OpenCodeZenAdapter extends LlmAdapter {
    readonly id = 'opencodezen';
    override createLanguageModel(apiKey: string, modelId: string) {
        return createOpenAICompatible({ baseURL: 'https://opencode.ai/zen/v1', apiKey, name: 'opencodezen' }).chatModel(modelId);
    }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://opencode.ai/zen/v1', apiKey);
        return models.map(m => ({ ...m, creator: normalizeCreator(m.owned_by || m.id.split('/')[0] || 'opencode') }));
    }
}

class ZaiAdapter extends LlmAdapter {
    readonly id = 'zai';
    override createLanguageModel(apiKey: string, modelId: string) {
        return createOpenAICompatible({ baseURL: 'https://api.z.ai/api/paas/v4', apiKey, name: 'zai' }).chatModel(modelId);
    }
    override async listModels(apiKey: string) {
        const models = await fetchOpenAIModels('https://api.z.ai/api/paas/v4', apiKey);
        return models.map(m => ({ ...m, creator: 'zhipu' }));
    }
}

// =============================================================================
// Service adapters
// =============================================================================

class JinaAdapter extends ProviderAdapter {
    readonly id = 'jina';
    override readonly healthCheckStrategy = 'passive' as const;
    override async listModels() {
        return [{ id: 'reader', name: 'Jina Reader', capability: 'scrape' as Capability, creator: 'jina' }];
    }
    override async testKey(_apiKey: string) {
        // Jina charges per scrape — no free validation endpoint.
        return { ok: true, message: 'Jina key configured (passive monitoring)' };
    }
}

class TinyFishAdapter extends ProviderAdapter {
    readonly id = 'tinyfish';
    override async listModels() {
        return [{ id: 'fetch', name: 'Fetch API', capability: 'scrape' as Capability, creator: 'tinyfish' }];
    }
    override async testKey(apiKey: string) {
        try {
            const res = await fetch('https://api.fetch.tinyfish.ai/usage?limit=1', {
                headers: { 'X-API-Key': apiKey },
                signal: AbortSignal.timeout(10_000),
            });
            return res.ok
                ? { ok: true, message: 'TinyFish Fetch OK' }
                : { ok: false, error: `TinyFish returned ${res.status}` };
        } catch (e) {
            return { ok: false, error: e instanceof Error ? e.message : 'Connection failed' };
        }
    }
}

class TavilyAdapter extends ProviderAdapter {
    readonly id = 'tavily';
    override async listModels() {
        return [{ id: 'extract', name: 'Extract API', capability: 'scrape' as Capability, creator: 'tavily' }];
    }
    override async testKey(apiKey: string) {
        try {
            const res = await fetch('https://api.tavily.com/usage', {
                headers: { Authorization: `Bearer ${apiKey}` },
                signal: AbortSignal.timeout(10_000),
            });
            if (res.ok) {
                const data = await res.json().catch(() => null) as { key?: { usage?: number; limit?: number } } | null;
                const usage = data?.key?.usage;
                const limit = data?.key?.limit;
                return {
                    ok: true,
                    message: typeof usage === 'number' && typeof limit === 'number'
                        ? `Tavily: ${usage.toLocaleString()}/${limit.toLocaleString()} credits`
                        : 'Tavily OK',
                };
            }
            return { ok: false, error: `Tavily returned ${res.status}` };
        } catch (e) {
            return { ok: false, error: e instanceof Error ? e.message : 'Connection failed' };
        }
    }
}

class MapsAdapter extends ProviderAdapter {
    readonly id = 'maps';
    override readonly healthCheckStrategy = 'passive' as const;
    override async listModels() {
        return [{ id: 'directions', name: 'Routes API', capability: 'route' as Capability, creator: 'google' }];
    }
    override async testKey(_apiKey: string) {
        // Every Maps API call is billable — no free validation endpoint.
        return { ok: true, message: 'Maps key configured (passive monitoring)' };
    }
}

class SerpApiAdapter extends ProviderAdapter {
    readonly id = 'serpapi';
    override async listModels() {
        return [{ id: 'google-jobs', name: 'Google Jobs Search', capability: 'search' as Capability, creator: 'serpapi' }];
    }
    override async testKey(apiKey: string) {
        try {
            const res = await fetch(`https://serpapi.com/account.json?api_key=${apiKey}`, {
                signal: AbortSignal.timeout(10_000),
            });
            return res.ok
                ? { ok: true, message: 'SerpApi OK' }
                : { ok: false, error: `SerpApi returned ${res.status}` };
        } catch (e) {
            return { ok: false, error: e instanceof Error ? e.message : 'Connection failed' };
        }
    }
}

class DeepLAdapter extends ProviderAdapter {
    readonly id = 'deepl';
    override async listModels() {
        return [{ id: 'translator', name: 'DeepL Translator', capability: 'translate' as Capability, creator: 'deepl' }];
    }
    override async testKey(apiKey: string) {
        try {
            const baseUrl = PROVIDERS.deepl!.resolveBaseUrl!(apiKey);
            const res = await fetch(`${baseUrl}/v2/usage`, {
                headers: { 'Authorization': `DeepL-Auth-Key ${apiKey}` },
                signal: AbortSignal.timeout(10_000),
            });
            if (res.ok) {
                const data = await res.json() as { character_count?: number; character_limit?: number };
                return { ok: true, message: `DeepL: ${(data.character_count ?? 0).toLocaleString()}/${(data.character_limit ?? 0).toLocaleString()} chars` };
            }
            return { ok: false, error: `DeepL returned ${res.status}` };
        } catch (e) {
            return { ok: false, error: e instanceof Error ? e.message : 'Connection failed' };
        }
    }
}

// =============================================================================
// Registry
// =============================================================================

/** All adapter instances, indexed by provider ID. Every provider in PROVIDERS has one. */
export const ADAPTERS: Record<string, ProviderAdapter> = {
    // LLM
    groq: new GroqAdapter(),
    openai: new OpenAIAdapter(),
    anthropic: new AnthropicAdapter(),
    google: new GoogleAdapter(),
    xai: new XaiAdapter(),
    deepseek: new DeepSeekAdapter(),
    mistral: new MistralAdapter(),
    cohere: new CohereAdapter(),
    together: new TogetherAdapter(),
    fireworks: new FireworksAdapter(),
    cerebras: new CerebrasAdapter(),
    deepinfra: new DeepInfraAdapter(),
    perplexity: new PerplexityAdapter(),
    openrouter: new OpenRouterAdapter(),
    nvidia: new NvidiaAdapter(),
    huggingface: new HuggingFaceAdapter(),
    cloudflareai: new CloudflareAiAdapter(),
    codestral: new CodestralAdapter(),
    opencodezen: new OpenCodeZenAdapter(),
    zai: new ZaiAdapter(),
    // Service
    jina: new JinaAdapter(),
    tinyfish: new TinyFishAdapter(),
    tavily: new TavilyAdapter(),
    maps: new MapsAdapter(),
    serpapi: new SerpApiAdapter(),
    deepl: new DeepLAdapter(),
};

/** Get the adapter for a provider ID. Throws for unknown providers. */
export function getAdapter(providerId: string): ProviderAdapter {
    const adapter = ADAPTERS[providerId];
    if (!adapter) throw new Error(`Unknown provider: "${providerId}"`);
    return adapter;
}

/** Get the LLM adapter for a provider ID. Throws if not an LLM provider. */
export function getLlmAdapter(providerId: string): LlmAdapter {
    const adapter = getAdapter(providerId);
    if (!(adapter instanceof LlmAdapter)) throw new Error(`"${providerId}" is not an LLM provider`);
    return adapter;
}

/** Type guard: is this adapter an LLM adapter? */
export function isLlmAdapter(adapter: ProviderAdapter): adapter is LlmAdapter {
    return adapter instanceof LlmAdapter;
}
