// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Provider catalog — the single source of truth for provider metadata, plus the
 * creator-normalization helper the adapters depend on.
 *
 * Covers all providers: LLM (chat), scraping, routing, search, translation.
 * Model lists come from dynamic discovery via ProviderAdapter.listModels().
 * Capability lives on the model (set by adapters), but each provider entry
 * declares its primary capability for UI grouping and filtering.
 */

/** What a provider's models can do. */
export type Capability = 'chat' | 'scrape' | 'route' | 'search' | 'translate';

export interface ProviderMeta {
  name: string;
  /** Primary capability of this provider's models. */
  capability: Capability;
  /** Env var name for dev-mode key resolution (e.g. 'GROQ_API_KEY'). */
  envVar: string;
  signupUrl: string;
  description: string;
  /** Route executes on the participant and does not accept an API key. */
  local?: boolean;
  /** Derive base URL from key content. For providers with tier-specific endpoints (e.g. DeepL free vs pro). */
  resolveBaseUrl?: (key: string) => string;
}

/**
 * All providers, LLM and service. Insertion order determines UI display order.
 * Providers with tier-specific endpoints (e.g. DeepL free/pro) declare a
 * resolveBaseUrl function to derive the correct endpoint from the key.
 */
export const PROVIDERS: Record<string, ProviderMeta> = {
  // LLM providers — order: free tiers first, then trial, then paid
  webllm: {
    name: 'Local AI',
    capability: 'chat',
    envVar: '',
    signupUrl: '',
    description: 'Private, free AI running on this device with WebGPU.',
    local: true,
  },
  groq: {
    name: 'Groq',
    capability: 'chat',
    envVar: 'GROQ_API_KEY',
    signupUrl: 'https://console.groq.com',
    description: 'Free tier with generous limits. Fast inference.',
  },
  openai: {
    name: 'OpenAI',
    capability: 'chat',
    envVar: 'OPENAI_API_KEY',
    signupUrl: 'https://platform.openai.com/api-keys',
    description: 'Most capable models. Pay-per-use.',
  },
  anthropic: {
    name: 'Anthropic',
    capability: 'chat',
    envVar: 'ANTHROPIC_API_KEY',
    signupUrl: 'https://console.anthropic.com',
    description: 'Claude models. Strong reasoning and structured output.',
  },
  google: {
    name: 'Google Gemini',
    capability: 'chat',
    envVar: 'GOOGLE_GENERATIVE_AI_API_KEY',
    signupUrl: 'https://aistudio.google.com/apikey',
    description: 'Generous free tier. Massive context windows.',
  },
  xai: {
    name: 'xAI',
    capability: 'chat',
    envVar: 'XAI_API_KEY',
    signupUrl: 'https://console.x.ai',
    description: 'Grok models from xAI.',
  },
  deepseek: {
    name: 'DeepSeek',
    capability: 'chat',
    envVar: 'DEEPSEEK_API_KEY',
    signupUrl: 'https://platform.deepseek.com/api_keys',
    description: 'Strong open models. Very competitive pricing.',
  },
  mistral: {
    name: 'Mistral AI',
    capability: 'chat',
    envVar: 'MISTRAL_API_KEY',
    signupUrl: 'https://console.mistral.ai/api-keys',
    description: 'European AI. Strong multilingual support.',
  },
  cohere: {
    name: 'Cohere',
    capability: 'chat',
    envVar: 'COHERE_API_KEY',
    signupUrl: 'https://dashboard.cohere.com/api-keys',
    description: 'Enterprise-grade. Strong multilingual RAG.',
  },
  together: {
    name: 'Together AI',
    capability: 'chat',
    envVar: 'TOGETHER_API_KEY',
    signupUrl: 'https://api.together.xyz/settings/api-keys',
    description: 'Wide model selection. Free trial credits.',
  },
  fireworks: {
    name: 'Fireworks AI',
    capability: 'chat',
    envVar: 'FIREWORKS_API_KEY',
    signupUrl: 'https://fireworks.ai/api-keys',
    description: 'Fast inference with competitive pricing.',
  },
  cerebras: {
    name: 'Cerebras',
    capability: 'chat',
    envVar: 'CEREBRAS_API_KEY',
    signupUrl: 'https://cloud.cerebras.ai',
    description: 'Fastest inference. Generous free tier (1M tokens/day).',
  },
  deepinfra: {
    name: 'DeepInfra',
    capability: 'chat',
    envVar: 'DEEPINFRA_API_KEY',
    signupUrl: 'https://deepinfra.com/dash/api_keys',
    description: 'Hosts many open models. Low pricing.',
  },
  perplexity: {
    name: 'Perplexity',
    capability: 'chat',
    envVar: 'PERPLEXITY_API_KEY',
    signupUrl: 'https://www.perplexity.ai/settings/api',
    description: 'Web-search augmented models.',
  },
  openrouter: {
    name: 'OpenRouter',
    capability: 'chat',
    envVar: 'OPENROUTER_API_KEY',
    signupUrl: 'https://openrouter.ai/keys',
    description: 'Access 300+ models from all major providers through one API.',
  },
  nvidia: {
    name: 'NVIDIA NIM',
    capability: 'chat',
    envVar: 'NVIDIA_API_KEY',
    signupUrl: 'https://build.nvidia.com',
    description: 'No daily cap. 189 models including Gemma 4, Llama 3.3, DeepSeek.',
  },
  huggingface: {
    name: 'HuggingFace',
    capability: 'chat',
    envVar: 'HUGGINGFACE_API_KEY',
    signupUrl: 'https://huggingface.co/settings/tokens',
    description: 'Thousands of open models. $0.10/month free credits.',
  },
  cloudflareai: {
    name: 'Cloudflare Workers AI',
    capability: 'chat',
    envVar: 'CLOUDFLARE_AI_API_KEY',
    signupUrl: 'https://dash.cloudflare.com/profile/api-tokens',
    description: '10K neurons/day free. Small models for fast structured output.',
    resolveBaseUrl: (key) => {
      const sep = key.indexOf(':');
      const accountId = sep > 0 ? key.slice(0, sep) : key;
      return `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1`;
    },
  },
  codestral: {
    name: 'Mistral Codestral',
    capability: 'chat',
    envVar: 'CODESTRAL_API_KEY',
    signupUrl: 'https://console.mistral.ai/codestral',
    description: 'Free coding model. 2K req/day. Separate budget from Mistral.',
  },
  opencodezen: {
    name: 'OpenCode Zen',
    capability: 'chat',
    envVar: 'OPENCODE_ZEN_API_KEY',
    signupUrl: 'https://opencode.ai/zen',
    description: 'Free coding-optimized models. OpenAI-compatible gateway.',
  },
  zai: {
    name: 'Z.AI (Zhipu)',
    capability: 'chat',
    envVar: 'ZAI_API_KEY',
    signupUrl: 'https://z.ai/model-api',
    description: 'GLM models from Zhipu AI. OpenAI-compatible.',
  },

  // Service providers — each has a single capability
  jina: {
    name: 'Jina Reader',
    capability: 'scrape',
    envVar: 'JINA_API_KEY',
    signupUrl: 'https://jina.ai/api-key',
    description: 'Web scraping for importing jobs from URLs. Works without a key but rate-limits aggressively.',
  },
  tinyfish: {
    name: 'TinyFish Fetch',
    capability: 'scrape',
    envVar: 'TINYFISH_API_KEY',
    signupUrl: 'https://agent.tinyfish.ai/api-keys',
    description: 'Browser-rendered web fetching for job URLs. Fetch is free and works well on JavaScript-heavy pages.',
  },
  tavily: {
    name: 'Tavily Extract',
    capability: 'scrape',
    envVar: 'TAVILY_API_KEY',
    signupUrl: 'https://app.tavily.com',
    description: 'Credit-based web extraction fallback for hard-to-read job postings.',
  },
  maps: {
    name: 'Google Maps',
    capability: 'route',
    envVar: 'GOOGLE_MAPS_API_KEY',
    signupUrl: 'https://console.cloud.google.com/apis/credentials',
    description: 'Commute time calculations. Enable Routes API, Geocoding API, and Places API.',
  },
  serpapi: {
    name: 'SerpApi (Google Jobs)',
    capability: 'search',
    envVar: 'SERPAPI_API_KEY',
    signupUrl: 'https://serpapi.com/manage-api-key',
    description: 'Job search aggregator for Radar. 250 free searches/month.',
  },
  deepl: {
    name: 'DeepL',
    capability: 'translate',
    envVar: 'DEEPL_API_KEY',
    signupUrl: 'https://www.deepl.com/pro-api',
    description: 'Translates location names across CV/CL languages. 500k chars/month free.',
    resolveBaseUrl: (key) => key.endsWith(':fx') ? 'https://api-free.deepl.com' : 'https://api.deepl.com',
  },
};

/** Provider IDs that have chat capability (LLM providers). */
export const CHAT_PROVIDER_IDS = new Set(
    Object.entries(PROVIDERS).filter(([, m]) => m.capability === 'chat').map(([id]) => id),
);

/** Check if a provider ID has chat capability (i.e., is an LLM provider). */
export function isChatProvider(pid: string): boolean {
    return CHAT_PROVIDER_IDS.has(pid);
}

/** Get provider IDs with a specific capability. */
export function providersWithCapability(cap: Capability): string[] {
    return Object.entries(PROVIDERS).filter(([, m]) => m.capability === cap).map(([id]) => id);
}

// =============================================================================
// Creator normalization — the single piece of model-catalog the adapters need
// =============================================================================

/**
 * Map raw creator strings to normalized slugs. Only non-identity mappings —
 * unrecognized inputs fall through to the lowercased raw value.
 */
const CREATOR_ALIASES: Record<string, string> = {
    'meta-llama': 'meta',
    'facebook': 'meta',
    'google-deepmind': 'google',
    'mistralai': 'mistral',
    'mistral-ai': 'mistral',
    'deepseek-ai': 'deepseek',
    'x-ai': 'xai',
    'zhipuai': 'zhipu',
    'zai': 'zhipu',
    'alibaba': 'qwen',
};

export function normalizeCreator(raw: string): string {
    const lower = raw.toLowerCase().trim();
    return CREATOR_ALIASES[lower] || lower;
}
