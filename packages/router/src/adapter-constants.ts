// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Provider Adapter — constants and base types.
 *
 * This file contains ONLY types, abstract base classes, and constants that
 * do NOT import any @ai-sdk/* packages. It exists so that lightweight
 * consumers (e.g. health UIs) can import PASSIVE_PROVIDERS without pulling
 * the entire AI SDK tree into the initial bundle.
 *
 * Full provider implementations live in adapters.ts (imports @ai-sdk/*).
 */

import type { LanguageModel } from 'ai';
import type { Capability } from './catalog';

// =============================================================================
// Types
// =============================================================================

export interface DiscoveredModel {
    id: string;
    name?: string;
    owned_by?: string;
    /** Normalized creator slug (e.g., "anthropic", "meta"). Set by each adapter. */
    creator?: string;
    /** What this model can do. Set by adapter; defaults to provider's capability. */
    capability?: Capability;
}

// =============================================================================
// Base class — all providers
// =============================================================================

/**
 * Health check strategy:
 * - 'active':  has a free validation endpoint. Safe to poll on interval.
 * - 'passive': every API call costs money. Track health from real usage only.
 *              testKey() returns { ok: true } without making any call.
 */
export type HealthCheckStrategy = 'active' | 'passive';

export abstract class ProviderAdapter {
    abstract readonly id: string;

    /** How this provider should be health-checked. Default: 'active'. */
    readonly healthCheckStrategy: HealthCheckStrategy = 'active';

    /** Validate an API key. Must be implemented by every adapter. */
    abstract testKey(apiKey: string): Promise<{ ok: boolean; error?: string; message?: string }>;

    /** Discover models available with this API key. Base returns empty; overridden by LLM and service adapters. */
    async listModels(_apiKey: string): Promise<DiscoveredModel[]> { return []; }
}

// =============================================================================
// LLM adapter — extends base with model creation + discovery
// =============================================================================

export abstract class LlmAdapter extends ProviderAdapter {
    /** Create an AI SDK LanguageModel for inference. */
    abstract createLanguageModel(apiKey: string, modelId: string): LanguageModel;

    /** Discover models available with this API key. */
    abstract override listModels(apiKey: string): Promise<DiscoveredModel[]>;

    /** Default: try listModels, return ok if any models found. */
    override async testKey(apiKey: string): Promise<{ ok: boolean; error?: string; message?: string }> {
        try {
            const models = await this.listModels(apiKey);
            if (models.length > 0) return { ok: true };
            return { ok: false, error: 'No models returned' };
        } catch (e) {
            return { ok: false, error: e instanceof Error ? e.message : 'Unknown error' };
        }
    }
}

// =============================================================================
// Constants — no SDK imports below this line
// =============================================================================

/**
 * Provider IDs that use passive health monitoring (no active polling).
 *
 * These providers charge per API call and have no free validation endpoint.
 * Health is inferred from real usage results, not periodic probes.
 *
 * Keep in sync with adapters that set healthCheckStrategy = 'passive' in adapters.ts.
 */
export const PASSIVE_PROVIDERS = new Set<string>(['jina', 'maps']);
