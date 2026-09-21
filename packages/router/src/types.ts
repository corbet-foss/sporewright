// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * Local generic value types for the cascade.
 *
 * These are plain structural TS interfaces that mirror the Zod-inferred shapes a
 * host (e.g. a persisted cloud-keys schema) produces, so host values pass straight
 * through by structural typing. The router never owns the storage envelope, key
 * caps, or seed versioning — those stay host-side; the router only needs the wire
 * shape it resolves over.
 */

/** One routing option: a (provider, model) pair. */
export interface ChainSlot {
    provider: string;
    model: string;
    /** Human policy for this declaration; hard authority never enters belief. */
    routing?: 'auto' | 'prefer' | 'fixed' | 'never';
}

/** A chain value is capability-keyed: capability -> ordered slots. */
export type ChainValue = Record<string, ChainSlot[]>;

/** Per-provider multi-key usage strategy. */
export type ProviderKeyUsage = 'ordered' | 'balanced';

/**
 * The cascade config the host resolves over. This is the wire shape returned by
 * a {@link KeyStore}, NOT a persistence schema.
 */
export interface ChainConfig {
    /** Canonical owner of the workspace-scoped tensor branch. */
    workspaceId: string;
    /** provider id -> ordered API keys */
    providerKeys: Record<string, string[]>;
    /** Host-approved, participant-local routes which do not use an API key. */
    credentiallessProviders?: string[];
    /** provider id -> multi-key usage strategy. Missing provider defaults to ordered. */
    keyUsage?: Record<string, ProviderKeyUsage>;
    /** chainKey -> capability -> ChainSlot[] */
    chains: Record<string, ChainValue>;
    /** Optional learned field supplied by the CareerVector tensor host. */
    routingState?: {
        tensorRevision: string;
        snapshot: string;
        explorationTemperature?: number;
        exploration?: {
            maxExecutions: number;
            reservationUnits: number;
            accountId: string;
            fundingKind: 'byok' | 'platform';
        };
    };
}
