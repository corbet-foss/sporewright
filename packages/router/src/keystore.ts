// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
/**
 * KeyStore — the host-supplied key/cascade-config seam.
 *
 * The router NEVER reads persistence, env vars, tiers, or grants. It depends only
 * on the {@link ChainConfig} a host returns through a `KeyStore`, plus the
 * resolution + execution machinery. `scopeId` carries zero host vocabulary (a
 * host passes whatever scope it routes by — a workspace id, a tenant id, …).
 */

import type { ChainConfig } from './types';

export type { ChainConfig } from './types';

export interface KeyStore {
    /** Resolve all key material + cascade chains for a routing scope. */
    get(scopeId: string, signal?: AbortSignal): Promise<ChainConfig>;
    /** Drop the cached config so the next get() re-reads fresh keys. */
    invalidate(): void;
}
