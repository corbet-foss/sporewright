// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
import { describe, expect, it } from 'bun:test';
import {
    createRouterState,
    providerCooldownKey,
    setProviderCooldown,
    activeCooldownSeconds,
    shouldSuppressRepeat,
} from './index';

describe('RouterState cooldown', () => {
    it('keys are provider|model', () => {
        expect(providerCooldownKey('groq', 'llama')).toBe('groq|llama');
    });

    it('isolated state instances do not share cooldowns', () => {
        const a = createRouterState();
        const b = createRouterState();
        setProviderCooldown(a, 'groq', 'm', 60);
        expect(activeCooldownSeconds(a, 'groq', 'm')).toBeGreaterThan(0);
        expect(activeCooldownSeconds(b, 'groq', 'm')).toBeUndefined();
    });

    it('clamps cooldown to [30, 3600] and defaults to 300', () => {
        const s = createRouterState();
        setProviderCooldown(s, 'p', 'm', 5); // below floor → 30
        expect(activeCooldownSeconds(s, 'p', 'm')).toBeLessThanOrEqual(30);
        const s2 = createRouterState();
        setProviderCooldown(s2, 'p', 'm', undefined); // default 300
        const v = activeCooldownSeconds(s2, 'p', 'm')!;
        expect(v).toBeGreaterThan(290);
        expect(v).toBeLessThanOrEqual(300);
    });

    it('dedup suppresses repeats within the TTL window', () => {
        const s = createRouterState();
        expect(shouldSuppressRepeat(s, '429:groq')).toBe(false); // first
        expect(shouldSuppressRepeat(s, '429:groq')).toBe(true); // repeat
        expect(shouldSuppressRepeat(s, '429:other')).toBe(false); // distinct key
    });
});
