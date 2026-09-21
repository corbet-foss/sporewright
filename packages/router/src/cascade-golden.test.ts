// SPDX-License-Identifier: LGPL-3.0-only WITH LGPL-3.0-linking-exception
//
// REGRESSION GUARD — golden cascade vectors for the CareerVector LLM fallback chain.
//
// Each vector pins the resolved (provider, model) order of `resolveCascadeTensor`
// for a representative CareerVector chain config. The configs are modelled on the
// real PIPELINE_STAGES seeds (lib/domain/src/machines/jobPipeline.ts): L0 stage
// defaults, L1 consumer overrides, L2 per-instance + flat per-job overrides, and
// the capability-keyed read.
//
// This is the same resolver CareerVector consumes via the `@cv/domain/cascade-tensor`
// and `@cv/domain/chain` (resolveChain) shims — both re-export `router` unchanged,
// so pinning it here pins CV's live routing decision. The goldens were captured
// from the live `resolveCascadeTensor` — never hand-fabricated. A refactor that
// silently reorders the fallback chain breaks here.

import { describe, expect, it } from "bun:test";
import { resolveCascadeTensor } from "./cascade-tensor";
import type { ChainSlot, ChainValue } from "./types";
import golden from "./cascade-golden.json";

interface CascadeVector {
  name: string;
  description?: string;
  input: {
    chains: Record<string, ChainValue>;
    stageId: string;
    consumerId: string;
    capability: string;
    instanceId?: string;
    l2Override?: ChainSlot[];
  };
  expected: ChainSlot[];
}

const vectors = golden as unknown as CascadeVector[];

describe("CareerVector LLM cascade — golden resolved provider/model order", () => {
  it("has the expected number of pinned configs", () => {
    expect(vectors.length).toBe(4);
  });

  for (const v of vectors) {
    it(`pins ${v.name}`, () => {
      const { chains, stageId, consumerId, capability, instanceId, l2Override } = v.input;
      const resolved = resolveCascadeTensor(chains, {
        workspaceId: 'golden-workspace',
        stageId,
        consumerId,
        capability,
        instanceId,
        instanceOverride: l2Override,
      });
      expect(resolved).toEqual(v.expected);
    });
  }
});
