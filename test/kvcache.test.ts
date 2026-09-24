import { describe, it, expect } from "vitest";
import { kvCacheBytes, maxContextForBudget } from "../src/kvcache.js";

describe("kvCacheBytes", () => {
  it("matches a hand-computed value for a small example architecture", () => {
    // Small toy architecture: 4 layers, 8 KV heads, head_dim 64, 2048 tokens, fp16 (2 bytes/elem).
    // kv_bytes = 2 * layers * kv_heads * head_dim * context * bytes_per_elem
    //          = 2 * 4 * 8 * 64 * 2048 * 2
    // Step by step: 2*4=8; 8*8=64; 64*64=4096; 4096*2048=8,388,608; *2=16,777,216
    const bytes = kvCacheBytes({
      numLayers: 4,
      numKvHeads: 8,
      headDim: 64,
      contextLength: 2048,
      bytesPerKvElement: 2,
    });
    expect(bytes).toBe(16_777_216); // exactly 16 MiB
    expect(bytes).toBe(2 * 4 * 8 * 64 * 2048 * 2);
  });

  it("defaults to fp16 (2 bytes/elem) when bytesPerKvElement is omitted", () => {
    const withDefault = kvCacheBytes({ numLayers: 4, numKvHeads: 8, headDim: 64, contextLength: 2048 });
    const explicit = kvCacheBytes({
      numLayers: 4,
      numKvHeads: 8,
      headDim: 64,
      contextLength: 2048,
      bytesPerKvElement: 2,
    });
    expect(withDefault).toBe(explicit);
  });

  it("scales linearly with context length", () => {
    const base = kvCacheBytes({ numLayers: 4, numKvHeads: 8, headDim: 64, contextLength: 1024 });
    const doubled = kvCacheBytes({ numLayers: 4, numKvHeads: 8, headDim: 64, contextLength: 2048 });
    expect(doubled).toBe(base * 2);
  });

  it("halves when using a GQA model with half as many KV heads as query heads", () => {
    const mha = kvCacheBytes({ numLayers: 4, numKvHeads: 8, headDim: 64, contextLength: 2048 });
    const gqa = kvCacheBytes({ numLayers: 4, numKvHeads: 4, headDim: 64, contextLength: 2048 });
    expect(gqa).toBe(mha / 2);
  });
});

describe("maxContextForBudget", () => {
  it("inverts kvCacheBytes exactly for the same toy architecture", () => {
    const budget = 16_777_216; // 16 MiB, matches the hand-computed case above
    const maxContext = maxContextForBudget(
      { numLayers: 4, numKvHeads: 8, headDim: 64, bytesPerKvElement: 2 },
      budget,
    );
    expect(maxContext).toBe(2048);
  });

  it("floors to a whole number of tokens when the budget doesn't divide evenly", () => {
    const perToken = 2 * 4 * 8 * 64 * 2; // 8192 bytes/token
    const budget = perToken * 2048 + 100; // a bit more than 2048 tokens' worth
    const maxContext = maxContextForBudget(
      { numLayers: 4, numKvHeads: 8, headDim: 64, bytesPerKvElement: 2 },
      budget,
    );
    expect(maxContext).toBe(2048);
  });

  it("returns 0 for a budget too small to hold even one token", () => {
    const maxContext = maxContextForBudget(
      { numLayers: 4, numKvHeads: 8, headDim: 64, bytesPerKvElement: 2 },
      100,
    );
    expect(maxContext).toBe(0);
  });
});
