import { describe, it, expect } from "vitest";
import { BYTES_PER_PARAM, parseParamCount, weightBytes } from "../src/quant.js";

describe("quant byte-per-param table", () => {
  it("matches the real, block-exact legacy GGML quant ratios", () => {
    // block_q4_0 = 18 bytes / 32 elements (ggml.c)
    expect(BYTES_PER_PARAM.Q4_0).toBeCloseTo(0.5625, 10);
    // block_q8_0 = 34 bytes / 32 elements (ggml.c)
    expect(BYTES_PER_PARAM.Q8_0).toBeCloseTo(1.0625, 10);
    // fp16, no blocking
    expect(BYTES_PER_PARAM.F16).toBe(2);
    expect(BYTES_PER_PARAM.F32).toBe(4);
  });

  it("matches the published k-quants bits-per-weight measurements", () => {
    // ggerganov/llama.cpp PR #1684 (k-quants), measured bpw for a 7B model
    expect(BYTES_PER_PARAM.Q4_K_M).toBeCloseTo(4.85 / 8, 10);
    expect(BYTES_PER_PARAM.Q5_K_M).toBeCloseTo(5.69 / 8, 10);
    expect(BYTES_PER_PARAM.Q2_K).toBeCloseTo(2.63 / 8, 10);
  });

  it("computes a real 7B model's Q4_K_M size within the known real-world GGUF range", () => {
    // Real published 7B Q4_K_M GGUF files (e.g. TheBloke's Llama-2-7B-Chat-GGUF,
    // Mistral-7B-Instruct-GGUF) land at ~3.9-4.4 GB, since actual files also carry
    // some F32/F16 tensors (norms, embeddings) that push slightly above the pure
    // Q4_K_M formula. Our pure-quant estimate should land just under that band.
    const bytes = weightBytes(7_000_000_000, "Q4_K_M");
    const gib = bytes / (1024 * 1024 * 1024);
    expect(bytes).toBeCloseTo(7_000_000_000 * (4.85 / 8), 5);
    expect(gib).toBeGreaterThan(3.5);
    expect(gib).toBeLessThan(4.5);
  });

  it("computes a real 7B model's Q4_0 size matching the exact block formula", () => {
    const bytes = weightBytes(7_000_000_000, "Q4_0");
    expect(bytes).toBe(7_000_000_000 * 0.5625);
    expect(bytes).toBe(3_937_500_000);
  });

  it("computes fp16 size as exactly 2 bytes/param (e.g. a 7B fp16 model is 14GB)", () => {
    expect(weightBytes(7_000_000_000, "F16")).toBe(14_000_000_000);
  });
});

describe("parseParamCount", () => {
  it("parses shorthand suffixes", () => {
    expect(parseParamCount("7B")).toBe(7_000_000_000);
    expect(parseParamCount("13b")).toBe(13_000_000_000);
    expect(parseParamCount("1.5B")).toBe(1_500_000_000);
    expect(parseParamCount("70B")).toBe(70_000_000_000);
    expect(parseParamCount("500M")).toBe(500_000_000);
  });

  it("parses a raw integer with no suffix", () => {
    expect(parseParamCount("7000000000")).toBe(7_000_000_000);
  });

  it("rejects garbage input", () => {
    expect(() => parseParamCount("banana")).toThrow();
  });
});
