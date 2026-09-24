import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { writeFileSync, unlinkSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  parseGgufHeader,
  totalTensorBytes,
  dominantQuantName,
  deriveArchConfig,
} from "../src/gguf.js";
import { buildGgufFile, GGML_TYPE_F32, GGML_TYPE_Q4_0 } from "./fixtures/buildGguf.js";

describe("GGUF header parser (hand-built binary fixture)", () => {
  let fitPath: string;
  let fallbackPath: string;

  beforeAll(() => {
    const dir = mkdtempSync(join(tmpdir(), "vramfit-test-"));

    // Fixture 1: a small "llama-like" architecture with an explicit head_count_kv
    // (i.e. grouped-query attention declared explicitly), one large Q4_0 weight
    // tensor and one tiny F32 norm tensor.
    const qElements = 2048 * 2048; // 4,194,304
    const qBlocks = qElements / 32; // 131,072
    const qBytes = qBlocks * 18; // 2,359,296 (real block_q4_0 = 18 bytes/32 elems)
    const fElements = 64 * 64; // 4096
    const fBytes = fElements * 4; // 16,384

    const fixture1 = buildGgufFile(
      {
        "general.architecture": { type: "string", value: "llama" },
        "general.name": { type: "string", value: "vramfit-test-model" },
        "llama.context_length": { type: "u32", value: 2048 },
        "llama.embedding_length": { type: "u32", value: 512 },
        "llama.block_count": { type: "u32", value: 4 },
        "llama.attention.head_count": { type: "u32", value: 8 },
        "llama.attention.head_count_kv": { type: "u32", value: 8 },
      },
      [
        { name: "blk.0.attn_q.weight", dims: [2048, 2048], ggmlType: GGML_TYPE_Q4_0, data: Buffer.alloc(qBytes) },
        { name: "output_norm.weight", dims: [64, 64], ggmlType: GGML_TYPE_F32, data: Buffer.alloc(fBytes) },
      ],
    );
    fitPath = join(dir, "fixture1.gguf");
    writeFileSync(fitPath, fixture1);

    // Fixture 2: omits attention.head_count_kv entirely, to test the documented
    // "absent means plain multi-head attention" fallback (headCountKv === headCount).
    const fixture2 = buildGgufFile(
      {
        "general.architecture": { type: "string", value: "llama" },
        "llama.context_length": { type: "u32", value: 4096 },
        "llama.embedding_length": { type: "u32", value: 256 },
        "llama.block_count": { type: "u32", value: 2 },
        "llama.attention.head_count": { type: "u32", value: 4 },
      },
      [],
    );
    fallbackPath = join(dir, "fixture2.gguf");
    writeFileSync(fallbackPath, fixture2);
  });

  afterAll(() => {
    unlinkSync(fitPath);
    unlinkSync(fallbackPath);
  });

  it("parses magic, version, and metadata key-value pairs correctly", () => {
    const header = parseGgufHeader(fitPath);
    expect(header.version).toBe(3);
    expect(header.metadata.get("general.architecture")).toBe("llama");
    expect(header.metadata.get("general.name")).toBe("vramfit-test-model");
    expect(header.metadata.get("llama.context_length")).toBe(2048);
    expect(header.metadata.get("llama.embedding_length")).toBe(512);
    expect(header.metadata.get("llama.block_count")).toBe(4);
    expect(header.metadata.get("llama.attention.head_count")).toBe(8);
    expect(header.metadata.get("llama.attention.head_count_kv")).toBe(8);
  });

  it("parses the tensor-info table (names, dims, types)", () => {
    const header = parseGgufHeader(fitPath);
    expect(header.tensors).toHaveLength(2);
    expect(header.tensors[0].name).toBe("blk.0.attn_q.weight");
    expect(header.tensors[0].dims).toEqual([2048, 2048]);
    expect(header.tensors[0].nElements).toBe(2048 * 2048);
    expect(header.tensors[1].name).toBe("output_norm.weight");
  });

  it("computes the exact real total tensor byte size from per-tensor ggml types", () => {
    const header = parseGgufHeader(fitPath);
    const total = totalTensorBytes(header.tensors);
    // 2,359,296 (Q4_0 weight) + 16,384 (F32 norm) = 2,375,680
    expect(total).toBe(2_359_296 + 16_384);
  });

  it("identifies the dominant quant type by total bytes, not tensor count", () => {
    const header = parseGgufHeader(fitPath);
    // The single Q4_0 tensor (2.36MB) dominates over the tiny F32 tensor (16KB)
    // even though there's one tensor of each type.
    expect(dominantQuantName(header.tensors)).toBe("Q4_0");
  });

  it("derives architecture config for KV-cache math, using the explicit head_count_kv", () => {
    const header = parseGgufHeader(fitPath);
    const cfg = deriveArchConfig(header.metadata);
    expect(cfg.architecture).toBe("llama");
    expect(cfg.contextLength).toBe(2048);
    expect(cfg.blockCount).toBe(4);
    expect(cfg.headCount).toBe(8);
    expect(cfg.headCountKv).toBe(8);
    expect(cfg.headDim).toBe(512 / 8); // embedding_length / head_count
  });

  it("falls back headCountKv to headCount when head_count_kv is absent (pre-GQA models)", () => {
    const header = parseGgufHeader(fallbackPath);
    const cfg = deriveArchConfig(header.metadata);
    expect(header.metadata.has("llama.attention.head_count_kv")).toBe(false);
    expect(cfg.headCount).toBe(4);
    expect(cfg.headCountKv).toBe(4);
    expect(cfg.headDim).toBe(256 / 4);
  });

  it("rejects a file with a bad magic number", () => {
    const dir = mkdtempSync(join(tmpdir(), "vramfit-test-bad-"));
    const badPath = join(dir, "bad.gguf");
    writeFileSync(badPath, Buffer.from("NOPE1234"));
    expect(() => parseGgufHeader(badPath)).toThrow(/GGUF/);
    unlinkSync(badPath);
  });
});
