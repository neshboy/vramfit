/**
 * Quantization sizing table.
 *
 * These are the real, block-exact byte-per-element ratios used by GGML/llama.cpp
 * quantization formats, derived from the actual block layouts defined in
 * ggml's source (ggml.c / ggml-common.h), which every GGUF file on disk follows.
 * A "block" is a fixed group of elements (usually 32, or 256 for k-quants) that
 * shares one or two float16 scale/min values, followed by the packed low-bit
 * weight data. bytes-per-element = blockSizeBytes / elementsPerBlock.
 *
 * Legacy (non-k) quant block layouts (block of QK=32 elements):
 *   F32   : 4 bytes/elem                          (no blocking, raw float32)
 *   F16   : 2 bytes/elem                          (no blocking, raw float16)
 *   Q8_0  : 1 fp16 scale (2B) + 32x int8 (32B)   = 34B / 32 = 1.0625 B/elem
 *   Q5_1  : 1 fp16 d (2B) + 1 fp16 m (2B) + 4B qh + 16B ql = 24B / 32 = 0.75 B/elem
 *   Q5_0  : 1 fp16 d (2B) + 4B qh + 16B ql        = 22B / 32 = 0.6875 B/elem
 *   Q4_1  : 1 fp16 d (2B) + 1 fp16 m (2B) + 16B qs = 20B / 32 = 0.625 B/elem
 *   Q4_0  : 1 fp16 d (2B) + 16B qs (4-bit packed)  = 18B / 32 = 0.5625 B/elem
 *
 * K-quant block layouts (block of QK_K=256 elements), from ggml's
 * block_q{2,3,4,5,6}_K structs:
 *   Q6_K  : ql(128B) + qh(64B) + scales(16B) + d(2B)              = 210B/256 = 0.8203125 B/elem
 *   Q5_K  : d(2B)+dmin(2B)+scales(12B)+qh(32B)+qs(128B)           = 176B/256 = 0.6875   B/elem
 *   Q4_K  : d(2B)+dmin(2B)+scales(12B)+qs(128B)                   = 144B/256 = 0.5625   B/elem
 *   Q3_K  : hmask(32B)+qs(64B)+scales(12B)+d(2B)                  = 110B/256 = 0.4296875 B/elem
 *   Q2_K  : qs(64B)+scales(16B)+d(2B)+dmin(2B)                    = 84B/256  = 0.328125 B/elem
 *
 * The "_S" / "_M" / "_L" suffixes (e.g. Q4_K_M) do NOT change the block layout
 * above -- llama.cpp's quantize.cpp instead mixes block types *per tensor*
 * (e.g. attention.wv / feed_forward.w2 get bumped to a higher-precision block
 * type while everything else uses the base type), which raises the *average*
 * bits-per-weight for the whole model above the pure block value. The
 * following average bits-per-weight (bpw) figures for the "_S/_M/_L" variants
 * are the real, published measurements from the k-quants PR
 * (ggerganov/llama.cpp #1684, "k-quants", ikawrakow, June 2023) run against a
 * 7B LLaMA model, and are the numbers the llama.cpp project and community
 * still cite for these formats:
 *   Q2_K   : 2.63 bpw
 *   Q3_K_S : 3.35 bpw   Q3_K_M : 3.55 bpw   Q3_K_L : 3.76 bpw
 *   Q4_K_S : 4.58 bpw   Q4_K_M : 4.85 bpw
 *   Q5_K_S : 5.54 bpw   Q5_K_M : 5.69 bpw
 *   Q6_K   : 6.59 bpw
 * bytes/elem = bpw / 8.
 *
 * This module only estimates from a raw parameter count (manual mode). When a
 * real GGUF file is supplied, gguf.ts instead sums the *actual* per-tensor
 * ggml type of every tensor in the file (see ggmlType.ts), which is exact
 * rather than an average approximation.
 */

export type QuantLevel =
  | "F32"
  | "F16"
  | "Q8_0"
  | "Q6_K"
  | "Q5_1"
  | "Q5_0"
  | "Q5_K_M"
  | "Q5_K_S"
  | "Q4_1"
  | "Q4_0"
  | "Q4_K_M"
  | "Q4_K_S"
  | "Q3_K_L"
  | "Q3_K_M"
  | "Q3_K_S"
  | "Q2_K";

/** Bytes per parameter for each supported quantization level. See file header for sources. */
export const BYTES_PER_PARAM: Record<QuantLevel, number> = {
  F32: 4,
  F16: 2,
  Q8_0: 34 / 32, // 1.0625
  Q6_K: 6.59 / 8, // 0.82375 (measured bpw, k-quants PR)
  Q5_1: 24 / 32, // 0.75
  Q5_0: 22 / 32, // 0.6875
  Q5_K_M: 5.69 / 8, // 0.71125
  Q5_K_S: 5.54 / 8, // 0.6925
  Q4_1: 20 / 32, // 0.625
  Q4_0: 18 / 32, // 0.5625
  Q4_K_M: 4.85 / 8, // 0.60625
  Q4_K_S: 4.58 / 8, // 0.5725
  Q3_K_L: 3.76 / 8, // 0.47
  Q3_K_M: 3.55 / 8, // 0.44375
  Q3_K_S: 3.35 / 8, // 0.41875
  Q2_K: 2.63 / 8, // 0.32875
};

export const QUANT_LEVELS = Object.keys(BYTES_PER_PARAM) as QuantLevel[];

export function isQuantLevel(value: string): value is QuantLevel {
  return Object.prototype.hasOwnProperty.call(BYTES_PER_PARAM, value);
}

/**
 * Parse a human parameter-count shorthand like "7B", "13b", "1.5B", "70000000000"
 * into a raw integer parameter count.
 */
export function parseParamCount(input: string): number {
  const trimmed = input.trim();
  const match = /^([0-9]*\.?[0-9]+)\s*([bBmMkK]?)$/.exec(trimmed);
  if (!match) {
    throw new Error(
      `Invalid parameter count "${input}". Expected e.g. "7B", "13B", "1.5B", or a raw integer.`,
    );
  }
  const value = parseFloat(match[1]);
  const suffix = match[2].toLowerCase();
  const multiplier = suffix === "b" ? 1e9 : suffix === "m" ? 1e6 : suffix === "k" ? 1e3 : 1;
  return Math.round(value * multiplier);
}

/** Total weight bytes = paramCount * bytesPerParam(quant). */
export function weightBytes(paramCount: number, quant: QuantLevel): number {
  return paramCount * BYTES_PER_PARAM[quant];
}
