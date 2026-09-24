/**
 * ggml tensor type enum and exact block layouts.
 *
 * The numeric values below are the `enum ggml_type` values from ggml.h in the
 * ggml / llama.cpp source tree (the same enum GGUF tensor-info records use to
 * say "this tensor's data is stored as type N"). Order and IDs are stable and
 * part of the public GGUF ecosystem (they must match on-disk data or the file
 * is unreadable by any GGUF consumer, llama.cpp included).
 *
 * `blockElems`/`blockBytes` reproduce the same block_q*_K / block_q*_0 struct
 * sizes documented in quant.ts. For a tensor with `n` total elements, its
 * exact byte size on disk is `ceil(n / blockElems) * blockBytes` (GGUF pads
 * the trailing partial block up to a full block, same as ggml does in memory).
 */

export interface GgmlTypeInfo {
  name: string;
  blockElems: number;
  blockBytes: number;
}

// id -> info. Only the types actually seen in real-world published GGUF models
// are included; anything else surfaces as an explicit "unsupported tensor type"
// error rather than a silent wrong number.
export const GGML_TYPES: Record<number, GgmlTypeInfo> = {
  0: { name: "F32", blockElems: 1, blockBytes: 4 },
  1: { name: "F16", blockElems: 1, blockBytes: 2 },
  2: { name: "Q4_0", blockElems: 32, blockBytes: 18 },
  3: { name: "Q4_1", blockElems: 32, blockBytes: 20 },
  6: { name: "Q5_0", blockElems: 32, blockBytes: 22 },
  7: { name: "Q5_1", blockElems: 32, blockBytes: 24 },
  8: { name: "Q8_0", blockElems: 32, blockBytes: 34 },
  9: { name: "Q8_1", blockElems: 32, blockBytes: 40 },
  10: { name: "Q2_K", blockElems: 256, blockBytes: 84 },
  11: { name: "Q3_K", blockElems: 256, blockBytes: 110 },
  12: { name: "Q4_K", blockElems: 256, blockBytes: 144 },
  13: { name: "Q5_K", blockElems: 256, blockBytes: 176 },
  14: { name: "Q6_K", blockElems: 256, blockBytes: 210 },
  15: { name: "Q8_K", blockElems: 256, blockBytes: 292 },
  24: { name: "I8", blockElems: 1, blockBytes: 1 },
  25: { name: "I16", blockElems: 1, blockBytes: 2 },
  26: { name: "I32", blockElems: 1, blockBytes: 4 },
  28: { name: "BF16", blockElems: 1, blockBytes: 2 },
};

export function tensorByteSize(nElements: number, ggmlType: number): number {
  const info = GGML_TYPES[ggmlType];
  if (!info) {
    throw new Error(
      `Unsupported ggml tensor type id ${ggmlType} in GGUF file. ` +
        `This tool only recognizes common types (F32/F16/BF16 and Q4_0..Q8_K); ` +
        `newer or exotic quantizations (e.g. IQ*) aren't in the lookup table yet.`,
    );
  }
  const numBlocks = Math.ceil(nElements / info.blockElems);
  return numBlocks * info.blockBytes;
}
