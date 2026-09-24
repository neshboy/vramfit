/**
 * Hand-builds a minimal, byte-for-byte valid GGUF file in memory for tests,
 * the same "construct real bytes matching the real spec" approach used by
 * this portfolio's modelsbom project for its own GGUF test fixtures --
 * written independently here, not copied, but following the same public
 * GGUF layout documented in src/gguf.ts's header comment.
 *
 * We only need a handful of metadata keys plus a couple of tiny tensors to
 * exercise the parser end-to-end: header -> metadata KV -> tensor-info table.
 */

type MetaValue =
  | { type: "u32"; value: number }
  | { type: "i32"; value: number }
  | { type: "f32"; value: number }
  | { type: "string"; value: string }
  | { type: "bool"; value: boolean }
  | { type: "u64"; value: bigint };

const GGUF_VALUE_TYPE = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
} as const;

export const GGML_TYPE_F32 = 0;
export const GGML_TYPE_F16 = 1;
export const GGML_TYPE_Q4_0 = 2;

export interface FixtureTensor {
  name: string;
  dims: number[];
  ggmlType: number;
  /** raw bytes to place in the (unused-by-header-only-parser) data section; length must match the real on-disk size */
  data: Buffer;
}

function u32le(n: number): Buffer {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(n, 0);
  return b;
}
function u64le(n: bigint | number): Buffer {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(n), 0);
  return b;
}
function i32le(n: number): Buffer {
  const b = Buffer.alloc(4);
  b.writeInt32LE(n, 0);
  return b;
}
function f32le(n: number): Buffer {
  const b = Buffer.alloc(4);
  b.writeFloatLE(n, 0);
  return b;
}
function ggufString(s: string): Buffer {
  const strBytes = Buffer.from(s, "utf8");
  return Buffer.concat([u64le(strBytes.length), strBytes]);
}

function encodeMetaValue(v: MetaValue): Buffer {
  switch (v.type) {
    case "u32":
      return Buffer.concat([u32le(GGUF_VALUE_TYPE.UINT32), u32le(v.value)]);
    case "i32":
      return Buffer.concat([u32le(GGUF_VALUE_TYPE.INT32), i32le(v.value)]);
    case "f32":
      return Buffer.concat([u32le(GGUF_VALUE_TYPE.FLOAT32), f32le(v.value)]);
    case "string":
      return Buffer.concat([u32le(GGUF_VALUE_TYPE.STRING), ggufString(v.value)]);
    case "bool": {
      const b = Buffer.alloc(1);
      b.writeUInt8(v.value ? 1 : 0, 0);
      return Buffer.concat([u32le(GGUF_VALUE_TYPE.BOOL), b]);
    }
    case "u64":
      return Buffer.concat([u32le(GGUF_VALUE_TYPE.UINT64), u64le(v.value)]);
  }
}

export function buildGgufFile(
  metadata: Record<string, MetaValue>,
  tensors: FixtureTensor[],
): Buffer {
  const parts: Buffer[] = [];
  parts.push(Buffer.from("GGUF", "ascii"));
  parts.push(u32le(3)); // version 3
  parts.push(u64le(tensors.length)); // tensor_count
  parts.push(u64le(Object.keys(metadata).length)); // metadata_kv_count

  for (const [key, value] of Object.entries(metadata)) {
    parts.push(ggufString(key));
    parts.push(encodeMetaValue(value));
  }

  for (const t of tensors) {
    parts.push(ggufString(t.name));
    parts.push(u32le(t.dims.length));
    for (const d of t.dims) parts.push(u64le(d));
    parts.push(u32le(t.ggmlType));
    parts.push(u64le(0)); // offset (unused by header-only parser, 0 is fine for the fixture)
  }

  // Tensor data section (not needed by the header parser, but a real file has it).
  for (const t of tensors) parts.push(t.data);

  return Buffer.concat(parts);
}
