/**
 * Real GGUF binary format parser.
 *
 * GGUF (GGML Universal File) is the file format llama.cpp and the broader
 * local-LLM ecosystem uses to ship quantized models. It is a public,
 * documented binary format (spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).
 * Layout (version >= 2, the current stable layout):
 *
 *   magic          : 4 bytes, ASCII "GGUF"
 *   version        : uint32
 *   tensor_count   : uint64
 *   metadata_kv_count : uint64
 *   metadata_kv[metadata_kv_count]:
 *     key   : gguf_string  (uint64 length + that many UTF-8 bytes)
 *     value_type : uint32  (see GgufValueType below)
 *     value : encoding depends on value_type
 *   tensor_info[tensor_count]:
 *     name         : gguf_string
 *     n_dimensions : uint32
 *     dimensions   : n_dimensions x uint64
 *     type         : uint32 (ggml_type, see ggmlType.ts)
 *     offset       : uint64 (byte offset into the tensor data section)
 *
 * We only need this header (metadata + tensor-info table), never the actual
 * tensor data blob that follows it, so we read the file lazily in growable
 * chunks from the front instead of loading potentially many GB into memory.
 */

import { openSync, closeSync, fstatSync, readSync } from "node:fs";
import { GGML_TYPES, tensorByteSize } from "./ggmlType.js";

export enum GgufValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

export type GgufValue = number | bigint | string | boolean | GgufValue[];

export interface GgufTensorInfo {
  name: string;
  dims: number[];
  nElements: number;
  ggmlType: number;
  offset: bigint;
}

export interface GgufHeader {
  version: number;
  tensorCount: number;
  metadata: Map<string, GgufValue>;
  tensors: GgufTensorInfo[];
}

const INITIAL_CHUNK = 4 * 1024 * 1024; // 4MB is generously more than most GGUF headers need

/** Reads a growable prefix of a file, refilling from disk on demand as the parser advances. */
class ChunkedFileReader {
  private fd: number;
  private fileSize: number;
  private buffer: Buffer;
  offset = 0;

  constructor(path: string) {
    this.fd = openSync(path, "r");
    this.fileSize = fstatSync(this.fd).size;
    const initial = Math.min(this.fileSize, INITIAL_CHUNK);
    this.buffer = Buffer.alloc(initial);
    readSync(this.fd, this.buffer, 0, initial, 0);
  }

  close(): void {
    closeSync(this.fd);
  }

  private ensure(nBytes: number): void {
    if (this.offset + nBytes <= this.buffer.length) return;
    const needed = this.offset + nBytes;
    if (needed > this.fileSize) {
      throw new Error(
        `Unexpected end of file while parsing GGUF header (needed byte ${needed}, file is ${this.fileSize} bytes). ` +
          `This file is likely truncated or not a valid GGUF file.`,
      );
    }
    const newSize = Math.min(this.fileSize, Math.max(this.buffer.length * 2, needed));
    const newBuffer = Buffer.alloc(newSize);
    readSync(this.fd, newBuffer, 0, newSize, 0);
    this.buffer = newBuffer;
  }

  bytes(n: number): Buffer {
    this.ensure(n);
    const b = this.buffer.subarray(this.offset, this.offset + n);
    this.offset += n;
    return b;
  }

  u8(): number {
    return this.bytes(1).readUInt8(0);
  }
  i8(): number {
    return this.bytes(1).readInt8(0);
  }
  u16(): number {
    return this.bytes(2).readUInt16LE(0);
  }
  i16(): number {
    return this.bytes(2).readInt16LE(0);
  }
  u32(): number {
    return this.bytes(4).readUInt32LE(0);
  }
  i32(): number {
    return this.bytes(4).readInt32LE(0);
  }
  f32(): number {
    return this.bytes(4).readFloatLE(0);
  }
  f64(): number {
    return this.bytes(8).readDoubleLE(0);
  }
  u64(): bigint {
    return this.bytes(8).readBigUInt64LE(0);
  }
  i64(): bigint {
    return this.bytes(8).readBigInt64LE(0);
  }
  bool(): boolean {
    return this.u8() !== 0;
  }
  str(): string {
    const len = Number(this.u64());
    return this.bytes(len).toString("utf8");
  }
}

/** Reads one value of the given GGUF metadata value type. Recurses for ARRAY. */
function readValue(r: ChunkedFileReader, type: GgufValueType): GgufValue {
  switch (type) {
    case GgufValueType.UINT8:
      return r.u8();
    case GgufValueType.INT8:
      return r.i8();
    case GgufValueType.UINT16:
      return r.u16();
    case GgufValueType.INT16:
      return r.i16();
    case GgufValueType.UINT32:
      return r.u32();
    case GgufValueType.INT32:
      return r.i32();
    case GgufValueType.FLOAT32:
      return r.f32();
    case GgufValueType.BOOL:
      return r.bool();
    case GgufValueType.STRING:
      return r.str();
    case GgufValueType.UINT64:
      return safeNumber(r.u64());
    case GgufValueType.INT64:
      return safeNumber(r.i64());
    case GgufValueType.FLOAT64:
      return r.f64();
    case GgufValueType.ARRAY: {
      const elemType = r.u32() as GgufValueType;
      const len = Number(r.u64());
      const arr: GgufValue[] = new Array(len);
      for (let i = 0; i < len; i++) arr[i] = readValue(r, elemType);
      return arr;
    }
    default:
      throw new Error(`Unknown GGUF metadata value type id ${type}`);
  }
}

function safeNumber(v: bigint): number | bigint {
  return v <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(v) : v;
}

export function parseGgufHeader(path: string): GgufHeader {
  const r = new ChunkedFileReader(path);
  try {
    const magic = r.bytes(4).toString("ascii");
    if (magic !== "GGUF") {
      throw new Error(`Not a GGUF file: expected magic "GGUF", got "${magic}".`);
    }
    const version = r.u32();
    if (version < 2) {
      throw new Error(
        `GGUF version ${version} uses a legacy 32-bit string/count layout that this parser does not support. ` +
          `Please use a file written by a modern (post-2023) llama.cpp/gguf tool (version >= 2).`,
      );
    }
    const tensorCount = Number(r.u64());
    const metadataKvCount = Number(r.u64());

    const metadata = new Map<string, GgufValue>();
    for (let i = 0; i < metadataKvCount; i++) {
      const key = r.str();
      const valueType = r.u32() as GgufValueType;
      const value = readValue(r, valueType);
      metadata.set(key, value);
    }

    const tensors: GgufTensorInfo[] = [];
    for (let i = 0; i < tensorCount; i++) {
      const name = r.str();
      const nDims = r.u32();
      const dims: number[] = [];
      for (let d = 0; d < nDims; d++) dims.push(Number(r.u64()));
      const ggmlType = r.u32();
      const offset = r.u64();
      const nElements = dims.reduce((a, b) => a * b, 1);
      tensors.push({ name, dims, nElements, ggmlType, offset });
    }

    return { version, tensorCount, metadata, tensors };
  } finally {
    r.close();
  }
}

/** Sum of the exact on-disk byte size of every tensor, per its own real ggml type. */
export function totalTensorBytes(tensors: GgufTensorInfo[]): number {
  return tensors.reduce((sum, t) => sum + tensorByteSize(t.nElements, t.ggmlType), 0);
}

/** Human name for the dominant (most tensor-bytes) ggml type, for display purposes. */
export function dominantQuantName(tensors: GgufTensorInfo[]): string {
  const byType = new Map<number, number>();
  for (const t of tensors) {
    const bytes = tensorByteSize(t.nElements, t.ggmlType);
    byType.set(t.ggmlType, (byType.get(t.ggmlType) ?? 0) + bytes);
  }
  let bestType = -1;
  let bestBytes = -1;
  for (const [type, bytes] of byType) {
    if (bytes > bestBytes) {
      bestBytes = bytes;
      bestType = type;
    }
  }
  return GGML_TYPES[bestType]?.name ?? `unknown(${bestType})`;
}

export interface ArchConfig {
  architecture: string;
  contextLength?: number;
  embeddingLength?: number;
  blockCount?: number;
  headCount?: number;
  headCountKv?: number;
  headDim?: number;
}

/**
 * Pulls the architecture parameters needed for the KV-cache formula out of
 * GGUF metadata. Keys are namespaced by architecture, e.g. "llama.block_count",
 * "qwen2.attention.head_count_kv" -- see the GGUF key schema in the spec's
 * "General metadata" / per-architecture sections.
 */
export function deriveArchConfig(metadata: Map<string, GgufValue>): ArchConfig {
  const architecture = String(metadata.get("general.architecture") ?? "unknown");
  const get = (suffix: string): number | undefined => {
    const v = metadata.get(`${architecture}.${suffix}`);
    if (v === undefined) return undefined;
    return typeof v === "bigint" ? Number(v) : (v as number);
  };

  const contextLength = get("context_length");
  const embeddingLength = get("embedding_length");
  const blockCount = get("block_count");
  const headCount = get("attention.head_count");
  // Older/non-GQA models never set head_count_kv; llama.cpp's own loader
  // treats "absent" as "same as head_count" (i.e. plain multi-head attention).
  const headCountKv = get("attention.head_count_kv") ?? headCount;
  // Some architectures publish an explicit per-head key/value dimension;
  // otherwise it's embedding_length / head_count, the standard transformer split.
  const headDim =
    get("attention.key_length") ??
    (embeddingLength !== undefined && headCount ? embeddingLength / headCount : undefined);

  return { architecture, contextLength, embeddingLength, blockCount, headCount, headCountKv, headDim };
}
