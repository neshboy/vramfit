# VRAM Fit

A small, offline command-line calculator for local-LLM users: **will this model fit in your GPU VRAM?**

Give it either a real `.gguf` model file or just a parameter count + quant level, tell it how much VRAM you have (or let it auto-detect), and it computes:

- The real weight size of the model at a given quantization level.
- The real KV-cache size for a target context length.
- Whether the total fits in your VRAM, with a runtime overhead margin.
- The maximum context length you can actually afford.
- How the load would split across multiple GPUs, if you have more than one.

Everything is arithmetic on real, documented formulas and a real binary file parser -- there is no model loading, no GPU driver calls beyond an optional `nvidia-smi` shell-out, and no invented numbers. See "The formulas" below to check the math yourself.

## Quick start

```bash
npm install
npm run build

# Inspect a real GGUF file directly
node dist/cli.js check ./llama-2-7b.Q4_K_M.gguf --vram 8192 --context 4096

# Manual mode: no file needed, just a param count and quant level
node dist/cli.js check --params 7B --quant Q4_K_M --vram 8192
```

Example output:

```
=== VRAM Fit report ===
Weights (Q4_K_M): 3.95 GB  (4047.2 MB)
KV cache: not estimated (need model architecture info -- pass a GGUF file, or --layers/--kv-heads/--head-dim in manual mode)
Overhead margin: 10% (CUDA context/activations/fragmentation)
Total required: 4.35 GB (4451.9 MB)

GPUs (manually specified): #0=8192MB
Total VRAM: 8192 MB

RESULT: FITS
  Fits on a single GPU (GPU #0).
```

Multiple GPUs: pass a comma-separated VRAM list, e.g. `--vram 24576,24576` for two 24GB cards. If you omit `--vram` entirely, VRAM Fit tries `nvidia-smi --query-gpu=memory.total --format=csv,noheader` first and only asks you to specify it manually if that's unavailable.

## Usage

```
vramfit check [model] [options]

Arguments:
  model                path to a .gguf file to inspect directly

Options:
  --vram <mb>           VRAM per GPU in MB, comma-separated for multiple GPUs
                         (e.g. "8192" or "24576,24576"). If omitted, tries nvidia-smi.
  --context <tokens>    desired context length in tokens
  --params <count>      manual mode: parameter count, e.g. 7B, 13B, 70B
  --quant <level>       manual mode: quant level (F32, F16, Q8_0, Q6_K, Q5_1, Q5_0,
                         Q5_K_M, Q5_K_S, Q4_1, Q4_0, Q4_K_M, Q4_K_S, Q3_K_L, Q3_K_M,
                         Q3_K_S, Q2_K)
  --layers <n>          manual mode: transformer layer count, for KV-cache estimate
  --kv-heads <n>        manual mode: number of KV heads, for KV-cache estimate
  --head-dim <n>        manual mode: attention head dimension, for KV-cache estimate
  --overhead <fraction> runtime overhead margin, default 0.1 (10%)
  --kv-bytes <n>        bytes per KV-cache element, default 2 (fp16); use 1 for 8-bit KV cache
```

When you point VRAM Fit at a real `.gguf` file, it reads the model's own embedded architecture metadata (layer count, head counts, embedding size, trained context length) directly from the file header, so you don't need to look any of that up or type it in. In manual mode (no file), KV-cache estimation is skipped unless you supply `--layers`/`--kv-heads`/`--head-dim` yourself, because guessing architecture from a parameter count alone would not be a real number -- see "No faking" below.

## The formulas

**Weight size** (manual mode): `paramCount x bytesPerParam(quantLevel)`.

The bytes-per-param table is built from the real GGML block layouts that every GGUF file actually uses on disk (from `ggml.c`'s `block_q*` struct sizes), plus the published k-quants bits-per-weight measurements for the "_S/_M/_L" mixed-precision variants (ggerganov/llama.cpp PR #1684, "k-quants", June 2023, measured against a 7B LLaMA model -- the numbers the llama.cpp project and community still cite for these formats). Full derivations and byte-exact block math are documented in the header comments of [`src/quant.ts`](src/quant.ts). For example:

- `Q4_0`: 18 bytes / 32 elements = 0.5625 bytes/param (block-exact)
- `Q8_0`: 34 bytes / 32 elements = 1.0625 bytes/param (block-exact)
- `Q4_K_M`: 4.85 bits/param (measured) / 8 = 0.60625 bytes/param
- `F16`: 2 bytes/param (no blocking)

**Weight size** (real GGUF file mode): VRAM Fit parses the file's tensor-info table (name, dimensions, and real per-tensor `ggml_type`) and sums the *exact* on-disk byte size of every tensor using the same block-size table, rather than estimating from an average. This is more accurate than the manual-mode formula because it reflects the file's actual mixed quantization, not an average. See [`src/ggmlType.ts`](src/ggmlType.ts).

**KV-cache size**:

```
kv_cache_bytes = 2 (K and V) x num_layers x num_kv_heads x head_dim x context_length x bytes_per_kv_element
```

This is the standard transformer KV-cache formula (see e.g. Kipply's ["Transformer Inference Arithmetic"](https://kipp.ly/transformer-inference-arithmetic/) and EleutherAI's "Transformer Math 101"), and matches what llama.cpp itself allocates for its KV-cache buffer. `num_layers`, `num_kv_heads`, and `head_dim` come straight from a GGUF file's own metadata (`<arch>.block_count`, `<arch>.attention.head_count_kv`, `<arch>.embedding_length / <arch>.attention.head_count`) when a file is supplied. See [`src/kvcache.ts`](src/kvcache.ts).

**Overhead margin**: `(weights + KV cache) x (1 + overheadFraction)`, default 10%. This covers the CUDA/HIP context itself, intermediate activation buffers, and allocator fragmentation -- real and well-known to anyone who has loaded a model to exactly the free-VRAM number and still hit an out-of-memory error, but backend- and batch-size-dependent, so it's modeled as a configurable flat percentage rather than a claimed-precise number. See [`src/overhead.ts`](src/overhead.ts).

**Multi-GPU split**: sorts GPUs largest-first, finds the minimum number of GPUs whose combined VRAM covers the requirement, then splits the load across exactly those GPUs proportional to each one's share of that combined VRAM (the sane default when you haven't specified custom split ratios yourself, and the same idea as `--tensor-split` ratios in real backends). See [`src/split.ts`](src/split.ts).

**GGUF file parsing**: a from-scratch implementation of the real, public [GGUF binary format spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) -- magic bytes, version, tensor count, metadata key-value pairs (including typed arrays), and the tensor-info table -- reading only the header (never the multi-gigabyte tensor data blob that follows it). See [`src/gguf.ts`](src/gguf.ts).

## GPU VRAM auto-detection

If you don't pass `--vram`, VRAM Fit shells out to:

```
nvidia-smi --query-gpu=memory.total --format=csv,noheader
```

using a real `execFileSync` call with an argv array (no shell string, no injection surface), and parses its real, documented CSV output (one line per GPU, e.g. `24576 MiB`). If `nvidia-smi` isn't on `PATH` or fails for any reason, VRAM Fit falls back to requiring `--vram <MB>` and tells you so explicitly.

**Honesty note**: the machine this tool was built and tested on has no NVIDIA GPU and no `nvidia-smi` installed, so the auto-detection happy path (a real driver actually returning real GPU memory numbers) could not be exercised end-to-end here. What *is* tested and verified: the CSV-output parser against real, documented `nvidia-smi` output strings (`test/vram.test.ts`), and the fallback path (missing/failing `nvidia-smi` -> `null` -> manual `--vram` required) which is exactly what ran in this dev environment every time. If you run this on a machine with an NVIDIA GPU, the auto-detect path should work, but that claim is based on `nvidia-smi`'s documented, stable output format, not on having watched it happen here.

## Testing

```bash
npm test
```

Real automated tests (vitest), including:

- Quant-size arithmetic checked against the documented block-exact and k-quant bpw values (e.g. a 7B model at `Q4_K_M` lands in the real-world ~3.9-4.4GB range that published 7B `Q4_K_M` GGUF files actually ship at).
- The KV-cache formula checked against a hand-computed value for a small example architecture (4 layers, 8 KV heads, head_dim 64, 2048 tokens, fp16 -> exactly 16 MiB).
- A hand-built, byte-for-byte real minimal GGUF file (constructed in the test itself, magic bytes and all) that the parser must correctly read back: metadata key-value pairs, tensor dimensions/types, exact summed tensor byte size, and the "no `head_count_kv` in the file" GQA fallback.
- Multi-GPU split logic across fits-on-one / needs-a-split / doesn't-fit-at-all / minimum-GPUs-selected scenarios.
- The `nvidia-smi` CSV output parser against real documented output strings.

## Limitations

This is a **planning estimate**, not a guarantee. Read this section before trusting a number:

- **Real-world VRAM usage varies by inference engine and backend.** llama.cpp, vLLM, ExLlama, etc. all have different memory allocators, different activation-buffer strategies, and different defaults (batch size, flash-attention kernels, etc.). The formulas here are the standard, documented ones, but no two backends will use *exactly* the same number of bytes.
- **The overhead margin is a flat percentage assumption (10% by default), not a measurement on your hardware.** It's configurable via `--overhead` precisely because it's a guess at a real, known-to-exist cost, not a precise figure.
- **Manual mode's KV-cache estimate is skipped unless you supply the architecture flags** (`--layers`/`--kv-heads`/`--head-dim`). We intentionally do not guess a "typical" architecture from a parameter count alone (e.g. assuming every "7B" model has LLaMA's exact layer/head layout) -- that would be exactly the kind of invented number this tool is trying not to produce. Point it at the real GGUF file instead for a real KV-cache number.
- **The "_S/_M/_L" k-quant bpw figures used in manual mode are measurements from one reference model (7B LLaMA, from the original k-quants PR), not universal constants** -- actual mixed-precision layer assignment can vary slightly by model architecture and llama.cpp version. Real-file mode (pointing at an actual `.gguf`) sidesteps this entirely by reading each tensor's real on-disk type.
- **`nvidia-smi` auto-detection's happy path was not testable on this dev machine** (no NVIDIA GPU present) -- see the dedicated section above for exactly what was and wasn't verified.
- **Multi-GPU split math assumes a continuous, perfectly divisible split.** Real inference engines split at layer or tensor granularity, so the achievable split on real hardware is quantized by how evenly a model's layers divide across your GPUs, not perfectly continuous like the proportional numbers this tool reports.
- Only common GGUF tensor types are recognized (F32/F16/BF16 and the Q4_0..Q8_K family). Newer/exotic quantizations (e.g. the IQ* series) will produce an explicit error rather than a silently wrong number.

## License

MIT, see [LICENSE](LICENSE).
