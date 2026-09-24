/**
 * Runtime overhead margin.
 *
 * Loading weights + a KV cache onto a GPU is not the whole story: the
 * inference engine also needs room for the CUDA/HIP context itself (typically
 * 200-500MB just for the driver + cuBLAS/cuDNN handles), intermediate
 * activation buffers for the forward pass (attention scores, MLP
 * intermediates, logits), and allocator fragmentation. This is real and
 * well-known to anyone who has loaded a model to exactly "the free VRAM
 * number" and hit an out-of-memory error anyway -- but the exact amount is
 * backend- and batch-size-dependent, so we model it as a flat percentage
 * overhead on top of (weights + KV cache) rather than pretending we can
 * predict it exactly.
 *
 * Default: 10%. This is a deliberately middle-of-the-road planning margin:
 * llama.cpp's own single-GPU CUDA context overhead is commonly observed in
 * the ~300-600MB range, which is roughly 5-10% of a typical 4-8GB quantized
 * 7B footprint; larger models/contexts push the *absolute* overhead up too
 * (bigger activation buffers), so a percentage scales better than a flat MB
 * number across model sizes. It is configurable via --overhead because it is
 * an assumption, not a measurement -- see README "Limitations".
 */
export const DEFAULT_OVERHEAD_FRACTION = 0.1;

export function withOverhead(bytes: number, overheadFraction = DEFAULT_OVERHEAD_FRACTION): number {
  return bytes * (1 + overheadFraction);
}

/** Inverse: how many bytes of (weights + KV) can I afford if overhead eats a fraction of my budget? */
export function stripOverhead(budgetBytes: number, overheadFraction = DEFAULT_OVERHEAD_FRACTION): number {
  return budgetBytes / (1 + overheadFraction);
}
