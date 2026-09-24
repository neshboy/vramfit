/**
 * KV-cache memory formula.
 *
 * During autoregressive inference, the transformer caches the Key and Value
 * projections for every past token so it doesn't recompute them each step.
 * For a standard (optionally grouped-query) attention transformer, the cache
 * holds, per layer, one K tensor and one V tensor of shape
 * [context_length, num_kv_heads, head_dim]. This is the same formula used by
 * the well-known transformer-memory write-ups (e.g. Kipply's "Transformer
 * Inference Arithmetic", https://kipp.ly/transformer-inference-arithmetic/,
 * and EleutherAI's "Transformer Math 101") and matches what llama.cpp
 * allocates for its KV cache buffer:
 *
 *   kv_cache_bytes = 2 (K and V) x num_layers x num_kv_heads x head_dim
 *                    x context_length x bytes_per_kv_element
 *
 * - num_layers        : number of transformer blocks (GGUF: <arch>.block_count)
 * - num_kv_heads      : number of key/value heads (GGUF: <arch>.attention.head_count_kv,
 *                        or head_count if the model doesn't use grouped-query attention)
 * - head_dim          : dimension of each attention head (embedding_length / head_count,
 *                        or an explicit key_length if the architecture publishes one)
 * - context_length    : how many tokens of cache you want to be able to hold
 * - bytes_per_kv_element : 2 for fp16 (llama.cpp's default KV cache dtype); use 1 if
 *                        you're quantizing the KV cache itself (e.g. Q8_0 KV cache)
 */

export interface KvCacheParams {
  numLayers: number;
  numKvHeads: number;
  headDim: number;
  contextLength: number;
  bytesPerKvElement?: number; // default 2 (fp16)
}

export function kvCacheBytes(params: KvCacheParams): number {
  const bytesPerKvElement = params.bytesPerKvElement ?? 2;
  return (
    2 *
    params.numLayers *
    params.numKvHeads *
    params.headDim *
    params.contextLength *
    bytesPerKvElement
  );
}

/**
 * Solve the KV-cache formula for the maximum context length that fits in a
 * given byte budget (everything else held fixed). Floors to a whole number
 * of tokens.
 */
export function maxContextForBudget(
  params: Omit<KvCacheParams, "contextLength">,
  budgetBytes: number,
): number {
  const bytesPerKvElement = params.bytesPerKvElement ?? 2;
  const perToken = 2 * params.numLayers * params.numKvHeads * params.headDim * bytesPerKvElement;
  if (perToken <= 0) return 0;
  return Math.max(0, Math.floor(budgetBytes / perToken));
}
