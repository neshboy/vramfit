/**
 * Multi-GPU split planning.
 *
 * Given the real VRAM capacity of each installed GPU and the real total
 * bytes a model+KV-cache+overhead needs, this answers three questions:
 *   1. Does it fit at all, across all available GPUs combined?
 *   2. What is the *minimum* number of GPUs (largest-first) needed?
 *   3. How should the load be split across exactly that many GPUs?
 *
 * The split itself is a weighted proportional split: each selected GPU is
 * assigned a share of the total load proportional to its share of the
 * combined VRAM of the GPUs actually used. This mirrors how tensor-parallel
 * / layer-split backends (e.g. llama.cpp's --tensor-split) let you divide
 * work by explicit ratios; an even VRAM-proportional ratio is the sane
 * default when the user hasn't specified custom ratios themselves.
 *
 * Known simplification (documented, not hidden): real engines split at
 * layer/tensor granularity, so the achievable split is quantized by how
 * evenly a model's layers divide across GPUs, not perfectly continuous like
 * the proportional numbers below. See README "Limitations".
 */

export interface GpuAllocation {
  index: number;
  vramMb: number;
  allocatedMb: number;
}

export interface SplitPlan {
  fits: boolean;
  gpuCount: number;
  gpusUsed: number;
  totalVramMb: number;
  requiredMb: number;
  shortfallMb: number;
  allocations: GpuAllocation[];
}

const BYTES_PER_MB = 1024 * 1024;

export function computeSplitPlan(gpuVramMb: number[], requiredBytes: number): SplitPlan {
  if (gpuVramMb.length === 0) {
    throw new Error("computeSplitPlan requires at least one GPU VRAM amount");
  }
  const requiredMb = requiredBytes / BYTES_PER_MB;
  const totalVramMb = gpuVramMb.reduce((a, b) => a + b, 0);
  const fits = totalVramMb >= requiredMb;

  // Largest-first greedy selection: find the minimum number of GPUs whose
  // combined VRAM covers the requirement (or all of them, if it doesn't fit).
  const indexed = gpuVramMb.map((vramMb, index) => ({ index, vramMb }));
  const sortedDesc = [...indexed].sort((a, b) => b.vramMb - a.vramMb);

  let cumulative = 0;
  let gpusUsed = sortedDesc.length;
  for (let k = 0; k < sortedDesc.length; k++) {
    cumulative += sortedDesc[k].vramMb;
    if (cumulative >= requiredMb) {
      gpusUsed = k + 1;
      break;
    }
  }

  const selected = sortedDesc.slice(0, gpusUsed);
  const selectedTotalVramMb = selected.reduce((a, b) => a + b.vramMb, 0);

  const allocations: GpuAllocation[] = selected
    .map((g) => ({
      index: g.index,
      vramMb: g.vramMb,
      // Proportional share of the requirement if it fits; if it doesn't fit
      // even using every GPU, report each GPU maxed out at its own capacity
      // (there's nowhere else for the remainder to go).
      allocatedMb: fits ? requiredMb * (g.vramMb / selectedTotalVramMb) : g.vramMb,
    }))
    .sort((a, b) => a.index - b.index);

  return {
    fits,
    gpuCount: gpuVramMb.length,
    gpusUsed,
    totalVramMb,
    requiredMb,
    shortfallMb: fits ? 0 : requiredMb - totalVramMb,
    allocations,
  };
}
