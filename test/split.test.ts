import { describe, it, expect } from "vitest";
import { computeSplitPlan } from "../src/split.js";

const GB = 1024 * 1024 * 1024;
const MB = 1024 * 1024;

describe("computeSplitPlan", () => {
  it("fits on a single GPU when it has enough capacity alone", () => {
    const plan = computeSplitPlan([24576], 10 * GB);
    expect(plan.fits).toBe(true);
    expect(plan.gpusUsed).toBe(1);
    expect(plan.allocations).toHaveLength(1);
    expect(plan.allocations[0].index).toBe(0);
    expect(plan.allocations[0].allocatedMb).toBeCloseTo(10240, 3);
  });

  it("needs a split across two equal GPUs and divides evenly between them", () => {
    // 14 GiB required, two 8192MB (8GiB) GPUs = 16384MB total. Doesn't fit on
    // one alone (8192 < 14336) but fits split across both.
    const requiredBytes = 14 * GB;
    const plan = computeSplitPlan([8192, 8192], requiredBytes);
    expect(plan.fits).toBe(true);
    expect(plan.gpusUsed).toBe(2);
    expect(plan.requiredMb).toBeCloseTo(14336, 3);
    expect(plan.allocations).toHaveLength(2);
    // Equal-capacity GPUs => equal 50/50 split of the requirement.
    for (const a of plan.allocations) {
      expect(a.allocatedMb).toBeCloseTo(7168, 3);
    }
  });

  it("picks the minimum number of GPUs needed (largest-first), not all of them", () => {
    // 30000MB required. Sorted desc: 24576, 8192, 8192. First alone isn't enough
    // (24576 < 30000), first two are (24576+8192=32768 >= 30000) -> only 2 GPUs used,
    // and the third (smallest) GPU should be left out of the plan entirely.
    const plan = computeSplitPlan([24576, 8192, 8192], 30000 * MB);
    expect(plan.fits).toBe(true);
    expect(plan.gpusUsed).toBe(2);
    expect(plan.allocations).toHaveLength(2);
    const usedIndices = plan.allocations.map((a) => a.index).sort();
    expect(usedIndices).toEqual([0, 1]); // GPU #0 (24576) and GPU #1 (8192), not #2
  });

  it("reports it does not fit at all when total VRAM across every GPU is insufficient", () => {
    const plan = computeSplitPlan([8192], 20 * GB);
    expect(plan.fits).toBe(false);
    expect(plan.gpusUsed).toBe(1);
    expect(plan.totalVramMb).toBe(8192);
    expect(plan.shortfallMb).toBeCloseTo(20 * 1024 - 8192, 3);
  });

  it("reports allocations maxed out per-GPU capacity when it doesn't fit even using all GPUs", () => {
    const plan = computeSplitPlan([8192, 8192], 100 * GB);
    expect(plan.fits).toBe(false);
    expect(plan.gpusUsed).toBe(2);
    for (const a of plan.allocations) {
      expect(a.allocatedMb).toBe(a.vramMb);
    }
  });

  it("preserves original GPU index ordering in the output regardless of sort order used internally", () => {
    const plan = computeSplitPlan([4096, 24576, 8192], 30000 * MB);
    // sorted desc by vram: index1(24576), index2(8192), index0(4096)
    // cumulative: 24576 < 30000; 24576+8192=32768 >= 30000 -> gpusUsed=2, uses indices 1 and 2
    expect(plan.gpusUsed).toBe(2);
    const indices = plan.allocations.map((a) => a.index);
    expect(indices).toEqual([1, 2]); // sorted ascending by original index in output
  });
});
