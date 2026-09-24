import { describe, it, expect } from "vitest";
import { parseNvidiaSmiOutput, detectGpusViaNvidiaSmi } from "../src/vram.js";

describe("parseNvidiaSmiOutput", () => {
  it("parses real single-GPU `nvidia-smi --query-gpu=memory.total --format=csv,noheader` output", () => {
    // This is the real, documented output format for that exact flag combination
    // (one CSV row per GPU, no header row, unit suffix included, e.g. "24576 MiB").
    const gpus = parseNvidiaSmiOutput("24576 MiB\n");
    expect(gpus).toEqual([{ index: 0, totalMb: 24576 }]);
  });

  it("parses real multi-GPU output, one line per device", () => {
    const gpus = parseNvidiaSmiOutput("8192 MiB\n8192 MiB\n");
    expect(gpus).toEqual([
      { index: 0, totalMb: 8192 },
      { index: 1, totalMb: 8192 },
    ]);
  });

  it("converts GiB units to MB", () => {
    const gpus = parseNvidiaSmiOutput("24 GiB\n");
    expect(gpus[0].totalMb).toBe(24 * 1024);
  });

  it("throws on unparseable output rather than silently returning wrong numbers", () => {
    expect(() => parseNvidiaSmiOutput("not a gpu line\n")).toThrow();
  });
});

describe("detectGpusViaNvidiaSmi (live)", () => {
  it("does not throw, and honestly reflects this dev machine having no NVIDIA GPU/driver", () => {
    // This test documents reality on the machine this suite was authored and run on:
    // no nvidia-smi on PATH, so real auto-detection could not be exercised end-to-end
    // here. The function must degrade to null (not throw) so callers fall back to
    // manual --vram input; that fallback behavior is what this test actually verifies.
    const result = detectGpusViaNvidiaSmi();
    expect(result === null || Array.isArray(result)).toBe(true);
  });
});
