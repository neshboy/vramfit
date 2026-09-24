/**
 * GPU VRAM auto-detection.
 *
 * Shells out to `nvidia-smi --query-gpu=memory.total --format=csv,noheader`,
 * the standard, documented way to ask the NVIDIA driver how much VRAM each
 * installed GPU has, without needing CUDA itself or any native bindings.
 * Output is one line per GPU, e.g. "24576 MiB". We use execFileSync with an
 * argv array (never a shell string) so there's no shell-injection surface and
 * no dependency on how the user's shell quotes things.
 *
 * IMPORTANT (honesty note): this repo's dev/build machine has no NVIDIA GPU
 * and no `nvidia-smi` on PATH, so this function's happy path could not be
 * exercised against a real driver in this environment -- only the fallback
 * (nvidia-smi missing/failing -> null -> caller falls back to --vram) was
 * actually observed here. The output-parsing logic is unit-tested against
 * real, documented nvidia-smi output strings instead. See README "Limitations".
 */

import { execFileSync } from "node:child_process";

export interface DetectedGpu {
  index: number;
  totalMb: number;
}

/** Parses `nvidia-smi --query-gpu=memory.total --format=csv,noheader` stdout. */
export function parseNvidiaSmiOutput(stdout: string): DetectedGpu[] {
  const lines = stdout
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  return lines.map((line, index) => {
    // Typical line: "24576 MiB" (csv,noheader gives just the value + unit column)
    const match = /^([0-9]+(?:\.[0-9]+)?)\s*(MiB|GiB|MB|GB)?$/i.exec(line);
    if (!match) {
      throw new Error(`Unexpected nvidia-smi output line: "${line}"`);
    }
    const value = parseFloat(match[1]);
    const unit = (match[2] ?? "MiB").toLowerCase();
    const totalMb = unit === "gib" ? value * 1024 : unit === "gb" ? value * 953.674 : value;
    return { index, totalMb: Math.round(totalMb) };
  });
}

/** Attempts real auto-detection; returns null (never throws) if unavailable. */
export function detectGpusViaNvidiaSmi(): DetectedGpu[] | null {
  try {
    const stdout = execFileSync(
      "nvidia-smi",
      ["--query-gpu=memory.total", "--format=csv,noheader"],
      { encoding: "utf8", windowsHide: true },
    );
    const gpus = parseNvidiaSmiOutput(stdout);
    return gpus.length > 0 ? gpus : null;
  } catch {
    // nvidia-smi not on PATH, no driver, permission error, etc. -- fall back to manual input.
    return null;
  }
}
