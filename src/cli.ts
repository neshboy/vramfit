#!/usr/bin/env node
import { Command } from "commander";
import { deriveArchConfig, dominantQuantName, parseGgufHeader, totalTensorBytes } from "./gguf.js";
import { isQuantLevel, parseParamCount, weightBytes, QUANT_LEVELS } from "./quant.js";
import { kvCacheBytes, maxContextForBudget } from "./kvcache.js";
import { DEFAULT_OVERHEAD_FRACTION, withOverhead, stripOverhead } from "./overhead.js";
import { detectGpusViaNvidiaSmi } from "./vram.js";
import { computeSplitPlan } from "./split.js";
import { fmtGb, fmtMb, fmtTokens } from "./format.js";

const FALLBACK_CONTEXT_LENGTH = 4096;

interface CheckOptions {
  vram?: string;
  context?: string;
  params?: string;
  quant?: string;
  layers?: string;
  kvHeads?: string;
  headDim?: string;
  overhead?: string;
  kvBytes?: string;
}

function parseVramList(raw: string | undefined): number[] | null {
  if (!raw) return null;
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => {
      const n = Number(s);
      if (!Number.isFinite(n) || n <= 0) {
        throw new Error(`Invalid --vram value "${s}". Expected a positive number of MB.`);
      }
      return n;
    });
}

function resolveVram(options: CheckOptions): { vramMb: number[]; source: "manual" | "nvidia-smi" } {
  const manual = parseVramList(options.vram);
  if (manual) return { vramMb: manual, source: "manual" };

  const detected = detectGpusViaNvidiaSmi();
  if (detected) {
    return { vramMb: detected.map((g) => g.totalMb), source: "nvidia-smi" };
  }

  throw new Error(
    "No GPU VRAM could be determined: nvidia-smi is not on PATH (or found no GPUs), and no --vram was given.\n" +
      "Pass it manually, e.g. --vram 8192 for one GPU or --vram 24576,24576 for two.",
  );
}

function runCheck(modelPath: string | undefined, options: CheckOptions): void {
  const overheadFraction =
    options.overhead !== undefined ? Number(options.overhead) : DEFAULT_OVERHEAD_FRACTION;
  const kvBytesPerElement = options.kvBytes !== undefined ? Number(options.kvBytes) : 2;

  let weightBytesTotal: number;
  let quantLabel: string;
  let arch: {
    numLayers?: number;
    numKvHeads?: number;
    headDim?: number;
    modelContextLength?: number;
    architecture?: string;
  } = {};

  if (modelPath) {
    console.log(`Reading GGUF header: ${modelPath}`);
    const header = parseGgufHeader(modelPath);
    weightBytesTotal = totalTensorBytes(header.tensors);
    quantLabel = dominantQuantName(header.tensors);
    const cfg = deriveArchConfig(header.metadata);
    arch = {
      numLayers: cfg.blockCount,
      numKvHeads: cfg.headCountKv,
      headDim: cfg.headDim,
      modelContextLength: cfg.contextLength,
      architecture: cfg.architecture,
    };
    console.log(
      `  architecture=${cfg.architecture} tensors=${header.tensors.length} dominant-quant=${quantLabel}`,
    );
  } else {
    if (!options.params || !options.quant) {
      throw new Error(
        "Manual mode requires both --params <e.g. 7B> and --quant <e.g. Q4_K_M> when no model file is given.",
      );
    }
    if (!isQuantLevel(options.quant)) {
      throw new Error(`Unknown --quant "${options.quant}". Supported: ${QUANT_LEVELS.join(", ")}`);
    }
    const paramCount = parseParamCount(options.params);
    weightBytesTotal = weightBytes(paramCount, options.quant);
    quantLabel = options.quant;
    if (options.layers && options.kvHeads && options.headDim) {
      arch = {
        numLayers: Number(options.layers),
        numKvHeads: Number(options.kvHeads),
        headDim: Number(options.headDim),
      };
    }
    console.log(
      `Manual mode: ${options.params} params (${paramCount.toLocaleString("en-US")}) @ ${quantLabel}`,
    );
  }

  const haveArchForKv =
    arch.numLayers !== undefined && arch.numKvHeads !== undefined && arch.headDim !== undefined;

  const requestedContext = options.context
    ? Number(options.context)
    : arch.modelContextLength ?? (haveArchForKv ? FALLBACK_CONTEXT_LENGTH : undefined);

  let kvBytesTotal = 0;
  if (haveArchForKv && requestedContext) {
    kvBytesTotal = kvCacheBytes({
      numLayers: arch.numLayers!,
      numKvHeads: arch.numKvHeads!,
      headDim: arch.headDim!,
      contextLength: requestedContext,
      bytesPerKvElement: kvBytesPerElement,
    });
  }

  const totalBeforeOverhead = weightBytesTotal + kvBytesTotal;
  const totalRequired = withOverhead(totalBeforeOverhead, overheadFraction);

  const { vramMb, source } = resolveVram(options);
  const plan = computeSplitPlan(vramMb, totalRequired);

  console.log("");
  console.log("=== VRAM Fit report ===");
  console.log(`Weights (${quantLabel}): ${fmtGb(weightBytesTotal)}  (${fmtMb(weightBytesTotal)})`);
  if (haveArchForKv && requestedContext) {
    console.log(
      `KV cache @ ${fmtTokens(requestedContext)} tokens: ${fmtGb(kvBytesTotal)} ` +
        `(layers=${arch.numLayers} kv_heads=${arch.numKvHeads} head_dim=${arch.headDim} ` +
        `bytes/elem=${kvBytesPerElement})`,
    );
  } else {
    console.log(
      "KV cache: not estimated (need model architecture info -- pass a GGUF file, " +
        "or --layers/--kv-heads/--head-dim in manual mode)",
    );
  }
  console.log(`Overhead margin: ${(overheadFraction * 100).toFixed(0)}% (CUDA context/activations/fragmentation)`);
  console.log(`Total required: ${fmtGb(totalRequired)} (${fmtMb(totalRequired)})`);
  console.log("");
  console.log(
    `GPUs (${source === "nvidia-smi" ? "auto-detected via nvidia-smi" : "manually specified"}): ` +
      vramMb.map((mb, i) => `#${i}=${mb}MB`).join(", "),
  );
  console.log(`Total VRAM: ${plan.totalVramMb.toFixed(0)} MB`);
  console.log("");

  if (plan.fits) {
    console.log(`RESULT: FITS`);
    if (plan.gpusUsed === 1) {
      console.log(`  Fits on a single GPU (GPU #${plan.allocations[0].index}).`);
    } else {
      console.log(`  Needs ${plan.gpusUsed} of your ${plan.gpuCount} GPU(s), split as:`);
      for (const a of plan.allocations) {
        console.log(`    GPU #${a.index} (${a.vramMb.toFixed(0)} MB capacity): ${a.allocatedMb.toFixed(0)} MB`);
      }
    }
  } else {
    console.log(`RESULT: DOES NOT FIT`);
    console.log(
      `  Short by ${fmtMb(plan.shortfallMb * 1024 * 1024)} even using all ${plan.gpuCount} GPU(s) ` +
        `(${plan.totalVramMb.toFixed(0)} MB total).`,
    );
    console.log(`  Try a smaller quant level, fewer context tokens, or more/bigger GPUs.`);
  }

  if (haveArchForKv) {
    const totalVramBytes = plan.totalVramMb * 1024 * 1024;
    const availableForModel = stripOverhead(totalVramBytes, overheadFraction);
    const availableForKv = availableForModel - weightBytesTotal;
    const maxContext = maxContextForBudget(
      {
        numLayers: arch.numLayers!,
        numKvHeads: arch.numKvHeads!,
        headDim: arch.headDim!,
        bytesPerKvElement: kvBytesPerElement,
      },
      availableForKv,
    );
    console.log("");
    if (availableForKv <= 0) {
      console.log(`Max context length that fits: 0 (weights alone exceed your usable VRAM budget)`);
    } else {
      console.log(`Max context length that fits across all listed GPUs: ~${fmtTokens(maxContext)} tokens`);
    }
  }
}

const program = new Command();
program
  .name("vramfit")
  .description("Offline calculator: will this LLM fit in your GPU VRAM?")
  .version("0.1.0");

program
  .command("check")
  .argument("[model]", "path to a .gguf model file to inspect directly")
  .option("--vram <mb>", "VRAM per GPU in MB, comma-separated for multiple GPUs (e.g. 8192 or 24576,24576)")
  .option("--context <tokens>", "desired context length in tokens")
  .option("--params <count>", "manual mode: parameter count, e.g. 7B, 13B, 70B")
  .option("--quant <level>", `manual mode: quant level (${QUANT_LEVELS.join(", ")})`)
  .option("--layers <n>", "manual mode: number of transformer layers, for KV-cache estimate")
  .option("--kv-heads <n>", "manual mode: number of KV heads, for KV-cache estimate")
  .option("--head-dim <n>", "manual mode: attention head dimension, for KV-cache estimate")
  .option("--overhead <fraction>", `runtime overhead margin as a fraction, default ${DEFAULT_OVERHEAD_FRACTION}`)
  .option("--kv-bytes <n>", "bytes per KV cache element, default 2 (fp16); use 1 for an 8-bit KV cache")
  .action((model: string | undefined, options: CheckOptions) => {
    try {
      runCheck(model, options);
    } catch (err) {
      console.error(`Error: ${(err as Error).message}`);
      process.exitCode = 1;
    }
  });

program.parse();
