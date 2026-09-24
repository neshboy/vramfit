/** Small formatting helpers shared by the CLI output. */

export function bytesToMb(bytes: number): number {
  return bytes / (1024 * 1024);
}

export function bytesToGb(bytes: number): number {
  return bytes / (1024 * 1024 * 1024);
}

export function fmtMb(bytes: number): string {
  return `${bytesToMb(bytes).toFixed(1)} MB`;
}

export function fmtGb(bytes: number): string {
  return `${bytesToGb(bytes).toFixed(2)} GB`;
}

export function fmtTokens(n: number): string {
  return n.toLocaleString("en-US");
}
