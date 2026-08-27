// ARC-AGI's own 10-color palette, transcribed from
// third_party/ARC-AGI/apps/css/common.css (.symbol_0 .. .symbol_9), per
// ADR-0007's "follow apps/js's palette conventions" decision.
export const PALETTE: readonly string[] = [
  "#000000", // 0 black
  "#0074D9", // 1 blue
  "#FF4136", // 2 red
  "#2ECC40", // 3 green
  "#FFDC00", // 4 yellow
  "#AAAAAA", // 5 grey
  "#F012BE", // 6 fuschia
  "#FF851B", // 7 orange
  "#7FDBFF", // 8 teal
  "#870C25", // 9 brown
];

// Marks a padded (off-grid) cell in the env's fixed-size observation array -
// never appears in an actual ARC grid, only relevant if a future slice
// renders the raw observation instead of the logged grid.
export const PAD_COLOR = "#222222";

export function colorForSymbol(symbol: number): string {
  if (symbol < 0 || symbol >= PALETTE.length) {
    return PAD_COLOR;
  }
  return PALETTE[symbol];
}
