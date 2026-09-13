import { colorForSymbol } from "./palette";

export type Grid = number[][];

export interface CellRect {
  row: number;
  col: number;
  value: number;
  color: string;
  x: number;
  y: number;
  size: number;
  selectedSlot: "a" | "b" | null;
}

// A step's post-action dual-slot selection (ADR-0011/ADR-0012/ADR-0020):
// slot "a" (the original single-selection channel) and slot "b" (new), each
// as `[row, col]` pairs or `null` when that slot has nothing selected -
// `EpisodeStep["selected"]`'s own shape. The whole value is `null`/undefined
// for steps/episodes predating ADR-0020's shape.
export interface SelectionSlots {
  a: [number, number][] | null;
  b: [number, number][] | null;
}
export type SelectionCells = SelectionSlots | null | undefined;

const MAX_CELL_SIZE = 40;
// Slot "a" keeps the pre-ADR-0020 amber outline (backward-compatible - every
// existing single-selection task/test renders identically). Slot "b" gets a
// second, visually distinct color (a light violet, not used anywhere in
// `palette.ts`'s ARC palette).
const SELECTION_OUTLINE_COLOR_A = "#facc15"; // amber
const SELECTION_OUTLINE_COLOR_B = "#a78bfa"; // light violet
const SELECTION_OUTLINE_WIDTH = 3;

export function cellSizeFor(grid: Grid, maxWidth: number, maxHeight: number): number {
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  if (rows === 0 || cols === 0) return MAX_CELL_SIZE;
  return Math.max(1, Math.min(MAX_CELL_SIZE, Math.floor(maxWidth / cols), Math.floor(maxHeight / rows)));
}

function selectionKeySet(cells: [number, number][] | null | undefined): Set<string> {
  return new Set((cells ?? []).map(([row, col]) => `${row},${col}`));
}

// Pure layout function - lets the palette/geometry logic be unit-tested
// without a real Canvas 2D context (jsdom's canvas support is limited).
export function computeCellRects(grid: Grid, cellSize: number, selected?: SelectionCells): CellRect[] {
  const aKeys = selectionKeySet(selected?.a);
  const bKeys = selectionKeySet(selected?.b);
  const rects: CellRect[] = [];
  for (let row = 0; row < grid.length; row++) {
    for (let col = 0; col < grid[row].length; col++) {
      const value = grid[row][col];
      const key = `${row},${col}`;
      // Slot "b" wins on any overlap, matching `arc_env.env`'s own
      // `_selected_mask` precedence (drawn second).
      const selectedSlot: "a" | "b" | null = bKeys.has(key) ? "b" : aKeys.has(key) ? "a" : null;
      rects.push({
        row,
        col,
        value,
        color: colorForSymbol(value),
        x: col * cellSize,
        y: row * cellSize,
        size: cellSize,
        selectedSlot,
      });
    }
  }
  return rects;
}

export function drawGrid(
  ctx: CanvasRenderingContext2D,
  grid: Grid,
  cellSize: number,
  selected?: SelectionCells,
): void {
  const rects = computeCellRects(grid, cellSize, selected);
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  ctx.clearRect(0, 0, cols * cellSize, rows * cellSize);
  for (const rect of rects) {
    ctx.fillStyle = rect.color;
    ctx.fillRect(rect.x, rect.y, rect.size, rect.size);
    ctx.strokeStyle = "#555555";
    ctx.strokeRect(rect.x, rect.y, rect.size, rect.size);
  }
  // Drawn as a second pass so a selection outline is never clipped under an
  // adjacent cell's own border. Slot "a" first, then slot "b", so both
  // outlines are equally visible when their cells happen to be adjacent.
  for (const rect of rects) {
    if (rect.selectedSlot !== "a") continue;
    ctx.strokeStyle = SELECTION_OUTLINE_COLOR_A;
    ctx.lineWidth = SELECTION_OUTLINE_WIDTH;
    const inset = SELECTION_OUTLINE_WIDTH / 2;
    ctx.strokeRect(rect.x + inset, rect.y + inset, rect.size - SELECTION_OUTLINE_WIDTH, rect.size - SELECTION_OUTLINE_WIDTH);
  }
  for (const rect of rects) {
    if (rect.selectedSlot !== "b") continue;
    ctx.strokeStyle = SELECTION_OUTLINE_COLOR_B;
    ctx.lineWidth = SELECTION_OUTLINE_WIDTH;
    const inset = SELECTION_OUTLINE_WIDTH / 2;
    ctx.strokeRect(rect.x + inset, rect.y + inset, rect.size - SELECTION_OUTLINE_WIDTH, rect.size - SELECTION_OUTLINE_WIDTH);
  }
}
