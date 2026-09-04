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
  selected: boolean;
}

// A step's post-action selection (ADR-0011/ADR-0012), as `[row, col]` pairs
// - `EpisodeStep["selected"]`'s own shape, `null`/undefined when nothing is
// selected.
export type SelectionCells = [number, number][] | null | undefined;

const MAX_CELL_SIZE = 40;
const SELECTION_OUTLINE_COLOR = "#facc15"; // amber - distinct from any ARC palette color
const SELECTION_OUTLINE_WIDTH = 3;

export function cellSizeFor(grid: Grid, maxWidth: number, maxHeight: number): number {
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  if (rows === 0 || cols === 0) return MAX_CELL_SIZE;
  return Math.max(1, Math.min(MAX_CELL_SIZE, Math.floor(maxWidth / cols), Math.floor(maxHeight / rows)));
}

function selectionKeySet(selected: SelectionCells): Set<string> {
  return new Set((selected ?? []).map(([row, col]) => `${row},${col}`));
}

// Pure layout function - lets the palette/geometry logic be unit-tested
// without a real Canvas 2D context (jsdom's canvas support is limited).
export function computeCellRects(grid: Grid, cellSize: number, selected?: SelectionCells): CellRect[] {
  const selectedKeys = selectionKeySet(selected);
  const rects: CellRect[] = [];
  for (let row = 0; row < grid.length; row++) {
    for (let col = 0; col < grid[row].length; col++) {
      const value = grid[row][col];
      rects.push({
        row,
        col,
        value,
        color: colorForSymbol(value),
        x: col * cellSize,
        y: row * cellSize,
        size: cellSize,
        selected: selectedKeys.has(`${row},${col}`),
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
  // adjacent cell's own border.
  for (const rect of rects) {
    if (!rect.selected) continue;
    ctx.strokeStyle = SELECTION_OUTLINE_COLOR;
    ctx.lineWidth = SELECTION_OUTLINE_WIDTH;
    const inset = SELECTION_OUTLINE_WIDTH / 2;
    ctx.strokeRect(rect.x + inset, rect.y + inset, rect.size - SELECTION_OUTLINE_WIDTH, rect.size - SELECTION_OUTLINE_WIDTH);
  }
}
