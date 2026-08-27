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
}

const MAX_CELL_SIZE = 40;

export function cellSizeFor(grid: Grid, maxWidth: number, maxHeight: number): number {
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  if (rows === 0 || cols === 0) return MAX_CELL_SIZE;
  return Math.max(1, Math.min(MAX_CELL_SIZE, Math.floor(maxWidth / cols), Math.floor(maxHeight / rows)));
}

// Pure layout function - lets the palette/geometry logic be unit-tested
// without a real Canvas 2D context (jsdom's canvas support is limited).
export function computeCellRects(grid: Grid, cellSize: number): CellRect[] {
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
      });
    }
  }
  return rects;
}

export function drawGrid(ctx: CanvasRenderingContext2D, grid: Grid, cellSize: number): void {
  const rects = computeCellRects(grid, cellSize);
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  ctx.clearRect(0, 0, cols * cellSize, rows * cellSize);
  for (const rect of rects) {
    ctx.fillStyle = rect.color;
    ctx.fillRect(rect.x, rect.y, rect.size, rect.size);
    ctx.strokeStyle = "#555555";
    ctx.strokeRect(rect.x, rect.y, rect.size, rect.size);
  }
}
