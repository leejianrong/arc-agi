import { describe, it, expect } from "vitest";
import { colorForSymbol, PALETTE } from "../src/palette";
import { computeCellRects, cellSizeFor } from "../src/grid";

describe("colorForSymbol", () => {
  // Transcribed from third_party/ARC-AGI/apps/css/common.css (.symbol_0..9),
  // per SLICES.md V1's unit test requirement.
  const expected: Record<number, string> = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
  };

  for (const [symbol, hex] of Object.entries(expected)) {
    it(`draws color ${hex} for symbol ${symbol}`, () => {
      expect(colorForSymbol(Number(symbol))).toBe(hex);
    });
  }

  it("has exactly 10 palette colors", () => {
    expect(PALETTE.length).toBe(10);
  });

  it("falls back to a pad color outside 0-9", () => {
    expect(colorForSymbol(10)).not.toBe(undefined);
    expect(PALETTE).not.toContain(colorForSymbol(-1));
  });
});

describe("computeCellRects", () => {
  it("maps each grid cell to its palette color and pixel position", () => {
    const grid = [
      [0, 1],
      [2, 3],
    ];
    const rects = computeCellRects(grid, 10);
    expect(rects).toEqual([
      { row: 0, col: 0, value: 0, color: "#000000", x: 0, y: 0, size: 10, selected: false },
      { row: 0, col: 1, value: 1, color: "#0074D9", x: 10, y: 0, size: 10, selected: false },
      { row: 1, col: 0, value: 2, color: "#FF4136", x: 0, y: 10, size: 10, selected: false },
      { row: 1, col: 1, value: 3, color: "#2ECC40", x: 10, y: 10, size: 10, selected: false },
    ]);
  });

  // ADR-0011/ADR-0012's object-selection mechanism: a step's `selected`
  // cells (visualizer selection-overlay slice).
  it("marks only the given cells as selected", () => {
    const grid = [
      [0, 1],
      [2, 3],
    ];
    const rects = computeCellRects(grid, 10, [[0, 1], [1, 0]]);
    const selected = rects.filter((r) => r.selected).map((r) => [r.row, r.col]);
    expect(selected).toEqual([
      [0, 1],
      [1, 0],
    ]);
  });

  it("marks nothing selected when selected is null or omitted", () => {
    const grid = [[0, 1]];
    expect(computeCellRects(grid, 10, null).every((r) => !r.selected)).toBe(true);
    expect(computeCellRects(grid, 10).every((r) => !r.selected)).toBe(true);
  });
});

describe("cellSizeFor", () => {
  it("fits the grid within the given bounds", () => {
    const grid = Array.from({ length: 10 }, () => Array(20).fill(0));
    const size = cellSizeFor(grid, 400, 400);
    expect(size * 20).toBeLessThanOrEqual(400);
    expect(size * 10).toBeLessThanOrEqual(400);
  });

  it("caps cell size for tiny grids", () => {
    const grid = [[0]];
    expect(cellSizeFor(grid, 1000, 1000)).toBeLessThanOrEqual(40);
  });
});
