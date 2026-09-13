import { describe, it, expect } from "vitest";
import { colorForSymbol, PALETTE } from "../src/palette";
import { computeCellRects, cellSizeFor, drawGrid } from "../src/grid";

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
      { row: 0, col: 0, value: 0, color: "#000000", x: 0, y: 0, size: 10, selectedSlot: null },
      { row: 0, col: 1, value: 1, color: "#0074D9", x: 10, y: 0, size: 10, selectedSlot: null },
      { row: 1, col: 0, value: 2, color: "#FF4136", x: 0, y: 10, size: 10, selectedSlot: null },
      { row: 1, col: 1, value: 3, color: "#2ECC40", x: 10, y: 10, size: 10, selectedSlot: null },
    ]);
  });

  // ADR-0011/ADR-0012's object-selection mechanism: a step's `selected`
  // cells (visualizer selection-overlay slice).
  it("marks only slot a's cells as selected when slot b is empty", () => {
    const grid = [
      [0, 1],
      [2, 3],
    ];
    const rects = computeCellRects(grid, 10, { a: [[0, 1], [1, 0]], b: null });
    const selected = rects.filter((r) => r.selectedSlot === "a").map((r) => [r.row, r.col]);
    expect(selected).toEqual([
      [0, 1],
      [1, 0],
    ]);
    expect(rects.some((r) => r.selectedSlot === "b")).toBe(false);
  });

  it("marks nothing selected when selected is null or omitted", () => {
    const grid = [[0, 1]];
    expect(computeCellRects(grid, 10, null).every((r) => r.selectedSlot === null)).toBe(true);
    expect(computeCellRects(grid, 10).every((r) => r.selectedSlot === null)).toBe(true);
  });

  // ADR-0020: the new dual-slot selection - both slots render simultaneously,
  // each with its own distinct slot tag (slot "b" wins on overlap).
  it("marks slot a and slot b cells simultaneously with distinct slot tags", () => {
    const grid = [
      [0, 1, 2],
      [3, 4, 5],
    ];
    const rects = computeCellRects(grid, 10, { a: [[0, 0]], b: [[1, 2]] });
    const byCell = (row: number, col: number) =>
      rects.find((r) => r.row === row && r.col === col)?.selectedSlot;
    expect(byCell(0, 0)).toBe("a");
    expect(byCell(1, 2)).toBe("b");
    expect(byCell(0, 1)).toBe(null);
  });

  it("has slot b win on overlap between the two slots", () => {
    const grid = [[0, 1]];
    const rects = computeCellRects(grid, 10, { a: [[0, 0]], b: [[0, 0]] });
    expect(rects.find((r) => r.row === 0 && r.col === 0)?.selectedSlot).toBe("b");
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

// ADR-0020: a minimal fake 2D context (jsdom's own canvas support is too
// limited to assert on - see `computeCellRects`'s own doc comment above)
// recording every `strokeRect` call's `strokeStyle` at call time, so we can
// confirm slot "a" and slot "b" are actually drawn in two distinct colors.
class FakeCtx {
  strokeStyle = "";
  fillStyle = "";
  lineWidth = 0;
  strokeCalls: { style: string; x: number; y: number }[] = [];
  clearRect(): void {}
  fillRect(): void {}
  strokeRect(x: number, y: number): void {
    this.strokeCalls.push({ style: this.strokeStyle, x, y });
  }
}

describe("drawGrid", () => {
  it("draws slot a and slot b selection outlines in two distinct colors", () => {
    const grid = [
      [0, 1],
      [2, 3],
    ];
    const ctx = new FakeCtx();
    drawGrid(ctx as unknown as CanvasRenderingContext2D, grid, 10, { a: [[0, 0]], b: [[1, 1]] });

    // 4 cell fills + 4 cell borders + 1 slot-a outline + 1 slot-b outline.
    const outlineCalls = ctx.strokeCalls.filter((c) => c.style !== "#555555");
    expect(outlineCalls).toHaveLength(2);
    const styles = outlineCalls.map((c) => c.style);
    expect(new Set(styles).size).toBe(2); // two distinct colors
    expect(styles).toContain("#facc15"); // slot a - amber, backward-compatible
    expect(styles).toContain("#a78bfa"); // slot b - distinct
  });
});
