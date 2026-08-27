import type { MetricsRow } from "./api";

export interface Point {
  x: number;
  y: number;
}

const PADDING = 24;

/** Pure layout: maps a metric column to pixel points inside `width`x`height`,
 * skipping rows where that metric is null (e.g. an update with 0 completed
 * episodes). Testable without a Canvas 2D context, same seam as `grid.ts`'s
 * `computeCellRects`. */
export function computeLinePoints(
  rows: MetricsRow[],
  key: "mean_reward" | "success_rate",
  width: number,
  height: number
): Point[] {
  const values = rows
    .map((r) => ({ update: r.update, value: r[key] as number | null }))
    .filter((r): r is { update: number; value: number } => r.value !== null && r.value !== undefined);

  if (values.length === 0) return [];

  const minUpdate = values[0].update;
  const maxUpdate = values[values.length - 1].update;
  const updateSpan = Math.max(1, maxUpdate - minUpdate);

  const minValue = Math.min(0, ...values.map((v) => v.value));
  const maxValue = Math.max(...values.map((v) => v.value), minValue + 1e-9);
  const valueSpan = maxValue - minValue;

  const innerW = width - 2 * PADDING;
  const innerH = height - 2 * PADDING;

  return values.map((v) => ({
    x: PADDING + ((v.update - minUpdate) / updateSpan) * innerW,
    y: PADDING + innerH - ((v.value - minValue) / valueSpan) * innerH,
  }));
}

export function drawLineChart(
  ctx: CanvasRenderingContext2D,
  points: Point[],
  width: number,
  height: number,
  color: string,
  label: string
): void {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#444";
  ctx.strokeRect(PADDING, PADDING, width - 2 * PADDING, height - 2 * PADDING);

  ctx.fillStyle = "#ccc";
  ctx.font = "12px sans-serif";
  ctx.fillText(label, PADDING, 14);

  if (points.length === 0) {
    ctx.fillText("no data yet", width / 2 - 30, height / 2);
    return;
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
  ctx.stroke();

  ctx.fillStyle = color;
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2, 0, 2 * Math.PI);
    ctx.fill();
  }
}
