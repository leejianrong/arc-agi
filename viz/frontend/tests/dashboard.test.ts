import { describe, it, expect } from "vitest";
import { computeLinePoints } from "../src/dashboard";
import type { MetricsRow } from "../src/api";

function row(update: number, mean_reward: number | null, success_rate: number | null): MetricsRow {
  return { update, timestamp: `t${update}`, n_episodes: 1, mean_reward, success_rate };
}

describe("computeLinePoints", () => {
  it("skips rows where the metric is null", () => {
    const rows = [row(0, 0.1, 0.0), row(1, null, null), row(2, 0.9, 1.0)];
    const points = computeLinePoints(rows, "mean_reward", 100, 100);
    expect(points).toHaveLength(2);
  });

  it("maps the first and last update to the chart's horizontal extremes", () => {
    const rows = [row(0, 0.0, 0), row(10, 1.0, 1)];
    const points = computeLinePoints(rows, "mean_reward", 100, 100);
    expect(points[0].x).toBeCloseTo(24); // PADDING
    expect(points[1].x).toBeCloseTo(76); // width - PADDING
  });

  it("maps the minimum value to the bottom and maximum to the top", () => {
    const rows = [row(0, -1.0, 0), row(1, 2.0, 0)];
    const points = computeLinePoints(rows, "mean_reward", 100, 100);
    expect(points[0].y).toBeGreaterThan(points[1].y); // lower value -> lower on screen (higher y)
  });

  it("returns an empty array when every value is null", () => {
    const rows = [row(0, null, null)];
    expect(computeLinePoints(rows, "success_rate", 100, 100)).toEqual([]);
  });

  it("handles a single data point without dividing by zero", () => {
    const rows = [row(5, 0.5, 0.5)];
    const points = computeLinePoints(rows, "success_rate", 100, 100);
    expect(points).toHaveLength(1);
    expect(Number.isFinite(points[0].x)).toBe(true);
    expect(Number.isFinite(points[0].y)).toBe(true);
  });
});
