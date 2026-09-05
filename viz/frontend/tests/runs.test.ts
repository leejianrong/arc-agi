import { describe, it, expect } from "vitest";
import { groupRunsByDate } from "../src/runs";
import type { RunSummary } from "../src/api";

function run(run_id: string, created_at: string, algo = "ppo"): RunSummary {
  return { run_id, algo, created_at, task_ids: [] };
}

describe("groupRunsByDate", () => {
  it("returns an empty array for no runs", () => {
    expect(groupRunsByDate([])).toEqual([]);
  });

  it("groups runs on the same date into one bucket", () => {
    const runs = [
      run("a", "2026-09-04T15:53:52Z"),
      run("b", "2026-09-04T15:55:57Z"),
    ];
    const groups = groupRunsByDate(runs);
    expect(groups).toHaveLength(1);
    expect(groups[0].date).toBe("2026-09-04");
    expect(groups[0].runs.map((r) => r.run_id)).toEqual(["b", "a"]); // newest-first within group
  });

  it("orders groups most-recent-date first", () => {
    const runs = [
      run("old", "2026-08-30T18:53:04Z"),
      run("new", "2026-09-04T15:53:52Z"),
      run("mid", "2026-09-01T00:00:00Z"),
    ];
    const groups = groupRunsByDate(runs);
    expect(groups.map((g) => g.date)).toEqual(["2026-09-04", "2026-09-01", "2026-08-30"]);
  });

  it("puts runs with missing created_at in their own group, sorted last", () => {
    const runs = [
      run("dated", "2026-09-04T15:53:52Z"),
      { run_id: "no-date", algo: "ppo", created_at: "", task_ids: [] } as RunSummary,
    ];
    const groups = groupRunsByDate(runs);
    expect(groups).toHaveLength(2);
    expect(groups[groups.length - 1].date).toBe("unknown date");
    expect(groups[groups.length - 1].runs.map((r) => r.run_id)).toEqual(["no-date"]);
  });

  it("handles many distinct dates and many runs per date", () => {
    const runs = [
      run("d1-a", "2026-09-01T00:00:00Z"),
      run("d3-a", "2026-09-03T00:00:00Z"),
      run("d1-b", "2026-09-01T12:00:00Z"),
      run("d2-a", "2026-09-02T00:00:00Z"),
    ];
    const groups = groupRunsByDate(runs);
    expect(groups.map((g) => g.date)).toEqual(["2026-09-03", "2026-09-02", "2026-09-01"]);
    expect(groups[2].runs.map((r) => r.run_id)).toEqual(["d1-b", "d1-a"]);
  });
});
