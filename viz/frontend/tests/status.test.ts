import { describe, it, expect } from "vitest";
import { formatEpisodeStatus } from "../src/status";
import type { EpisodeStep } from "../src/api";

function step(overrides: Partial<EpisodeStep> = {}): EpisodeStep {
  return {
    type: "step",
    step: 0,
    grid_before: [[0]],
    action: { name: "vmirror", args: {} },
    grid_after: [[0]],
    reward: 0.5,
    terminated: false,
    truncated: false,
    done: false,
    valid_action: true,
    ...overrides,
  };
}

describe("formatEpisodeStatus", () => {
  it("shows just the frame position at the start frame (no step yet)", () => {
    expect(formatEpisodeStatus(null, 0, 5)).toBe("step 0 / 4");
  });

  it("labels an exact-match step SOLVED", () => {
    const s = step({ terminated: true, exact_match: true });
    expect(formatEpisodeStatus(s, 1, 2)).toContain("SOLVED");
  });

  // V3 (ADR-0002): a commit that ends the episode without matching must
  // read differently from a genuine solve - this is the "scratch canvas
  // mid-episode" vs "committed final grid" distinction the visualizer
  // needs to make.
  it("labels a non-matching commit as 'committed (no match)', not SOLVED", () => {
    const s = step({
      action: { name: "commit", args: { row: 0, col: 0, height: 3, width: 3 } },
      terminated: true,
      exact_match: false,
    });
    const status = formatEpisodeStatus(s, 1, 2);
    expect(status).toContain("committed (no match)");
    expect(status).not.toContain("SOLVED");
  });

  it("labels a matching commit SOLVED, not 'committed (no match)'", () => {
    const s = step({
      action: { name: "commit", args: { row: 0, col: 0, height: 2, width: 2 } },
      terminated: true,
      exact_match: true,
    });
    const status = formatEpisodeStatus(s, 1, 1);
    expect(status).toContain("SOLVED");
    expect(status).not.toContain("committed (no match)");
  });

  it("falls back to terminated when exact_match is absent (pre-V3 episode logs)", () => {
    const s = step({ terminated: true, exact_match: undefined });
    expect(formatEpisodeStatus(s, 1, 2)).toContain("SOLVED");
  });

  it("labels a truncated (non-terminated) episode without SOLVED or commit text", () => {
    const s = step({ truncated: true });
    const status = formatEpisodeStatus(s, 9, 10);
    expect(status).toContain("truncated (max steps)");
    expect(status).not.toContain("SOLVED");
    expect(status).not.toContain("committed");
  });

  it("includes the action name and reward", () => {
    const s = step({ action: { name: "canvas", args: { value: 3, height: 4, width: 4 } }, reward: -0.01 });
    const status = formatEpisodeStatus(s, 1, 3);
    expect(status).toContain("action: canvas");
    expect(status).toContain("reward: -0.01");
  });
});
