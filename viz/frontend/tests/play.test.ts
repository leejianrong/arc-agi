import { describe, it, expect } from "vitest";
import { parseArgValues, argInputLabel, formatPlayStatus } from "../src/play";
import type { PlayState } from "../src/api";

function state(overrides: Partial<PlayState> = {}): PlayState {
  return {
    session_id: "s1",
    task_id: "67a3c6ac",
    pair_index: 0,
    grid: [[0]],
    target_grid: [[0]],
    selected: null,
    terminated: false,
    truncated: false,
    valid_action: true,
    exact_match: false,
    ...overrides,
  };
}

describe("parseArgValues", () => {
  it("parses valid integer strings", () => {
    expect(parseArgValues(["1", "2", "30"])).toEqual([1, 2, 30]);
  });

  it("defaults empty or non-numeric input to 0", () => {
    expect(parseArgValues(["", "abc", "3.7"])).toEqual([0, 0, 3]);
  });

  it("handles negative numbers", () => {
    expect(parseArgValues(["-1"])).toEqual([-1]);
  });

  it("returns an empty array for an empty input list", () => {
    expect(parseArgValues([])).toEqual([]);
  });
});

describe("argInputLabel", () => {
  it("appends the kind when it differs from the name", () => {
    expect(argInputLabel("row", "coord")).toBe("row (coord)");
  });

  it("leaves the label bare when name and kind match", () => {
    expect(argInputLabel("color", "color")).toBe("color");
  });
});

describe("formatPlayStatus", () => {
  it("prompts for a task_id when there is no session yet", () => {
    expect(formatPlayStatus(null)).toContain("no session yet");
  });

  it("surfaces an error message over any session state", () => {
    expect(formatPlayStatus(null, "unknown task_id: nope")).toBe("error: unknown task_id: nope");
    expect(formatPlayStatus(state(), "boom")).toBe("error: boom");
  });

  it("labels an exact-match step SOLVED", () => {
    const s = state({ terminated: true, exact_match: true, reward: 1.0 });
    expect(formatPlayStatus(s)).toContain("SOLVED");
  });

  it("labels a non-matching commit as 'committed (no match)', not SOLVED", () => {
    const s = state({ terminated: true, exact_match: false, reward: 0.1 });
    const text = formatPlayStatus(s);
    expect(text).toContain("committed (no match)");
    expect(text).not.toContain("SOLVED");
  });

  it("flags a truncated session", () => {
    expect(formatPlayStatus(state({ truncated: true }))).toContain("truncated (max steps)");
  });

  it("flags an invalid (no-op) action", () => {
    expect(formatPlayStatus(state({ valid_action: false }))).toContain("invalid action (no-op)");
  });

  it("shows the reward when present", () => {
    expect(formatPlayStatus(state({ reward: 0.5 }))).toContain("reward: 0.50");
  });

  it("falls back to 'ready' for a fresh, unterminated session with no reward yet", () => {
    expect(formatPlayStatus(state())).toBe("ready");
  });
});
