import { describe, it, expect } from "vitest";
import { EpisodePlayer } from "../src/player";
import type { Episode } from "../src/api";

function fixtureEpisode(): Episode {
  return {
    start: {
      type: "start",
      episode_id: "e",
      task_id: "t",
      pair_index: 0,
      grid: [[0]],
      target_grid: [[1]],
      max_steps: 2,
    },
    steps: [
      {
        type: "step",
        step: 0,
        grid_before: [[0]],
        action: { name: "fill_cell", args: {} },
        grid_after: [[1]],
        reward: 1.0,
        terminated: true,
        truncated: false,
        done: true,
        valid_action: true,
      },
    ],
    end: { type: "end", n_steps: 1, success: true, total_reward: 1.0 },
  };
}

describe("EpisodePlayer", () => {
  it("starts at frame 0 with the episode's initial grid", () => {
    const player = new EpisodePlayer(fixtureEpisode());
    expect(player.currentIndex).toBe(0);
    expect(player.currentGrid).toEqual([[0]]);
    expect(player.isAtStart).toBe(true);
  });

  it("steps forward through grid_after values and clamps at the end", () => {
    const player = new EpisodePlayer(fixtureEpisode());
    expect(player.stepForward()).toBe(true);
    expect(player.currentGrid).toEqual([[1]]);
    expect(player.isAtEnd).toBe(true);
    expect(player.stepForward()).toBe(false);
  });

  it("steps backward and clamps at the start", () => {
    const player = new EpisodePlayer(fixtureEpisode());
    player.stepForward();
    expect(player.stepBackward()).toBe(true);
    expect(player.currentGrid).toEqual([[0]]);
    expect(player.stepBackward()).toBe(false);
  });

  it("play() advances frames via the injected scheduler and stops at the end", () => {
    let tickFn: (() => void) | null = null;
    const scheduler = {
      setInterval: (fn: () => void) => {
        tickFn = fn;
        return 1;
      },
      clearInterval: () => {},
    };
    const player = new EpisodePlayer(fixtureEpisode(), scheduler);
    let ticks = 0;
    player.play(() => ticks++);
    expect(player.isPlaying).toBe(true);
    tickFn!(); // advance one frame (the only step in the fixture)
    expect(ticks).toBe(1);
    expect(player.currentGrid).toEqual([[1]]);
  });
});
