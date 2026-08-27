import type { EpisodeStep } from "./api";

/** Pure formatting of the replay status line - testable without touching
 * the DOM (same seam as `grid.ts`/`dashboard.ts`).
 *
 * V3 (ADR-0002): a `commit` action can end an episode (`terminated`) without
 * matching the target, so "SOLVED" must key off `exact_match`, not
 * `terminated` - this is what lets a viewer tell "the agent painted onto a
 * scratch canvas and committed a wrong crop" apart from "the agent solved
 * it", both of which otherwise show as the episode ending. */
export function formatEpisodeStatus(
  step: EpisodeStep | null,
  currentIndex: number,
  frameCount: number
): string {
  const parts = [`step ${currentIndex} / ${frameCount - 1}`];
  if (!step) return parts.join(" | ");

  parts.push(`action: ${step.action.name ?? "(invalid)"}`, `reward: ${step.reward.toFixed(2)}`);

  const exactMatch = step.exact_match ?? step.terminated;
  if (exactMatch) {
    parts.push("SOLVED");
  } else if (step.action.name === "commit" && step.terminated) {
    parts.push("committed (no match)");
  }
  if (step.truncated) parts.push("truncated (max steps)");

  return parts.join(" | ");
}
