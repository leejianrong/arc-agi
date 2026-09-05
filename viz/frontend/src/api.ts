import type { Grid } from "./grid";

// Mirrors arc_env/episode_log.py's JSONL record shapes 1:1 (ADR-0006,
// ADR-0007's "no translation layer beyond parsing" decision).

export interface RunSummary {
  run_id: string;
  algo: string;
  created_at: string;
  task_ids: string[];
}

export interface EpisodeStart {
  type: "start";
  episode_id: string;
  task_id: string;
  pair_index: number;
  grid: Grid;
  target_grid: Grid;
  max_steps: number;
}

export interface EpisodeStep {
  type: "step";
  step: number;
  grid_before: Grid;
  action: { name: string | null; args: Record<string, number> };
  grid_after: Grid;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  done: boolean;
  valid_action: boolean;
  // V3 (ADR-0002): `commit` can end an episode (terminated=true) without
  // matching the target - exact_match is what actually means "solved".
  // Optional for episodes logged before this field existed.
  exact_match?: boolean;
  // ADR-0011/ADR-0012's object-selection mechanism: the post-step selected
  // patch as `[row, col]` pairs, or `null`/absent when nothing is selected
  // (or the episode predates this field).
  selected?: [number, number][] | null;
}

export interface EpisodeEnd {
  type: "end";
  n_steps: number;
  success: boolean;
  total_reward: number;
}

export interface Episode {
  start: EpisodeStart;
  steps: EpisodeStep[];
  end: EpisodeEnd;
}

// One metrics.jsonl row (ADR-0006). Only the fields the dashboard actually
// reads are typed strictly; trainer-specific extras pass through untyped.
export interface MetricsRow {
  update: number;
  timestamp: string;
  n_episodes: number;
  mean_reward: number | null;
  success_rate: number | null;
  [key: string]: unknown;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${path}: HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function fetchRuns(): Promise<RunSummary[]> {
  return getJSON(`/api/runs`);
}

export function fetchEpisodeIds(runId: string): Promise<string[]> {
  return getJSON(`/api/runs/${encodeURIComponent(runId)}/episodes`);
}

export function fetchEpisode(runId: string, episodeId: string): Promise<Episode> {
  return getJSON(`/api/runs/${encodeURIComponent(runId)}/episodes/${encodeURIComponent(episodeId)}`);
}

export function fetchMetrics(runId: string): Promise<MetricsRow[]> {
  return getJSON(`/api/runs/${encodeURIComponent(runId)}/metrics`);
}

// One run's representative task grids (first train pair), for the run
// picker's thumbnail (Slice 3) - not episode data, so it's available even
// for a run with no logged episodes.
export interface RunThumbnail {
  task_id: string;
  input: Grid;
  output: Grid;
}

export function fetchThumbnail(runId: string): Promise<RunThumbnail> {
  return getJSON(`/api/runs/${encodeURIComponent(runId)}/thumbnail`);
}
