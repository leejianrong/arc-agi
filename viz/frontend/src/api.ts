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
