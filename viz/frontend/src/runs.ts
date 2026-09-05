import type { RunSummary } from "./api";

export interface RunDateGroup {
  date: string;
  runs: RunSummary[];
}

/** Pure grouping: buckets runs by the date portion of `created_at` (the ISO
 * string's part before "T"), most recent date first. Within a group, runs
 * are ordered newest-first by full `created_at` timestamp. Runs missing
 * `created_at` land in their own "unknown date" group, sorted after every
 * dated group - never dropped, never silently merged into a real date.
 * Testable without touching the DOM, same seam as `dashboard.ts`'s
 * `computeLinePoints` / `status.ts`'s `formatEpisodeStatus`. */
export function groupRunsByDate(runs: RunSummary[]): RunDateGroup[] {
  const byDate = new Map<string, RunSummary[]>();
  const UNKNOWN = "unknown date";

  for (const run of runs) {
    const date = run.created_at ? run.created_at.split("T")[0] : UNKNOWN;
    const bucket = byDate.get(date);
    if (bucket) {
      bucket.push(run);
    } else {
      byDate.set(date, [run]);
    }
  }

  for (const bucket of byDate.values()) {
    bucket.sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""));
  }

  const dates = [...byDate.keys()].sort((a, b) => {
    if (a === UNKNOWN) return 1;
    if (b === UNKNOWN) return -1;
    return b.localeCompare(a);
  });

  return dates.map((date) => ({ date, runs: byDate.get(date)! }));
}
