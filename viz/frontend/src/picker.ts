import { fetchThumbnail, type RunSummary, type RunThumbnail } from "./api";
import { groupRunsByDate, representativeRunIdByTaskId } from "./runs";
import { drawGrid, cellSizeFor, type Grid } from "./grid";

const THUMB_MAX = 56;

function renderThumbCanvas(grid: Grid): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.className = "run-picker-thumb";
  const cellSize = cellSizeFor(grid, THUMB_MAX, THUMB_MAX);
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  const ctx = canvas.getContext("2d");
  if (ctx) drawGrid(ctx, grid, cellSize);
  return canvas;
}

/** Replaces the native `<select>` run picker (Slice 3): a toggle button
 * showing the selected run, opening a panel of run cards grouped by date
 * (`groupRunsByDate`, most-recent-first), each with a small input->output
 * thumbnail so a run is recognizable before it's opened. Thumbnails are
 * fetched once per unique task_id (`representativeRunIdByTaskId`), not once
 * per run, since runs commonly share a task_id (e.g. a `<task>-gp`/
 * `<task>-ppo` pair). */
export class RunPicker {
  private root: HTMLElement;
  private toggle: HTMLButtonElement;
  private panel: HTMLDivElement;
  private onSelect: (runId: string) => void;
  private runs: RunSummary[] = [];
  private selectedRunId = "";

  constructor(root: HTMLElement, onSelect: (runId: string) => void) {
    this.root = root;
    this.onSelect = onSelect;
    this.root.innerHTML = `
      <button type="button" class="run-picker-toggle">loading runs...</button>
      <div class="run-picker-panel" hidden></div>
    `;
    this.toggle = this.root.querySelector(".run-picker-toggle")!;
    this.panel = this.root.querySelector(".run-picker-panel")!;

    this.toggle.addEventListener("click", () => {
      this.panel.hidden = !this.panel.hidden;
    });
    document.addEventListener("click", (event) => {
      if (!this.root.contains(event.target as Node)) this.panel.hidden = true;
    });
  }

  get selectedRun(): string {
    return this.selectedRunId;
  }

  async setRuns(runs: RunSummary[]): Promise<void> {
    this.runs = runs;
    if (runs.length === 0) {
      this.toggle.textContent = "(no runs)";
      this.panel.innerHTML = "";
      return;
    }

    const thumbnails = await this.loadThumbnails(runs);
    this.renderPanel(runs, thumbnails);
    this.select(runs[0].run_id);
  }

  private async loadThumbnails(runs: RunSummary[]): Promise<Map<string, RunThumbnail>> {
    const representatives = representativeRunIdByTaskId(runs);
    const thumbnails = new Map<string, RunThumbnail>();
    await Promise.all(
      [...representatives.entries()].map(async ([taskId, runId]) => {
        try {
          thumbnails.set(taskId, await fetchThumbnail(runId));
        } catch {
          // No thumbnail available for this task (e.g. missing task data) -
          // the card below just renders without one.
        }
      })
    );
    return thumbnails;
  }

  private renderPanel(runs: RunSummary[], thumbnails: Map<string, RunThumbnail>): void {
    this.panel.innerHTML = "";
    for (const group of groupRunsByDate(runs)) {
      const heading = document.createElement("div");
      heading.className = "run-picker-group-label";
      heading.textContent = group.date;
      this.panel.appendChild(heading);

      for (const run of group.runs) {
        this.panel.appendChild(this.renderCard(run, thumbnails.get(run.task_ids[0])));
      }
    }
  }

  private renderCard(run: RunSummary, thumbnail: RunThumbnail | undefined): HTMLButtonElement {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "run-picker-card";
    card.dataset.runId = run.run_id;

    const label = document.createElement("div");
    label.className = "run-picker-card-label";
    label.textContent = `${run.run_id} (${run.algo})`;
    card.appendChild(label);

    if (thumbnail) {
      const row = document.createElement("div");
      row.className = "run-picker-thumb-row";
      row.appendChild(renderThumbCanvas(thumbnail.input));
      row.appendChild(renderThumbCanvas(thumbnail.output));
      card.appendChild(row);
    }

    card.addEventListener("click", () => this.select(run.run_id));
    return card;
  }

  private select(runId: string): void {
    this.selectedRunId = runId;
    const run = this.runs.find((r) => r.run_id === runId);
    this.toggle.textContent = run ? `${run.run_id} (${run.algo})` : runId;
    this.panel.hidden = true;
    this.panel.querySelectorAll<HTMLButtonElement>(".run-picker-card").forEach((card) => {
      card.classList.toggle("selected", card.dataset.runId === runId);
    });
    this.onSelect(runId);
  }
}
