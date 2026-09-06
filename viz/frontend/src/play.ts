import { drawGrid, cellSizeFor, type Grid, type SelectionCells } from "./grid";
import {
  fetchActions,
  startPlaySession,
  stepPlaySession,
  savePlaySession,
  type ActionSpec,
  type PlayState,
} from "./api";

function renderGridToCanvas(canvas: HTMLCanvasElement, grid: Grid, selected?: SelectionCells): void {
  const cellSize = cellSizeFor(grid, 320, 320);
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  const ctx = canvas.getContext("2d");
  if (ctx) drawGrid(ctx, grid, cellSize, selected);
}

/** Parses each arg input's raw text into an int, defaulting to 0 for empty
 * or non-numeric input - Stage 0 keeps the arg UI cheap (raw
 * `RAW_ARG_RANGE`-scale ints, no decode-aware widgets per kind), so this is
 * the one bit of validation standing between a human's typing and the
 * network request. Pure/testable, same seam as `grid.ts`'s
 * `computeCellRects` / `dashboard.ts`'s `computeLinePoints`. */
export function parseArgValues(raws: string[]): number[] {
  return raws.map((raw) => {
    const n = Number.parseInt(raw, 10);
    return Number.isFinite(n) ? n : 0;
  });
}

/** One arg slot's input label - appends its `kind` (color/factor/coord/dim/
 * direction) only when that's not already redundant with the arg's own
 * name, so a human has some idea what range to type. */
export function argInputLabel(name: string, kind: string): string {
  return name === kind ? name : `${name} (${kind})`;
}

/** Pure status-line formatting for the play panel, mirroring `status.ts`'s
 * `formatEpisodeStatus` for replay: "SOLVED" keys off `exact_match`, not the
 * broader `terminated` (a `commit`/`commit_selection` action can end the
 * episode on a chosen crop without matching the target). */
export function formatPlayStatus(state: PlayState | null, error?: string | null): string {
  if (error) return `error: ${error}`;
  if (!state) return "no session yet - enter a task_id and press Start";

  const parts: string[] = [];
  if (state.reward !== undefined) parts.push(`reward: ${state.reward.toFixed(2)}`);
  if (state.exact_match) {
    parts.push("SOLVED");
  } else if (state.terminated) {
    parts.push("committed (no match)");
  }
  if (state.truncated) parts.push("truncated (max steps)");
  if (!state.valid_action) parts.push("invalid action (no-op)");

  return parts.length > 0 ? parts.join(" | ") : "ready";
}

/** F13 Stage 0 (ADR-0017): a minimal interactive "play" panel - a human
 * solves an ARC-AGI-1 task by hand, one curated action at a time, against a
 * live `ArcEnv` session on the backend (`viz/backend/play.py`). Mirrors
 * `PlayerPanel` (`main.ts`)'s rendering pattern (grid + target canvases via
 * `grid.ts`), but drives a live session instead of replaying a logged one. */
export class PlayPanel {
  private root: HTMLElement;
  private taskIdInput!: HTMLInputElement;
  private pairIndexInput!: HTMLInputElement;
  private btnStart!: HTMLButtonElement;
  private primitiveSelect!: HTMLSelectElement;
  private argsContainer!: HTMLElement;
  private btnStep!: HTMLButtonElement;
  private btnSave!: HTMLButtonElement;
  private statusEl!: HTMLElement;
  private saveStatusEl!: HTMLElement;
  private canvasGrid!: HTMLCanvasElement;
  private canvasTarget!: HTMLCanvasElement;

  private actionSpecs: ActionSpec[] = [];
  private state: PlayState | null = null;

  constructor(root: HTMLElement) {
    this.root = root;
    this.root.innerHTML = `
      <div class="play-start">
        <label>task_id: <input class="play-task-id" type="text" placeholder="e.g. 67a3c6ac" /></label>
        <label>pair index: <input class="play-pair-index" type="number" value="0" min="0" /></label>
        <button type="button" class="play-btn-start">Start</button>
      </div>
      <div class="player-grids">
        <div class="grid-panel"><span>Your grid</span><canvas class="grid-canvas play-canvas-grid"></canvas></div>
        <div class="grid-panel"><span>Target</span><canvas class="grid-canvas play-canvas-target"></canvas></div>
      </div>
      <div class="play-action-picker">
        <label>Action: <select class="play-primitive-select"></select></label>
        <span class="play-args"></span>
        <button type="button" class="play-btn-step">Step</button>
      </div>
      <div class="player-status play-status"></div>
      <div class="play-save">
        <button type="button" class="play-btn-save">Save as run</button>
        <span class="play-save-status"></span>
      </div>
    `;
    this.taskIdInput = this.root.querySelector(".play-task-id")!;
    this.pairIndexInput = this.root.querySelector(".play-pair-index")!;
    this.btnStart = this.root.querySelector(".play-btn-start")!;
    this.primitiveSelect = this.root.querySelector(".play-primitive-select")!;
    this.argsContainer = this.root.querySelector(".play-args")!;
    this.btnStep = this.root.querySelector(".play-btn-step")!;
    this.btnSave = this.root.querySelector(".play-btn-save")!;
    this.statusEl = this.root.querySelector(".play-status")!;
    this.saveStatusEl = this.root.querySelector(".play-save-status")!;
    this.canvasGrid = this.root.querySelector(".play-canvas-grid")!;
    this.canvasTarget = this.root.querySelector(".play-canvas-target")!;

    this.setPlayingEnabled(false);
    this.statusEl.textContent = formatPlayStatus(null);

    this.btnStart.addEventListener("click", () => void this.start());
    this.primitiveSelect.addEventListener("change", () => this.renderArgInputs());
    this.btnStep.addEventListener("click", () => void this.step());
    this.btnSave.addEventListener("click", () => void this.save());

    void this.loadActions();
  }

  private setPlayingEnabled(enabled: boolean): void {
    this.primitiveSelect.disabled = !enabled;
    this.btnStep.disabled = !enabled;
    this.btnSave.disabled = !enabled;
  }

  private async loadActions(): Promise<void> {
    this.actionSpecs = await fetchActions();
    this.primitiveSelect.innerHTML = "";
    for (const action of this.actionSpecs) {
      const opt = document.createElement("option");
      opt.value = action.name;
      opt.textContent = `${action.name} (${action.kind})`;
      this.primitiveSelect.appendChild(opt);
    }
    this.renderArgInputs();
  }

  private renderArgInputs(): void {
    this.argsContainer.innerHTML = "";
    const action = this.actionSpecs.find((a) => a.name === this.primitiveSelect.value);
    if (!action) return;
    for (const arg of action.args) {
      const label = document.createElement("label");
      label.className = "play-arg-label";
      label.textContent = `${argInputLabel(arg.name, arg.kind)}: `;
      const input = document.createElement("input");
      input.type = "number";
      input.className = "play-arg-input";
      input.value = "0";
      label.appendChild(input);
      this.argsContainer.appendChild(label);
    }
  }

  private currentArgValues(): number[] {
    const raws = [...this.argsContainer.querySelectorAll<HTMLInputElement>(".play-arg-input")].map((i) => i.value);
    return parseArgValues(raws);
  }

  private async start(): Promise<void> {
    const taskId = this.taskIdInput.value.trim();
    const pairIndex = Number.parseInt(this.pairIndexInput.value, 10) || 0;
    this.saveStatusEl.textContent = "";
    try {
      this.state = await startPlaySession(taskId, pairIndex);
      this.setPlayingEnabled(true);
      this.renderFrame();
    } catch (err) {
      this.state = null;
      this.setPlayingEnabled(false);
      this.statusEl.textContent = formatPlayStatus(null, (err as Error).message);
    }
  }

  private async step(): Promise<void> {
    if (!this.state) return;
    const primitive = this.primitiveSelect.value;
    const args = this.currentArgValues();
    try {
      this.state = await stepPlaySession(this.state.session_id, primitive, args);
      this.renderFrame();
    } catch (err) {
      this.statusEl.textContent = formatPlayStatus(this.state, (err as Error).message);
    }
  }

  private async save(): Promise<void> {
    if (!this.state) return;
    try {
      const saved = await savePlaySession(this.state.session_id);
      this.saveStatusEl.textContent = `Saved as ${saved.run_id} - refresh the run picker above to see it.`;
    } catch (err) {
      this.saveStatusEl.textContent = `save failed: ${(err as Error).message}`;
    }
  }

  private renderFrame(): void {
    if (!this.state) return;
    renderGridToCanvas(this.canvasGrid, this.state.grid, this.state.selected);
    renderGridToCanvas(this.canvasTarget, this.state.target_grid);
    this.statusEl.textContent = formatPlayStatus(this.state);
  }
}
