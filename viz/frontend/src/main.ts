import { drawGrid, cellSizeFor, type Grid, type SelectionCells } from "./grid";
import { fetchRuns, fetchEpisodeIds, fetchEpisode, fetchMetrics, type Episode } from "./api";
import { EpisodePlayer } from "./player";
import { computeLinePoints, drawLineChart } from "./dashboard";
import { formatEpisodeStatus } from "./status";
import { RunPicker } from "./picker";

const runPicker = new RunPicker(document.getElementById("run-picker")!, (runId) => void loadRun(runId));
const rewardChart = document.getElementById("reward-chart") as HTMLCanvasElement;
const successChart = document.getElementById("success-chart") as HTMLCanvasElement;

function renderGridToCanvas(canvas: HTMLCanvasElement, grid: Grid, selected?: SelectionCells): void {
  const cellSize = cellSizeFor(grid, 320, 320);
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  const ctx = canvas.getContext("2d");
  if (ctx) drawGrid(ctx, grid, cellSize, selected);
}

/** One independent replay panel: its own episode picker, play/pause/step/
 * speed controls, and grid canvases - built so two of these, wired to the
 * same run's episode list, let you compare an early- vs. late-training
 * episode side by side. */
class PlayerPanel {
  private root: HTMLElement;
  private episodeSelect!: HTMLSelectElement;
  private canvasCurrent!: HTMLCanvasElement;
  private canvasTarget!: HTMLCanvasElement;
  private statusEl!: HTMLDivElement;
  private btnPlay!: HTMLButtonElement;
  private player: EpisodePlayer | null = null;
  private runId = "";

  constructor(root: HTMLElement, label: string) {
    this.root = root;
    this.root.innerHTML = `
      <strong>Panel ${label}</strong>
      <label>Episode: <select class="episode-select"></select></label>
      <div class="player-controls">
        <button class="btn-back">&#9664; step</button>
        <button class="btn-play">&#9654; play</button>
        <button class="btn-forward">step &#9654;</button>
        <label>Speed: <input class="speed" type="range" min="50" max="2000" step="50" value="500" /></label>
      </div>
      <div class="player-grids">
        <div class="grid-panel"><span>Agent's grid</span><canvas class="grid-canvas canvas-current"></canvas></div>
        <div class="grid-panel"><span>Target</span><canvas class="grid-canvas canvas-target"></canvas></div>
      </div>
      <div class="player-status"></div>
    `;
    this.episodeSelect = this.root.querySelector(".episode-select")!;
    this.canvasCurrent = this.root.querySelector(".canvas-current")!;
    this.canvasTarget = this.root.querySelector(".canvas-target")!;
    this.statusEl = this.root.querySelector(".player-status")!;
    this.btnPlay = this.root.querySelector(".btn-play")!;

    this.root.querySelector(".btn-forward")!.addEventListener("click", () => {
      this.player?.stepForward();
      this.renderFrame();
    });
    this.root.querySelector(".btn-back")!.addEventListener("click", () => {
      this.player?.stepBackward();
      this.renderFrame();
    });
    this.btnPlay.addEventListener("click", () => {
      if (!this.player) return;
      if (this.player.isPlaying) {
        this.player.pause();
      } else {
        this.player.play(() => this.renderFrame());
      }
      this.renderFrame();
    });
    (this.root.querySelector(".speed") as HTMLInputElement).addEventListener("input", (e) => {
      this.player?.setSpeedMs(Number((e.target as HTMLInputElement).value));
    });
    this.episodeSelect.addEventListener("change", () => void this.loadEpisode(this.episodeSelect.value));
  }

  async setRun(runId: string, episodeIds: string[], preferredIndex: number): Promise<void> {
    this.runId = runId;
    this.episodeSelect.innerHTML = "";
    for (const id of episodeIds) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      this.episodeSelect.appendChild(opt);
    }
    if (episodeIds.length === 0) return;
    const index = Math.min(Math.max(preferredIndex, 0), episodeIds.length - 1);
    this.episodeSelect.selectedIndex = index;
    await this.loadEpisode(episodeIds[index]);
  }

  private async loadEpisode(episodeId: string): Promise<void> {
    this.player?.pause();
    const episode: Episode = await fetchEpisode(this.runId, episodeId);
    this.player = new EpisodePlayer(episode);
    renderGridToCanvas(this.canvasTarget, episode.start.target_grid);
    this.renderFrame();
  }

  private renderFrame(): void {
    if (!this.player) return;
    // `currentStep` is null at frame 0 (the episode's starting grid, before
    // any action) - nothing is selected yet, so no `selected` is passed.
    renderGridToCanvas(this.canvasCurrent, this.player.currentGrid, this.player.currentStep?.selected);
    this.statusEl.textContent = formatEpisodeStatus(
      this.player.currentStep,
      this.player.currentIndex,
      this.player.frameCount
    );
    this.btnPlay.textContent = this.player.isPlaying ? "⏸ pause" : "▶ play";
  }
}

const panelA = new PlayerPanel(document.getElementById("panel-a")!, "A");
const panelB = new PlayerPanel(document.getElementById("panel-b")!, "B");

function renderDashboard(rows: Awaited<ReturnType<typeof fetchMetrics>>): void {
  const rewardPoints = computeLinePoints(rows, "mean_reward", rewardChart.width, rewardChart.height);
  const successPoints = computeLinePoints(rows, "success_rate", successChart.width, successChart.height);
  const rewardCtx = rewardChart.getContext("2d");
  const successCtx = successChart.getContext("2d");
  if (rewardCtx) drawLineChart(rewardCtx, rewardPoints, rewardChart.width, rewardChart.height, "#2ECC40", "mean reward / update");
  if (successCtx) drawLineChart(successCtx, successPoints, successChart.width, successChart.height, "#0074D9", "success rate / update");
}

async function loadRuns(): Promise<void> {
  const runs = await fetchRuns();
  await runPicker.setRuns(runs);
}

async function loadRun(runId: string): Promise<void> {
  const [episodeIds, metrics] = await Promise.all([fetchEpisodeIds(runId), fetchMetrics(runId)]);
  renderDashboard(metrics);
  // Panel A defaults to the earliest episode, Panel B to the latest - the
  // early-vs-late-training comparison the demo target asks for. For a run
  // with only one episode (e.g. V1's random rollouts), both show it.
  await panelA.setRun(runId, episodeIds, 0);
  await panelB.setRun(runId, episodeIds, episodeIds.length - 1);
}

void loadRuns();
