import { drawGrid, cellSizeFor, type Grid } from "./grid";
import { fetchRuns, fetchEpisodeIds, fetchEpisode, type Episode } from "./api";
import { EpisodePlayer } from "./player";

const runSelect = document.getElementById("run-select") as HTMLSelectElement;
const episodeSelect = document.getElementById("episode-select") as HTMLSelectElement;
const btnBack = document.getElementById("btn-back") as HTMLButtonElement;
const btnPlay = document.getElementById("btn-play") as HTMLButtonElement;
const btnForward = document.getElementById("btn-forward") as HTMLButtonElement;
const speedInput = document.getElementById("speed") as HTMLInputElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const canvasCurrent = document.getElementById("canvas-current") as HTMLCanvasElement;
const canvasTarget = document.getElementById("canvas-target") as HTMLCanvasElement;

let player: EpisodePlayer | null = null;

function renderGridToCanvas(canvas: HTMLCanvasElement, grid: Grid): void {
  const cellSize = cellSizeFor(grid, 480, 480);
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  const ctx = canvas.getContext("2d");
  if (ctx) drawGrid(ctx, grid, cellSize);
}

function renderFrame(): void {
  if (!player) return;
  renderGridToCanvas(canvasCurrent, player.currentGrid);
  const step = player.currentStep;
  const parts = [`step ${player.currentIndex} / ${player.frameCount - 1}`];
  if (step) {
    parts.push(`action: ${step.action.name ?? "(invalid)"}`, `reward: ${step.reward.toFixed(2)}`);
    if (step.terminated) parts.push("SOLVED");
    if (step.truncated) parts.push("truncated (max steps)");
  }
  statusEl.textContent = parts.join(" | ");
  btnPlay.textContent = player.isPlaying ? "⏸ pause" : "▶ play";
}

async function loadRuns(): Promise<void> {
  const runs = await fetchRuns();
  runSelect.innerHTML = "";
  for (const run of runs) {
    const opt = document.createElement("option");
    opt.value = run.run_id;
    opt.textContent = `${run.run_id} (${run.algo})`;
    runSelect.appendChild(opt);
  }
  if (runs.length > 0) await loadEpisodesForRun(runs[0].run_id);
}

async function loadEpisodesForRun(runId: string): Promise<void> {
  const episodeIds = await fetchEpisodeIds(runId);
  episodeSelect.innerHTML = "";
  for (const id of episodeIds) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = id;
    episodeSelect.appendChild(opt);
  }
  if (episodeIds.length > 0) await loadEpisode(runId, episodeIds[0]);
}

async function loadEpisode(runId: string, episodeId: string): Promise<void> {
  player?.pause();
  const episode: Episode = await fetchEpisode(runId, episodeId);
  player = new EpisodePlayer(episode);
  renderGridToCanvas(canvasTarget, episode.start.target_grid);
  renderFrame();
}

runSelect.addEventListener("change", () => void loadEpisodesForRun(runSelect.value));
episodeSelect.addEventListener("change", () => void loadEpisode(runSelect.value, episodeSelect.value));

btnForward.addEventListener("click", () => {
  player?.stepForward();
  renderFrame();
});
btnBack.addEventListener("click", () => {
  player?.stepBackward();
  renderFrame();
});
btnPlay.addEventListener("click", () => {
  if (!player) return;
  if (player.isPlaying) {
    player.pause();
  } else {
    player.play(renderFrame);
  }
  renderFrame();
});
speedInput.addEventListener("input", () => {
  player?.setSpeedMs(Number(speedInput.value));
});

void loadRuns();
