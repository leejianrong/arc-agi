import type { Episode } from "./api";
import type { Grid } from "./grid";

export interface Scheduler {
  setInterval(fn: () => void, ms: number): number;
  clearInterval(id: number): void;
}

const defaultScheduler: Scheduler = {
  setInterval: (fn, ms) => window.setInterval(fn, ms),
  clearInterval: (id) => window.clearInterval(id),
};

// Frame 0 is the episode's starting grid; frame i (i >= 1) is the grid
// after step i - 1. Drives play/pause/step/speed controls (ADR-0007) over
// one already-loaded episode.
export class EpisodePlayer {
  private episode: Episode;
  private index = 0;
  private intervalId: number | null = null;
  private speedMs = 500;
  private scheduler: Scheduler;

  constructor(episode: Episode, scheduler: Scheduler = defaultScheduler) {
    this.episode = episode;
    this.scheduler = scheduler;
  }

  get frameCount(): number {
    return this.episode.steps.length + 1;
  }

  get currentIndex(): number {
    return this.index;
  }

  get currentGrid(): Grid {
    return this.index === 0 ? this.episode.start.grid : this.episode.steps[this.index - 1].grid_after;
  }

  get currentStep(): typeof this.episode.steps[number] | null {
    return this.index === 0 ? null : this.episode.steps[this.index - 1];
  }

  get isAtEnd(): boolean {
    return this.index >= this.frameCount - 1;
  }

  get isAtStart(): boolean {
    return this.index === 0;
  }

  get isPlaying(): boolean {
    return this.intervalId !== null;
  }

  stepForward(): boolean {
    if (this.isAtEnd) return false;
    this.index += 1;
    return true;
  }

  stepBackward(): boolean {
    if (this.isAtStart) return false;
    this.index -= 1;
    return true;
  }

  reset(): void {
    this.pause();
    this.index = 0;
  }

  setSpeedMs(ms: number): void {
    this.speedMs = Math.max(16, ms);
    if (this.isPlaying) {
      this.pause();
      this.play(this.onTickCallback!);
    }
  }

  private onTickCallback: (() => void) | null = null;

  play(onTick: () => void): void {
    if (this.isPlaying) return;
    this.onTickCallback = onTick;
    this.intervalId = this.scheduler.setInterval(() => {
      if (!this.stepForward()) {
        this.pause();
      }
      onTick();
    }, this.speedMs);
  }

  pause(): void {
    if (this.intervalId !== null) {
      this.scheduler.clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }
}
