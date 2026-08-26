# arc_visualize_matplotlib.py
# ----------------------------
# Show ARC task samples using matplotlib with grids and labels

import json, os, glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

# Fixed 10-color palette for ARC
PALETTE = np.array([
    [0, 0, 0],        # 0 - black
    [0, 116, 217],    # 1 - blue
    [255, 65, 54],    # 2 - red
    [46, 204, 64],    # 3 - green
    [255, 220, 0],    # 4 - yellow
    [170, 170, 170],    # 5 - grey
    [240, 18, 190],   # 6 - magenta
    [255, 133, 27],   # 7 - orange
    [127, 219, 255],  # 8 - light blue
    [135, 12, 37],    # 9 - brown
], dtype=np.uint8)


def load_arc_folder(folder: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict: task_id -> {"train": [...], "test": [...]}
    where each list item is {"input": [[...]], "output": [[...]]}
    """
    tasks = {}
    for f in glob.glob(os.path.join(folder, "*.json")):
        with open(f, "r") as fp:
            obj = json.load(fp)
        tasks[Path(f).stem] = obj
    return tasks


def load_arc_task(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    

def grid_to_rgb(grid):
    """Convert integer grid to RGB numpy array"""
    arr = np.array(grid, dtype=int)
    return PALETTE[arr]


def plot_task(task: dict, task_id: str, save_path=None):
    """
    Plot all train and test input-output pairs in a single matplotlib figure.
    """
    n_train = len(task["train"])
    n_test = len(task["test"])
    total = n_train + n_test

    # Each example has input + output, so total * 2 subplots
    ncols = 2
    nrows = total
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 3*total))

    if nrows == 1:  # ensure axes is always 2D
        axes = np.array([axes])

    for idx, ex in enumerate(task["train"] + task["test"]):
        # input
        ax_in = axes[idx, 0]
        img_in = grid_to_rgb(ex["input"])
        ax_in.imshow(img_in, interpolation="nearest")
        ax_in.set_title(f"{'train' if idx < n_train else 'test'}_{idx if idx < n_train else idx-n_train}_input")
        draw_grid(ax_in, ex["input"])
        
        # output
        ax_out = axes[idx, 1]
        if "output" in ex:
            img_out = grid_to_rgb(ex["output"])
            ax_out.imshow(img_out, interpolation="nearest")
            ax_out.set_title(f"{'train' if idx < n_train else 'test'}_{idx if idx < n_train else idx-n_train}_output")
            draw_grid(ax_out, ex["output"])
        else:
            ax_out.set_visible(False)

    plt.suptitle(f"Task: {task_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def draw_grid(ax, grid):
    """Add gridlines matching the cells of the ARC grid"""
    h, w = len(grid), len(grid[0])
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


if __name__ == "__main__":
    # Example usage
    ARC_TRAIN_DIR = "ARC-AGI/data/training"
    OUT_DIR = "visualizations"
    files = glob.glob(os.path.join(ARC_TRAIN_DIR, "*.json"))
    # Just plot one task for demo
    print(f"Visualizing {len(files)} tasks...")
    for task_file in files:
        task_id = Path(task_file).stem
        print(f"Working on task {task_id}")
        task = load_arc_task(task_file)
        plot_task(task, task_id, save_path=f"visualizations/{task_id}.png")
    print(f"Done.")
