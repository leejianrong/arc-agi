import os

from typing import Dict, Any, List, Tuple
from arc_io import load_arc_folder, plot_task
from baseline import predict_task, equal

def evaluate_folder(
    folder: str, 
    solved_dir: str = None,
    print_transforms: bool = False
) -> Tuple[float, float, int, int]:
    """
    Returns: (grid_acc, task_acc, n_grids, n_tasks)
    """
    tasks = load_arc_folder(folder)
    n_tasks = len(tasks)
    correct_tasks = 0
    correct_grids = 0
    total_grids = 0

    for tid, task in tasks.items():
        preds, transforms = predict_task(task)
        test_pairs = task.get("test", [])
        all_ok = True
        for pred, ex in zip(preds, test_pairs):
            total_grids += 1
            if pred is not None and equal(pred, ex["output"]):
                correct_grids += 1
            else:
                all_ok = False
        if all_ok and len(test_pairs) > 0:
            correct_tasks += 1
            if solved_dir:
                plot_task(task=task, task_id=tid, save_path=f"{solved_dir}/{tid}.png")
            if print_transforms:
                print(f"\nFound solution for task {tid}:")
                print(transforms)

    grid_acc = correct_grids / max(1, total_grids)
    task_acc = correct_tasks / max(1, n_tasks)
    return grid_acc, task_acc, total_grids, n_tasks


def remove_all_files_in_folder(folder_path):
    """
    Removes all files within a specified folder, leaving subdirectories intact.

    Args:
        folder_path (str): The path to the folder from which to remove files.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed file: {filename}")
        else:
            print(f"Skipping directory: {filename}")


if __name__ == "__main__":
    ARC_TRAIN_DIR = "third_party/ARC-AGI/data/training"
    SOLVED_VIZ_DIR = "legacy/solved-viz"
    os.makedirs(SOLVED_VIZ_DIR, exist_ok=True)
    remove_all_files_in_folder(SOLVED_VIZ_DIR)
    grid_acc, task_acc, n_grids, n_tasks = evaluate_folder(
        folder=ARC_TRAIN_DIR, 
        solved_dir=SOLVED_VIZ_DIR,
        print_transforms=True
    )
    print(f"Evaluated {n_tasks} tasks, {n_grids} test grids")
    print(f"Grid accuracy: {grid_acc:.3f}")
    print(f"Task accuracy (all test grids correct): {task_acc:.3f}")