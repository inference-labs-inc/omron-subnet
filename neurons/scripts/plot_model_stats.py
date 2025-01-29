import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_scores(scores_path: Path) -> Union[List, Dict]:
    if not scores_path.exists():
        return {}
    try:
        scores = torch.load(scores_path)
        if isinstance(scores, torch.Tensor):
            return scores.tolist()
        elif isinstance(scores, dict):
            return {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in scores.items()
            }
        return scores
    except Exception as e:
        print(f"Error loading scores from {scores_path}: {str(e)}")
        return {}


def process_evaluation_data(
    eval_file: Path, model_id: str, scores: Union[List, Dict]
) -> Optional[Dict]:
    if not eval_file.exists():
        return None

    try:
        with open(eval_file, "r") as f:
            data = json.load(f)
            if not data:
                return None

            uid_latest = {}
            seen_uids = set()

            for item in data:
                if item["verification_result"]:
                    uid = item["uid"]
                    if uid in seen_uids:
                        print(
                            f"\nWarning: Found duplicate UID {uid} in model {model_id}, keeping most recent entry"
                        )
                    seen_uids.add(uid)

                    try:
                        score = (
                            scores[int(uid)]
                            if isinstance(scores, list)
                            else scores.get(uid, {}).get(model_id, 0.0)
                        )
                    except (IndexError, ValueError):
                        score = 0.0

                    uid_latest[uid] = (uid, item["response_time"], score)

            if not uid_latest:
                return None

            response_times = list(uid_latest.values())
            uids, times, scores = zip(*response_times)
            return {
                "uids": uids,
                "times": times,
                "scores": scores,
                "min_time": min(times),
                "max_time": max(times),
            }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing evaluation data for model {model_id}: {str(e)}")
        return None


def load_evaluation_data(models_path: Path, scores_path: Path) -> Dict:
    scores = load_scores(scores_path)
    model_stats = {}

    for model_dir in models_path.glob("model_*"):
        model_id = model_dir.name.replace("model_", "")
        eval_file = model_dir / "evaluation_data.json"

        stats = process_evaluation_data(eval_file, model_id, scores)
        if stats:
            model_stats[model_id] = stats

    return dict(
        sorted(model_stats.items(), key=lambda x: (x[1]["min_time"], x[1]["max_time"]))
    )


def create_scatter_plot(ax: plt.Axes, data: Dict, title: str) -> None:

    colors = [
        (0.8, 0.1, 0.1),
        (0.95, 0.9, 0.25),
        (0.1, 0.8, 0.1),
    ]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    scores = np.array(data["scores"])
    if len(scores) > 0:
        score_range = scores.max() - scores.min()
        if score_range > 0:
            normalized_scores = (scores - scores.min()) / score_range
        else:
            normalized_scores = np.zeros_like(scores)
    else:
        normalized_scores = np.array([])

    scatter = ax.scatter(
        data["uids"],
        data["times"],
        c=normalized_scores,
        cmap=cmap,
        alpha=0.9,
        s=100,
        edgecolor="white",
        linewidth=0.5,
    )
    cbar = plt.colorbar(scatter, label="Score")
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_facecolor("#f8f9fa")

    ax.set_title(title, fontsize=14, pad=15, weight="bold")
    ax.set_ylabel("Response Time (s)", fontsize=12, weight="bold")
    ax.grid(True, alpha=0.2, linestyle="--", color="#cccccc")
    ax.tick_params(axis="both", labelsize=10)
    ax.set_facecolor("#f8f9fa")

    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")
        spine.set_linewidth(1.5)


def calculate_average_times(model_stats: Dict) -> Dict:
    uid_times = defaultdict(list)
    uid_scores = defaultdict(list)

    for stats in model_stats.values():
        for uid, time, score in zip(stats["uids"], stats["times"], stats["scores"]):
            uid_times[uid].append(time)
            uid_scores[uid].append(score)

    return {
        "uids": sorted(uid_times.keys()),
        "times": [np.mean(uid_times[uid]) for uid in sorted(uid_times.keys())],
        "scores": [np.mean(uid_scores[uid]) for uid in sorted(uid_times.keys())],
    }


def plot_stats(models_path: Path, scores_path: Path) -> None:
    print(f"\nLoading model stats from {models_path}")
    model_stats = load_evaluation_data(models_path, scores_path)

    if not model_stats:
        print("No evaluation data found")
        return

    print(f"\nProcessing {len(model_stats)} models...")
    n_models = len(model_stats)

    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#f8f9fa"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.2
    plt.rcParams["grid.color"] = "#cccccc"

    fig = plt.figure(figsize=(20, 5 * (n_models + 1)))
    fig.patch.set_facecolor("#ffffff")

    for idx, (model_id, stats) in enumerate(model_stats.items(), 1):
        print(f"Plotting model {model_id} ({idx}/{n_models})")
        ax = plt.subplot(n_models + 1, 1, idx)
        create_scatter_plot(ax, stats, f"Model {model_id}")
        if idx != n_models:
            ax.set_xticklabels([])

    print("Calculating average times...")
    avg_stats = calculate_average_times(model_stats)
    ax = plt.subplot(n_models + 1, 1, n_models + 1)
    create_scatter_plot(ax, avg_stats, "Average Response Time Across All Models")
    ax.set_xlabel("UID", fontsize=12, weight="bold")

    print("Saving plot...")
    plt.tight_layout(pad=3.0)
    plt.savefig("model_stats.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Done! Plot saved as model_stats.png")


if __name__ == "__main__":
    models_path = Path(os.path.expanduser("~/.bittensor/omron/models/"))
    scores_path = Path(os.path.expanduser("~/.bittensor/omron/scores.pt"))

    if not models_path.exists():
        print(f"Models directory not found: {models_path}")
        sys.exit(1)

    if not scores_path.exists():
        print(f"Scores file not found: {scores_path}")
        print("Will proceed with empty scores...")

    plot_stats(models_path, scores_path)
