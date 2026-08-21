"""One-step identity baseline: predict s' = s, ignore the action.

Phase 2 pass #1 is "one-step prediction beats predict no change." This script
is that bar, measured on data/random_parking.npz, before any network exists.

MSE is in the stored (scaled) coordinates the MLP will train on. x and y look
tiny because they are meters/100. Compare f against this script in the same
space, not in meters, or the numbers will lie.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

FEATURE_NAMES = ("x", "y", "vx", "vy", "cos_h", "sin_h")


def identity_mse(s: np.ndarray, s_next: np.ndarray) -> tuple[float, np.ndarray]:
    err2 = (s_next - s) ** 2
    per_feature = err2.mean(axis=0)
    return float(err2.mean()), per_feature


def plot_bars(per_feature: np.ndarray, overall: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(FEATURE_NAMES, per_feature, color="#9f1239")
    ax.set_ylabel("MSE of s' = s")
    ax.set_title(f"Identity baseline  overall MSE={overall:.5f}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "random_parking.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "identity_baseline.png",
    )
    args = parser.parse_args()

    data = np.load(args.data)
    s = data["observation"]
    s_next = data["next_observation"]
    n = len(s)
    n_success = int(data["is_success"].sum())
    ended = data["terminated"] | data["truncated"]
    n_ended = int(ended.sum())
    n_success_ended = int(data["is_success"][ended].sum()) if n_ended else 0

    overall, per_feature = identity_mse(s, s_next)
    print(f"transitions={n}  completed_episodes={n_ended}")
    print(f"success_flags={n_success}  success_among_ended={n_success_ended}/{n_ended}")
    print(f"identity overall MSE={overall:.6f}")
    for name, value in zip(FEATURE_NAMES, per_feature):
        print(f"  {name:6s} {value:.6e}")
    plot_bars(per_feature, overall, args.out)
    print("saved", args.out)
    print(
        "A later MLP must beat this overall MSE on held-out one-step preds. "
        "It still will not park if the train set has zero successful parks."
    )


if __name__ == "__main__":
    main()
