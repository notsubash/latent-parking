"""Train f(o, a) → o' and compare full kinematics to pose-only.

highway-env's bicycle step is position += velocity * dt. Drop vx, vy and two
cars that share (x, y, heading) are indistinguishable, so the same MLP cannot
pick the right next pose. That is the point of this script: a worse plot you
can explain, not a better park.

Score open-loop error on (x, y, cos_h, sin_h) only. Raw MSE of a pose-only
head looks smaller because it never pays for vx, vy, which dominate identity.

One successful CEM episode is a thin slice of planner states. Mix it in so f
has seen a park; judge the mask comparison on held-out random episodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

FEATURE_NAMES = ("x", "y", "vx", "vy", "cos_h", "sin_h")
POSE_IDX = (0, 1, 4, 5)
FULL_IDX = (0, 1, 2, 3, 4, 5)
STEP_KEYS = (
    "observation",
    "achieved_goal",
    "desired_goal",
    "action",
    "next_observation",
    "next_achieved_goal",
    "reward",
    "terminated",
    "truncated",
    "crashed",
    "is_success",
    "episode_id",
)
HORIZONS = (1, 2, 5, 10, 20)


class DynamicsMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path)
    return {key: raw[key] for key in raw.files}


def concat_transitions(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Offset episode_id so random 0 and CEM 0 do not collide."""
    offset = 0
    shifted: list[dict[str, np.ndarray]] = []
    for chunk in chunks:
        data = dict(chunk)
        data["episode_id"] = np.asarray(data["episode_id"], dtype=np.int32) + offset
        offset = int(data["episode_id"].max()) + 1
        shifted.append(data)
    out = {key: np.concatenate([c[key] for c in shifted], axis=0) for key in STEP_KEYS}
    first = chunks[0]
    for key in ("feature_names", "scales", "action_names"):
        if key in first:
            out[key] = first[key]
    return out


def split_episodes(
    data: dict[str, np.ndarray],
    *,
    train_frac: float,
    rng: np.random.Generator,
    force_train: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by episode, not by row. Consecutive (s, s') must not leak across."""
    ids = np.unique(data["episode_id"])
    force = set(int(x) for x in (force_train if force_train is not None else []))
    free = np.array([int(i) for i in ids if int(i) not in force], dtype=np.int32)
    rng.shuffle(free)
    n_train_free = int(round(train_frac * len(free)))
    train_ids = set(free[:n_train_free].tolist()) | force
    train = np.isin(data["episode_id"], list(train_ids))
    return train, ~train


def rows_for(obs: np.ndarray, episode_id: np.ndarray, history: int) -> np.ndarray:
    """history=2 needs the previous pose, so drop the first step of each episode."""
    n = len(obs)
    keep = np.ones(n, dtype=bool)
    if history <= 1:
        return keep
    keep[0] = False
    keep[1:] = episode_id[1:] == episode_id[:-1]
    return keep


def pack_input(prev: np.ndarray | None, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
    # Concatenating two scaled poses is the usual history patch. It is a weak
    # velocity signal here: x is meters/100, so Δx ≈ 0.01 while cos_h is O(1).
    if prev is None:
        return np.concatenate([obs, action], axis=-1)
    return np.concatenate([prev, obs, action], axis=-1)


def pose_of(obs: np.ndarray) -> np.ndarray:
    return obs[..., list(POSE_IDX)]


def identity_pose_mse(obs: np.ndarray, next_obs: np.ndarray) -> float:
    err = pose_of(next_obs) - pose_of(obs)
    return float(np.mean(err ** 2))


def train_model(
    model: DynamicsMLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x)
    last = 0.0
    for _ in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = torch.from_numpy(x[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
            seen += len(idx)
        last = total / max(seen, 1)
    return last


@torch.no_grad()
def one_step_pose_mse(
    model: DynamicsMLP,
    x: np.ndarray,
    next_obs: np.ndarray,
    feat_idx: tuple[int, ...],
    device: torch.device,
) -> float:
    model.eval()
    pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
    pred_pose = pred if feat_idx == POSE_IDX else pose_of(pred)
    true_pose = pose_of(next_obs)
    return float(np.mean((pred_pose - true_pose) ** 2))


def valid_starts(episode_id: np.ndarray, history: int, horizon: int) -> np.ndarray:
    n = len(episode_id)
    starts = []
    first = history - 1
    for t in range(first, n - horizon):
        window = episode_id[t - first : t + horizon]
        if np.all(window == episode_id[t]):
            starts.append(t)
    return np.asarray(starts, dtype=np.int32)


@torch.no_grad()
def open_loop_pose_mse(
    model: DynamicsMLP | None,
    obs: np.ndarray,
    action: np.ndarray,
    episode_id: np.ndarray,
    feat_idx: tuple[int, ...],
    history: int,
    horizons: tuple[int, ...],
    device: torch.device,
    rng: np.random.Generator,
    n_starts: int = 256,
) -> dict[int, float]:
    """Feed predictions back. Identity (model is None) freezes the start pose."""
    max_h = max(horizons)
    starts = valid_starts(episode_id, history, max_h)
    if len(starts) == 0:
        raise RuntimeError("no consecutive val windows; collect longer episodes")
    if len(starts) > n_starts:
        starts = rng.choice(starts, size=n_starts, replace=False)
    if model is not None:
        model.eval()

    acc = {h: [] for h in horizons}
    for t in starts:
        hat = obs[t, list(feat_idx)].copy()
        prev = obs[t - 1, list(feat_idx)].copy() if history > 1 else None
        for step in range(max_h):
            true = obs[t + step + 1]
            if model is None:
                pred = hat
            else:
                action_t = action[t + step]
                packed = pack_input(
                    None if prev is None else prev[None, :],
                    hat[None, :],
                    action_t[None, :],
                )
                pred = model(torch.from_numpy(packed.astype(np.float32)).to(device))
                pred = pred.cpu().numpy()[0]
                if prev is not None:
                    prev = hat
                hat = pred
            pred_pose = pred if feat_idx == POSE_IDX else pose_of(pred)
            h = step + 1
            if h in acc:
                acc[h].append(np.mean((pred_pose - pose_of(true)) ** 2))
    return {h: float(np.mean(acc[h])) for h in horizons}


@torch.no_grad()
def rollout_xy(
    model: DynamicsMLP,
    obs: np.ndarray,
    action: np.ndarray,
    feat_idx: tuple[int, ...],
    history: int,
    scales: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Open-loop x,y in meters from a real action tape (the CEM park)."""
    model.eval()
    xs = [float(obs[0, 0] * scales[0])]
    ys = [float(obs[0, 1] * scales[1])]
    hat = obs[0, list(feat_idx)].copy()
    # No previous frame at t=0. Duplicate pose so the stack sees zero delta.
    prev = hat.copy() if history > 1 else None
    for t in range(len(action)):
        packed = pack_input(
            None if prev is None else prev[None, :],
            hat[None, :],
            action[t][None, :],
        )
        pred = model(torch.from_numpy(packed.astype(np.float32)).to(device))
        pred = pred.cpu().numpy()[0]
        if history > 1:
            prev = hat
        hat = pred
        pose = pred if feat_idx == POSE_IDX else pose_of(pred)
        xs.append(float(pose[0] * scales[0]))
        ys.append(float(pose[1] * scales[1]))
    return np.asarray(xs), np.asarray(ys)


def plot_horizon(curves: dict[str, dict[int, float]], path: Path) -> None:
    colors = {
        "identity": "#a8a29e",
        "full": "#1c1917",
        "pose": "#9f1239",
        "stack": "#14532d",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout="constrained")
    for name, series in curves.items():
        hs = sorted(series)
        ax.plot(
            hs,
            [series[h] for h in hs],
            color=colors.get(name, "#57534e"),
            marker="o",
            lw=1.6,
            label=name,
        )
    ax.set_xlabel("open-loop horizon (steps)")
    ax.set_ylabel("pose MSE  (x, y, cos h, sin h)")
    ax.set_title("Same MLP, different observation")
    ax.legend(loc="upper left", frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_overlay(
    true_xy: tuple[np.ndarray, np.ndarray],
    goal_xy: tuple[float, float],
    imagined: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    # Stack must not share the goal green, or the legend grows a second X.
    colors = {"full": "#1c1917", "pose": "#9f1239", "stack": "#b45309"}
    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    tx, ty = true_xy
    ax.plot(tx, ty, color="#a8a29e", lw=2.0, label="true CEM")
    ax.scatter(tx[0], ty[0], c="#1c1917", s=28, zorder=3, label="start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="#14532d", s=42, marker="x", zorder=3, label="goal")
    for name, (xs, ys) in imagined.items():
        ax.plot(xs, ys, color=colors.get(name, "#57534e"), lw=1.4, ls="--", label=f"imagined {name}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Open-loop on one CEM park (f trained mostly on random)")
    # loc="best" sits on the path in an empty lot. Park labels beside the axes.
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fit_variant(
    name: str,
    feat_idx: tuple[int, ...],
    history: int,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    hidden: int,
    lr: float,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[DynamicsMLP, dict[int, float], float]:
    tr_keep = rows_for(train["observation"], train["episode_id"], history)
    va_keep = rows_for(val["observation"], val["episode_id"], history)
    tr_obs = train["observation"][tr_keep][:, list(feat_idx)]
    va_obs = val["observation"][va_keep][:, list(feat_idx)]
    tr_act = train["action"][tr_keep]
    va_act = val["action"][va_keep]
    tr_next = train["next_observation"][tr_keep]
    va_next = val["next_observation"][va_keep]
    tr_prev = None
    va_prev = None
    if history > 1:
        # rows_for already dropped episode starts, so t-1 is the same episode.
        tr_idx = np.flatnonzero(tr_keep)
        va_idx = np.flatnonzero(va_keep)
        tr_prev = train["observation"][tr_idx - 1][:, list(feat_idx)]
        va_prev = val["observation"][va_idx - 1][:, list(feat_idx)]

    x_tr = pack_input(tr_prev, tr_obs, tr_act).astype(np.float32)
    y_tr = tr_next[:, list(feat_idx)].astype(np.float32)
    x_va = pack_input(va_prev, va_obs, va_act).astype(np.float32)

    model = DynamicsMLP(x_tr.shape[1], y_tr.shape[1], hidden=hidden)
    train_loss = train_model(
        model, x_tr, y_tr, epochs=epochs, batch_size=batch_size, lr=lr, device=device, rng=rng
    )
    step1 = one_step_pose_mse(model, x_va, va_next, feat_idx, device)
    horizon = open_loop_pose_mse(
        model,
        val["observation"],
        val["action"],
        val["episode_id"],
        feat_idx,
        history,
        HORIZONS,
        device,
        rng,
    )
    print(
        f"{name:8s}  train MSE={train_loss:.5f}  "
        f"val pose one-step={step1:.6f}  "
        f"H=1={horizon[1]:.6f}  H=10={horizon[10]:.6f}"
    )
    return model, horizon, step1


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random", type=Path, default=root / "data" / "random_parking.npz")
    parser.add_argument(
        "--cem",
        type=Path,
        nargs="*",
        default=None,
        help="CEM npz files. Default: data/cem_parking.npz if it exists.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-horizon",
        type=Path,
        default=root / "artifacts" / "pose_vs_full_horizon.png",
    )
    parser.add_argument(
        "--out-overlay",
        type=Path,
        default=root / "artifacts" / "cem_openloop_overlay.png",
    )
    parser.add_argument("--ckpt", type=Path, default=root / "checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    cem_paths = args.cem
    if cem_paths is None:
        default_cem = root / "data" / "cem_parking.npz"
        cem_paths = [default_cem] if default_cem.exists() else []

    random_data = load_npz(args.random)
    random_id_hi = int(random_data["episode_id"].max()) + 1
    chunks = [random_data]
    cem_data = None
    for path in cem_paths:
        chunk = load_npz(path)
        chunks.append(chunk)
        if cem_data is None:
            cem_data = chunk
        print(
            f"loaded {path}  transitions={len(chunk['reward'])}  "
            f"successes={int(chunk['is_success'].sum())}"
        )

    data = concat_transitions(chunks)
    # concat offsets CEM ids to sit after the random block.
    cem_ids = np.unique(data["episode_id"][data["episode_id"] >= random_id_hi])
    print(
        f"mix transitions={len(data['reward'])}  "
        f"episodes={int(np.unique(data['episode_id']).size)}  "
        f"cem_episodes={len(cem_ids)}"
    )

    train_mask, val_mask = split_episodes(
        data, train_frac=0.8, rng=rng, force_train=np.asarray(cem_ids, dtype=np.int32)
    )
    train = {k: data[k][train_mask] if k in STEP_KEYS else data[k] for k in data}
    val = {k: data[k][val_mask] if k in STEP_KEYS else data[k] for k in data}
    # episode_id slices are rows; metadata arrays are not in STEP_KEYS.
    for key in ("feature_names", "scales", "action_names"):
        if key in data:
            train[key] = val[key] = data[key]

    ident = identity_pose_mse(val["observation"], val["next_observation"])
    ident_h = open_loop_pose_mse(
        None,
        val["observation"],
        val["action"],
        val["episode_id"],
        FULL_IDX,
        1,
        HORIZONS,
        device,
        rng,
    )
    print(f"identity val pose one-step={ident:.6f}  H=10={ident_h[10]:.6f}")

    full_model, full_h, full_step = fit_variant(
        "full",
        FULL_IDX,
        1,
        train,
        val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        lr=args.lr,
        device=device,
        rng=rng,
    )
    pose_model, pose_h, pose_step = fit_variant(
        "pose",
        POSE_IDX,
        1,
        train,
        val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        lr=args.lr,
        device=device,
        rng=rng,
    )
    stack_model, stack_h, stack_step = fit_variant(
        "stack",
        POSE_IDX,
        2,
        train,
        val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        lr=args.lr,
        device=device,
        rng=rng,
    )

    plot_horizon(
        {"identity": ident_h, "full": full_h, "pose": pose_h, "stack": stack_h},
        args.out_horizon,
    )
    print("saved", args.out_horizon)

    if cem_data is not None:
        scales = np.asarray(cem_data["scales"], dtype=np.float64)
        true_x = cem_data["observation"][:, 0] * scales[0]
        true_y = cem_data["observation"][:, 1] * scales[1]
        last = cem_data["next_observation"][-1]
        true_x = np.append(true_x, last[0] * scales[0])
        true_y = np.append(true_y, last[1] * scales[1])
        goal = cem_data["desired_goal"][0]
        imagined = {
            "full": rollout_xy(
                full_model, cem_data["observation"], cem_data["action"], FULL_IDX, 1, scales, device
            ),
            "pose": rollout_xy(
                pose_model, cem_data["observation"], cem_data["action"], POSE_IDX, 1, scales, device
            ),
            "stack": rollout_xy(
                stack_model, cem_data["observation"], cem_data["action"], POSE_IDX, 2, scales, device
            ),
        }
        plot_overlay(
            (true_x, true_y),
            (float(goal[0] * scales[0]), float(goal[1] * scales[1])),
            imagined,
            args.out_overlay,
        )
        print("saved", args.out_overlay)

    args.ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(full_model.state_dict(), args.ckpt / "dynamics_full.pt")
    torch.save(pose_model.state_dict(), args.ckpt / "dynamics_pose.pt")
    torch.save(stack_model.state_dict(), args.ckpt / "dynamics_stack.pt")

    if full_step >= ident:
        raise SystemExit(
            f"full one-step pose MSE {full_step:.6f} did not beat identity {ident:.6f}"
        )
    if pose_h[10] <= full_h[10]:
        print(
            "note: pose H=10 was not worse than full. "
            "Rerun or look at the overlay; the mask story may need more speed variation."
        )
    else:
        print(
            f"pose H=10 {pose_h[10]:.6f} > full H=10 {full_h[10]:.6f}  "
            "(worse, as expected without velocity)"
        )
    print(
        f"stack one-step={stack_step:.6f}  "
        "naive two-pose history hugs pose-only; stored x buries the delta."
    )


if __name__ == "__main__":
    main()
