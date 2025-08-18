#!/usr/bin/env python3
import argparse
import csv
import sys
from typing import Dict, Any, List

import numpy as np
from scipy.spatial.transform import Rotation
from isaaclab.utils.datasets import HDF5DatasetFileHandler


def to_numpy(v):
    try:
        return v.cpu().numpy()
    except Exception:
        return np.asarray(v)


def compute_per_demo_metrics(episode, device: str) -> Dict[str, Any]:
    """
    Compute raw (pre-normalization) per-demo metrics with the SAME logic you use elsewhere:
      - eef_path_length = sum ||Δeef||
      - joint_path_length = sum ||Δq||  (using obs joint_pos, first 7 DoF)
      - orientation_path_length = sum of relative quaternion angles (scipy Rotation, same call pattern)
      - jerk_eef_max, jerk_joint_max from variable-dt derivatives (vel/acc/jerk with step_durations[:-1/-2/-3])
    Returns a dict with raw jerk and other features + the raw step_durations for global scaling.
    """
    obs = episode.data["obs"]

    eef_pos = to_numpy(obs["eef_pos"])             # (T,3)
    eef_quat = to_numpy(obs["eef_quat"])           # (T,4)  (assumed order consistent with your pipeline)
    joint_pos = to_numpy(["actions"])[:, :-1]  # (T,7)  arm only

    T = eef_pos.shape[0]
    try:
        sd = to_numpy(obs["datagen_info"]["step_duration"]).flatten().astype(float)
    except Exception:
        sd = np.array([], dtype=float)

    if sd.size == 0:
        # print("[WARN] step_duration not found; assuming fixed 20 Hz (dt=0.05).")
        sd = np.full((T,), 1.0 / 20.0, dtype=float)
    elif sd.size == T - 1:
        # stored as per-interval; pad one value to make it per-sample length T
        sd = np.concatenate([sd, sd[-1:]], axis=0)
    elif sd.size > T:
        sd = sd[:T]
    elif sd.size < T - 1:
        # too short; pad to T with last or 0.05
        last = sd[-1] if sd.size else 1.0 / 20.0
        sd = np.concatenate([sd, np.full((T - sd.size,), last, dtype=float)], axis=0)

    # sanitize non-finite or non-positive entries
    bad = ~np.isfinite(sd) | (sd <= 0)
    if np.any(bad):
        sd[bad] = 1.0 / 20.0

    step_durations = sd  # length T

    # --- Path lengths (same as your code) ---
    eef_path_length = float(np.sum(np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)))
    joint_path_length = float(np.sum(np.linalg.norm(np.diff(joint_pos, axis=0), axis=1)))

    # --- Orientation path length (same as your code) ---
    r = Rotation.from_quat(eef_quat)  # assuming dataset order matches scipy here
    angular_distances = Rotation.inv(r[:-1]) * r[1:]
    orientation_path_length = float(np.sum(angular_distances.magnitude()))

    # --- Variable-dt derivatives (IDENTICAL indexing to your reference) ---
    # EEF
    vel = np.diff(eef_pos, axis=0) / step_durations[:-1, None]
    acc = np.diff(vel, axis=0) / step_durations[:-2, None]
    jerk_eef = np.diff(acc, axis=0) / step_durations[:-3, None]
    jerk_eef_max = float(np.max(np.linalg.norm(jerk_eef, axis=1))) if jerk_eef.size else 0.0

    # JOINT (use observed joint_pos for jerk; 7 DoF)
    vel_joint = np.diff(joint_pos, axis=0) / step_durations[:-1, None]
    acc_joint = np.diff(vel_joint, axis=0) / step_durations[:-2, None]
    jerk_joint = np.diff(acc_joint, axis=0) / step_durations[:-3, None]
    jerk_joint_max = float(np.max(np.linalg.norm(jerk_joint, axis=1))) if jerk_joint.size else 0.0

    return {
        "eef_path_length": eef_path_length,
        "joint_path_length": joint_path_length,
        "orientation_path_length": orientation_path_length,
        "jerk_eef_max_raw": jerk_eef_max,      # raw; we will scale later
        "jerk_joint_max_raw": jerk_joint_max,  # raw; we will scale later
        "step_durations": step_durations.tolist(),  # for global avg Hz
    }


def main():
    ap = argparse.ArgumentParser("Export per-demo metrics to CSV (jerk normalized to target Hz globally).")
    ap.add_argument("--dataset", required=True, help="Path to HDF5 dataset file")
    ap.add_argument("--device", default="cpu", help="Device string for loader (e.g., cpu/cuda)")
    ap.add_argument("--num-demos", type=int, default=0, help="Limit number of demos (0 = all)")
    ap.add_argument("--out-csv", default="per_demo_metrics.csv", help="Output CSV path")
    ap.add_argument("--target-hz", type=float, default=20.0, help="Target Hz for jerk normalization (global)")
    args = ap.parse_args()

    # Open HDF5
    handler = HDF5DatasetFileHandler()
    handler.open(args.dataset)
    episode_names = list(handler.get_episode_names())
    try:
        episode_names = sorted(episode_names)
    except Exception:
        pass

    if args.num_demos and args.num_demos > 0:
        episode_names = episode_names[: args.num_demos]

    rows_raw: List[Dict[str, Any]] = []
    all_step_dts: List[float] = []

    for ep_name in episode_names:
        try:
            ep = handler.load_episode(ep_name, args.device)
            m = compute_per_demo_metrics(ep, device=args.device)
            rows_raw.append({
                "episode": ep_name,
                "eef_path_length": m["eef_path_length"],
                "joint_path_length": m["joint_path_length"],
                "orientation_path_length": m["orientation_path_length"],
                "jerk_eef_max_raw": m["jerk_eef_max_raw"],
                "jerk_joint_max_raw": m["jerk_joint_max_raw"],
            })
            all_step_dts.extend(m["step_durations"])
        except Exception as e:
            print(f"[WARN] Skipping '{ep_name}': {e}", file=sys.stderr)
            continue

    if not rows_raw:
        print("[ERROR] No episodes processed; no CSV written.", file=sys.stderr)
        sys.exit(1)

    # --- Global 20Hz-style normalization for jerk (same as your pipeline) ---
    avg_dt = float(np.mean(all_step_dts)) if all_step_dts else 0.05
    avg_hz = 1.0 / avg_dt
    scale_j = (args.target_hz / avg_hz) ** 3 if avg_hz > 0 else 1.0

    # Build final rows with scaled jerk
    out_rows = []
    for r in rows_raw:
        out_rows.append({
            "episode": r["episode"],
            "jerk_eef_max": float(r["jerk_eef_max_raw"] * scale_j),         # 20 Hz normalized
            "jerk_joint_max": float(r["jerk_joint_max_raw"] * scale_j),     # 20 Hz normalized
            "eef_path_length": float(r["eef_path_length"]),
            "joint_path_length": float(r["joint_path_length"]),
            "orientation_path_length": float(r["orientation_path_length"]),
        })

    # Write CSV
    fieldnames = [
        "episode",
        "jerk_eef_max",
        "jerk_joint_max",
        "eef_path_length",
        "joint_path_length",
        "orientation_path_length",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"[OK] wrote {args.out_csv}")
    print(f"[INFO] Global avg Hz: {avg_hz:.2f}  target Hz: {args.target_hz:.2f}  jerk scale: {scale_j:.4f}")


if __name__ == "__main__":
    main()
