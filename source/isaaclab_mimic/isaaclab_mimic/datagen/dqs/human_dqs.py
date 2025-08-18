# isaaclab_mimic/datagen/dqs/human_loo_inline.py
import numpy as np
from typing import Dict, Any, List
from isaaclab.utils.datasets import HDF5DatasetFileHandler

from .metrics_path import compute_path_scores
from .metrics_orientation_path_length import compute_orientation_path_length, compute_orientation_score
from .metrics_cartesian_curvature import compute_curvature, compute_curvature_score
from .metrics_jerk import compute_jerk_metrics, score_jerk_against_human
from .metrics_joint_limit import compute_joint_limit_score

def _ensure_wxyz(quat_xyzw: np.ndarray, stored_order: str) -> np.ndarray:
    if stored_order.lower() == "xyzw":
        return np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=1)
    return quat_xyzw

def _load_demos(dataset_path: str, device: str, num_demos: int, stored_quat_order: str):
    h = HDF5DatasetFileHandler(); h.open(dataset_path)
    names = list(h.get_episode_names())
    try:
        names.sort(key=lambda s: int(str(s).split("_")[-1]))
    except Exception:
        pass
    demos = []
    for i, name in enumerate(names):
        if i >= num_demos:
            break
        ep = h.load_episode(name, device)
        obs = ep.data["obs"]
        eef = obs["eef_pos"].cpu().numpy()
        quat = _ensure_wxyz(obs["eef_quat"].cpu().numpy(), stored_quat_order)
        actions_arm = obs["joint_pos"].cpu().numpy()[:, :7] 

        joints = obs["joint_pos"].cpu().numpy()[:, :7]  
        if "datagen_info" in obs and "step_duration" in obs["datagen_info"]:
            dt = float(obs["datagen_info"]["step_duration"].cpu().numpy().flatten().mean())
        else:
            raise RuntimeError 
        demos.append(dict(name=name, eef=eef, quat=quat, joints=joints, joints_act=actions_arm,dt=dt))
    return demos

def _build_human_stats_same_logic(subset):
    eef_paths, joint_paths = [], []
    orient_lengths = []
    curv_means = []
    jerk_eef_max_vals, jerk_joint_max_vals = [], []
    for d in subset:
        eef_paths.append(float(np.linalg.norm(np.diff(d["eef"], axis=0), axis=1).sum()))
        joint_paths.append(float(np.linalg.norm(np.diff(d["joints"], axis=0), axis=1).sum()))
        orient_lengths.append(float(compute_orientation_path_length(d["quat"])))
        curv = compute_curvature(d["eef"], d["dt"])
        curv_means.append(float(curv.mean()) if curv.size else 0.0)
        jm = compute_jerk_metrics(d["eef"], d["joints"], dt=d["dt"])
        jerk_eef_max_vals.append(float(jm["jerk_eef_max"]))
        jerk_joint_max_vals.append(float(jm["jerk_joint_max"]))
    return {
        "jerk": {"jerk_eef_max_values": jerk_eef_max_vals, "jerk_joint_max_values": jerk_joint_max_vals},
        "path": {"eef": eef_paths, "joint": joint_paths},
        "curvature": {"values": curv_means},
        "orientation": {"orientation_path_lengths": orient_lengths},
    }

def _compute_dqs_inline(demo, human_stats, joint_limits):
    metrics = {}

    # 1) jerk
    jm = compute_jerk_metrics(demo["eef"], demo["joints"], dt=demo["dt"])
    metrics["jerk"] = float(score_jerk_against_human(jm, human_stats["jerk"]))

    # 2) path (EEF + joint)
    ps = compute_path_scores(demo["eef"], demo["joints"], human_stats)
    metrics["cartesian_path"] = float(ps["cartesian_path"])
    metrics["joint_path"] = float(ps["joint_path"])

    # 3) joint-limit — ensure 2D input
    qmin, qmax = joint_limits
    q_traj = demo["joints"]                 # (T,7)
    if q_traj.ndim == 1:                    # safety (shouldn't happen with loader)
        q_traj = q_traj[None, :]            # (1,7)

    # your jl funcs likely return per-step values; take mean
    jl_viol_val   = compute_joint_limit_score(q_traj, qmin, qmax)

    metrics["joint_limit"] = float(np.mean(jl_viol_val))

    # 4) curvature
    curv = compute_curvature(demo["eef"], dt=demo["dt"])
    metrics["curvature"] = float(compute_curvature_score(curv, human_stats["curvature"]))

    # 5) orientation (wxyz)
    metrics["orientation"] = float(compute_orientation_score(demo["quat"], human_stats["orientation"]))

    return metrics


def score_humans_loo_with_dqs_inline(dataset_path: str,
                                     device: str,
                                     num_demos: int,
                                     stored_quat_order: str,
                                     q_min_full: np.ndarray,
                                     q_max_full: np.ndarray):
    demos = _load_demos(dataset_path, device, num_demos, stored_quat_order)
    q_min, q_max = q_min_full[:7].astype(float), q_max_full[:7].astype(float)
    joint_limits = (q_min, q_max)

    rows = []
    for i, d in enumerate(demos):
        ref = [x for j, x in enumerate(demos) if j != i]
        human_stats = _build_human_stats_same_logic(ref)
        metrics = _compute_dqs_inline(d, human_stats, joint_limits)
        overall = float(sum(metrics.values()) / len(metrics))
        rows.append({"episode": d["name"], **metrics, "overall": overall})
    return rows

