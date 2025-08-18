import numpy as np
from typing import Dict, Any
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from scipy.spatial.transform import Rotation


def compute_reference_stats_from_demos(dataset_path: str, device: str, num_demos: int = 10) -> Dict[str, Any]:
    """
    Compute reference statistics from human demos for DQS scoring.
    Includes path lengths, orientation lengths, curvature, jerk, etc.
    Normalizes jerk to match 20 Hz simulation frequency.
    Returns a structured dict with mean and std for Gaussian scoring.
    """
    dataset_handler = HDF5DatasetFileHandler()
    dataset_handler.open(dataset_path)
    episode_names = dataset_handler.get_episode_names()

    raw_stats = {
        "eef_path_lengths": [],
        "joint_path_lengths": [],
        "orientation_path_lengths": [],
        "eef_curvature": [],
        "jerk_eef_max": [],
        "jerk_joint_max": [],
        "step_durations": [],  # for Hz normalization
        "eef_curvature_series": [],
    }

    for i, episode_name in enumerate(episode_names):
        if i >= num_demos:
            break

        episode = dataset_handler.load_episode(episode_name, device)
        
        obs = episode.data["obs"]
        eef_pos = obs["eef_pos"].cpu().numpy()
        eef_quat = obs["eef_quat"].cpu().numpy()
        joint_pos = obs["joint_pos"].cpu().numpy()[:, :7]  # [T, A-1] (exclude gripper)

        try:
            step_durations = obs["datagen_info"]["step_duration"].cpu().numpy().flatten()
            raw_stats["step_durations"].extend(step_durations.tolist())
        except KeyError:
            print(f"[WARNING] step_duration not found in {episode_name}.")
            raise RuntimeError("Cannot compute Hz without step_duration.")

        if len(step_durations) < 4:
            continue

        # Path lengths
        raw_stats["eef_path_lengths"].append(np.sum(np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)))
        raw_stats["joint_path_lengths"].append(np.sum(np.linalg.norm(np.diff(joint_pos, axis=0), axis=1)))

        # Orientation path length
        r = Rotation.from_quat(eef_quat)
        angular_distances = Rotation.inv(r[:-1]) * r[1:]
        raw_stats["orientation_path_lengths"].append(np.sum(angular_distances.magnitude()))

        # Vel/Acc/Jerk using per-step dt
        vel = np.diff(eef_pos, axis=0) / step_durations[:-1, None]
        acc = np.diff(vel, axis=0) / step_durations[:-2, None]
        jerk_eef = np.diff(acc, axis=0) / step_durations[:-3, None]

        if len(vel) >= 2:
            curvature = np.linalg.norm(np.cross(vel[:-1], acc), axis=1) / (np.linalg.norm(vel[:-1], axis=1) ** 3 + 1e-8)
            raw_stats["eef_curvature"].append(np.mean(curvature))
            raw_stats["eef_curvature_series"].append(curvature)

        raw_stats["jerk_eef_max"].append(np.max(np.linalg.norm(jerk_eef, axis=1)))
        vel_joint = np.diff(joint_pos, axis=0) / step_durations[:-1, None]
        acc_joint = np.diff(vel_joint, axis=0) / step_durations[:-2, None]
        jerk_joint = np.diff(acc_joint, axis=0) / step_durations[:-3, None]
        raw_stats["jerk_joint_max"].append(np.max(np.linalg.norm(jerk_joint, axis=1)))

    # === Normalize jerk to 20 Hz ===
    hz_target = 20.0
    avg_dt = np.mean(raw_stats["step_durations"]) if raw_stats["step_durations"] else 0.05
    hz_source = 1.0 / avg_dt
    scale_j = (hz_target / hz_source) ** 3

    raw_stats["jerk_eef_max"] = [x * scale_j for x in raw_stats["jerk_eef_max"]]
    raw_stats["jerk_joint_max"] = [x * scale_j for x in raw_stats["jerk_joint_max"]]        
    
    
        # === Per-demo curvature summary logging ===
    # print("\n[INFO] Per-demo EEF Curvature Stats:")
    # for i, curve in enumerate(raw_stats["eef_curvature_series"]):
    #     print(f"Demo {i}: mean={np.mean(curve):.2f}, std={np.std(curve):.2f}, "
    #           f"min={np.min(curve):.2f}, max={np.max(curve):.2f}")

    # === STRUCTURED OUTPUT FOR DQS ===
    stats = {
        "jerk": {
        # Robust scoring needs full value lists
        "jerk_eef_max_values": raw_stats["jerk_eef_max"],
        "jerk_joint_max_values": raw_stats["jerk_joint_max"],

        # # Optional: still include mean/std for legacy or filtering
        # "jerk_eef_max_mean": float(np.mean(raw_stats["jerk_eef_max"])),
        # "jerk_eef_max_std": float(np.std(raw_stats["jerk_eef_max"])),
        # "jerk_joint_max_mean": float(np.mean(raw_stats["jerk_joint_max"])),
        # "jerk_joint_max_std": float(np.std(raw_stats["jerk_joint_max"])),
    },
        "path": {
            "eef": raw_stats["eef_path_lengths"],
            "joint": raw_stats["joint_path_lengths"],
        },
        "curvature": {
            "mean": float(np.mean(raw_stats["eef_curvature"])) if raw_stats["eef_curvature"] else 0.0,
            "std": float(np.std(raw_stats["eef_curvature"])) if raw_stats["eef_curvature"] else 1e-8,
            # "per_demo": raw_stats["eef_curvature_series"],  # 🔧 per-demo list of arrays
            "values": raw_stats["eef_curvature"]
        },
        "orientation": {
            "orientation_path_lengths": raw_stats["orientation_path_lengths"],
        }
    }

    return stats
