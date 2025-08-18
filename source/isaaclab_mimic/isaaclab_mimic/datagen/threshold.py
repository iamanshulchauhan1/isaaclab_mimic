import numpy as np
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from typing import Dict, Any

def auto_compute_thresholds_from_demos(dataset_path, device, num_demos: int = 10):
    """
    Compute EEF and joint space velocity, acceleration, and jerk thresholds
    from high-quality human demos, using step_duration if available.
    """
    print("[INFO] Computing thresholds from source demos...")

    demo_thresholds = {
        "eef_max_jerk": [],
        "eef_max_velocity": [],
        "eef_max_acceleration": [],
        "joint_max_jerk": [],
        "joint_max_velocity": [],
        "joint_max_acceleration": [],
    }

    dataset_handler = HDF5DatasetFileHandler()
    dataset_handler.open(dataset_path)
    episode_names = dataset_handler.get_episode_names()

    all_step_durations = []

    for i, episode_name in enumerate(episode_names):
        if i >= num_demos:
            break

        episode = dataset_handler.load_episode(episode_name, device)
        obs = episode.data["obs"]
        eef_pos = obs["eef_pos"].cpu().numpy()                     # [T, 3]
        actions = obs["joint_pos"].cpu().numpy()[:, :7] # [T, A-1] (exclude gripper)

        try:
            step_durations = obs["datagen_info"]["step_duration"].cpu().numpy().flatten()  # [T]
            all_step_durations.extend(step_durations.tolist())
            
        except KeyError:
            print(f"[WARNING] step_duration not found in {episode_name}. Skipping.")
            continue

        if len(step_durations) < 4:
            continue

        # EEF dynamics
        vel_eef = np.diff(eef_pos, axis=0) / step_durations[:-1, None]
        acc_eef = np.diff(vel_eef, axis=0) / step_durations[:-2, None]
        jerk_eef = np.diff(acc_eef, axis=0) / step_durations[:-3, None]

        demo_thresholds["eef_max_velocity"].append(np.max(np.linalg.norm(vel_eef, axis=1)))
        demo_thresholds["eef_max_acceleration"].append(np.max(np.linalg.norm(acc_eef, axis=1)))
        demo_thresholds["eef_max_jerk"].append(np.max(np.linalg.norm(jerk_eef, axis=1)))

        # Joint dynamics
        vel_joint = np.diff(actions, axis=0) / step_durations[:-1, None]
        acc_joint = np.diff(vel_joint, axis=0) / step_durations[:-2, None]
        jerk_joint = np.diff(acc_joint, axis=0) / step_durations[:-3, None]

        demo_thresholds["joint_max_velocity"].append(np.max(np.linalg.norm(vel_joint, axis=1)))
        demo_thresholds["joint_max_acceleration"].append(np.max(np.linalg.norm(acc_joint, axis=1)))
        demo_thresholds["joint_max_jerk"].append(np.max(np.linalg.norm(jerk_joint, axis=1)))

    if not all_step_durations:
        raise RuntimeError("No valid step durations found in the selected demos.")

    avg_dt = np.mean(all_step_durations)
    hz_human = 1.0 / avg_dt
    hz_robot = 20.0

    scale_velocity = hz_robot / hz_human
    scale_acceleration = scale_velocity ** 2
    scale_jerk = scale_velocity ** 3

    thresholds_raw = {
        k: float(np.percentile(v, 95)) for k, v in demo_thresholds.items()
    }

    thresholds = {
        k: thresholds_raw[k] * (
            scale_velocity if "velocity" in k else scale_acceleration if "acceleration" in k else scale_jerk
        )
        for k in thresholds_raw
    }

    print(f"[INFO] Estimated human Hz: {hz_human:.2f}")
    print(f"[INFO] Raw thresholds: {thresholds_raw}")
    print(f"[INFO] Scaled thresholds for {hz_robot} Hz: {thresholds}")

    return thresholds

