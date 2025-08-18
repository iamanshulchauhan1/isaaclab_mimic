import numpy as np

def compute_jerk_norms(sequence: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute the jerk (3rd derivative) magnitude over a trajectory.
    Args:
        sequence: [T, D] array (EEF pos or joint angles)
        dt: timestep (1 / Hz)
    Returns:
        jerk_magnitudes: [T - 3] array of jerk norms
    """
    velocity = np.diff(sequence, axis=0) / dt            # [T-1, D]
    acceleration = np.diff(velocity, axis=0) / dt        # [T-2, D]
    jerk = np.diff(acceleration, axis=0) / dt            # [T-3, D]
    return np.linalg.norm(jerk, axis=1)                  # [T-3]

def compute_jerk_metrics(eef_traj: np.ndarray, joint_traj: np.ndarray, dt: float) -> dict:
    """
    Compute jerk metrics for both Cartesian and joint-space.
    Args:
        eef_traj: [T, 3] end-effector positions
        joint_traj: [T, J] joint angle positions
        dt: timestep in seconds
    Returns:
        Dictionary with max jerk and sum jerk in each space
    """
    jerk_eef = compute_jerk_norms(eef_traj, dt)
    jerk_joint = compute_jerk_norms(joint_traj, dt)
    jerk_metrics = {
        "jerk_eef_max": float(np.max(jerk_eef)) if len(jerk_eef) > 0 else 0.0,
        "jerk_joint_max": float(np.max(jerk_joint)) if len(jerk_joint) > 0 else 0.0,
        "jerk_eef_sum": float(np.sum(jerk_eef)) if len(jerk_eef) > 0 else 0.0,
        "jerk_joint_sum": float(np.sum(jerk_joint)) if len(jerk_joint) > 0 else 0.0,
    }
    return jerk_metrics
def score_jerk_against_human(jerk_metrics: dict, human_stats: dict,
                             low_weight: float = 0.1,
                             high_weight: float = 1.0,
                             gamma: float = 1.5) -> float:
    """
    Asymmetric jerk scoring with a hard cutoff:
      - If demo jerk is >= 95th percentile of human refs -> score = 0
      - Otherwise:
          * lower-than-median jerk: small penalty
          * higher-than-median jerk: stronger penalty
    """

    def asymmetric_percentile_score_with_cap(x, reference_values) -> float:
        ref = np.asarray(reference_values, dtype=float)
        if ref.size == 0 or not np.isfinite(x):
            return 0.0
        ref_sorted = np.sort(ref)
        p = np.searchsorted(ref_sorted, x, side="right") / ref_sorted.size  # percentile in [0,1]

        # Hard cap: too jerky vs humans -> zero
        if p >= 0.95:
            return 0.0

        # Otherwise asymmetric around the median
        d = abs(p - 0.5) / 0.5                 # distance from median in [0,1]
        weight = low_weight if p <= 0.5 else high_weight
        score = 1.0 - weight * (d ** gamma)
        return float(np.clip(score, 0.0, 1.0))

    scores = []
    for k, demo_val in jerk_metrics.items():
        ref_key = k + "_values"
        if ref_key in human_stats:
            ref_vals = human_stats[ref_key]
            scores.append(asymmetric_percentile_score_with_cap(demo_val, ref_vals))

    return float(np.mean(scores)) if scores else 0.0