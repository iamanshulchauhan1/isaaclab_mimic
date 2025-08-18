import numpy as np

def compute_curvature(eef_pos: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute curvature from end-effector positions.

    Args:
        eef_pos: [T, 3] array of end-effector positions.
        dt: Time step between positions.

    Returns:
        curvature: [T-3] array of scalar curvature values.
    """
    vel = np.diff(eef_pos, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    vel = vel[:-1]  # Align dimensions

    cross = np.cross(vel, acc)
    num = np.linalg.norm(cross, axis=1)
    denom = np.linalg.norm(vel, axis=1) ** 3 + 1e-8
    curvature = num / denom
    return curvature

def compute_curvature_score(curvatures: np.ndarray, ref_stats: dict) -> float:
    """
    Score curvature similarity using percentile-based band from human demos.

    Args:
        curvatures: [T] array of curvature values for the current demo
        ref_stats: dict with 'values' key containing list of human curvature means

    Returns:
        score: float in [0, 1]
    """
    if curvatures.size == 0 or ref_stats is None or "values" not in ref_stats:
        return 0.0
    curvature_mean = float(np.mean(curvatures))
    ref_values = np.array(ref_stats["values"])
    # print(f"Curvature mean: {curvature_mean},")
    p10 = np.percentile(ref_values, 2)
    p90 = np.percentile(ref_values, 90)


    if p10 <= curvature_mean <= p90:
        return 1.0
    elif curvature_mean < p10:
        return max(0.0, curvature_mean / p10)
    else:  # curvature_mean > p90
        return max(0.0, 2 - (curvature_mean / p90))


