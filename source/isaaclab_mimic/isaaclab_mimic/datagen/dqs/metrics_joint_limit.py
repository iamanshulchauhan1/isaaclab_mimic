import numpy as np

def compute_joint_limit_score(
    joint_trajectory: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    threshold: float = 0.01,
    print_violating_joints: bool = False,
    *,
    method: str = "gmean",      # "product" | "gmean" | "mean"
    s_floor: float = 1e-3,      # avoids zeros that nuke the aggregate
    gamma: float = 0.7        # <1 softer, >1 harsher; 1.0 = raw
) -> float:
    """
    Joint-limit quality score in [0,1] (higher = better).
    Per joint, per time: s = 4*u*(1-u) in [0,1], where u = (q - q_min)/(q_max - q_min).
    Aggregate across joints per timestep (default: geometric mean), then average over time.
    Captures persistence near limits (hanging near limits -> lower score).

    Args:
        joint_trajectory: (T, J) joint positions over time
        lower_limits: (J,) joint lower bounds
        upper_limits: (J,) joint upper bounds
        threshold: used only for optional reporting of near-limit joints
        print_violating_joints: print indices of joints that ever leave central band
        method: "product" (paper-aligned, harsh), "gmean" (balanced), "mean" (softest)
        s_floor: minimum per-joint score before aggregation (stability)
        gamma: shape factor applied to s (s = s**gamma)

    Returns:
        float in [0,1]
    """
    joint_trajectory = np.asarray(joint_trajectory)[:, :7]
    lower_limits = np.asarray(lower_limits)[:7]
    upper_limits = np.asarray(upper_limits)[:7]

    if joint_trajectory.ndim != 2:
        raise ValueError("joint_trajectory must be (T, J).")
    T, J = joint_trajectory.shape
    if lower_limits.shape != (J,) or upper_limits.shape != (J,):
        raise ValueError(f"lower_limits/upper_limits must be shape ({J},).")

    q_range = upper_limits - lower_limits
    if np.any(q_range <= 0):
        raise ValueError("Each joint must have upper_limits > lower_limits.")

    # Normalize to [0,1]
    u = (joint_trajectory - lower_limits) / q_range
    u = np.clip(u, 0.0, 1.0)

    # Per-joint, per-time score: 1 at mid-range, 0 at limits
    s = 4.0 * u * (1.0 - u)                 # (T, J)
    s = np.clip(s, s_floor, 1.0) ** gamma   # soften/shape

    # Aggregate across joints per timestep
    if method == "product":
        per_t = np.prod(s, axis=1)
    elif method == "gmean":
        per_t = np.exp(np.mean(np.log(s), axis=1))
    elif method == "mean":
        per_t = np.mean(s, axis=1)
    else:
        raise ValueError("method must be 'product', 'gmean', or 'mean'.")

    # Time average = persistence-aware score
    score = float(np.clip(np.mean(per_t), 0.0, 1.0))

    # Optional reporting of joints near limits
    if print_violating_joints:
        band_lo = threshold
        band_hi = 1.0 - threshold
        near_limit = (u < band_lo) | (u > band_hi)
        violating_joints = np.where(np.any(near_limit, axis=0))[0].tolist()
        if violating_joints:
            print(f"[VIOLATION] Joints near limits (ever outside central band): {violating_joints}")

    return score
