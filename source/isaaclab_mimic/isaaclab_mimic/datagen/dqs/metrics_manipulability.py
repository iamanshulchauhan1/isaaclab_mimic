import numpy as np

def compute_manipulability_score(joint_trajectory: np.ndarray, get_jacobian_fn) -> float:
    """
    Compute average manipulability score over a demo.

    Args:
        joint_trajectory: (T, J) array of joint positions over time
        get_jacobian_fn: function that takes in joint pos (J,) and returns Jacobian (6, J)

    Returns:
        manipulability_score: float (higher is better)
    """
    manipulabilities = []

    for q in joint_trajectory:
        J = get_jacobian_fn(q)
        if J is None or J.shape[0] != 6:
            manipulabilities.append(0.0)
            continue

        try:
            JJt = J @ J.T
            score = np.sqrt(np.linalg.det(JJt))
            manipulabilities.append(score)
        except np.linalg.LinAlgError:
            manipulabilities.append(0.0)

    return np.mean(manipulabilities)