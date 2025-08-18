import numpy as np

def compute_orientation_path_length(quaternions: np.ndarray) -> float:
    """
    Compute the orientation path length from a sequence of quaternions.

    Args:
        quaternions: (T, 4) array of quaternions (w, x, y, z)

    Returns:
        float: total orientation path length in radians
    """
    def quat_angle(q1, q2):
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)
        return 2 * np.arccos(dot)

    path_length = 0.0
    for t in range(1, len(quaternions)):
        path_length += quat_angle(quaternions[t], quaternions[t - 1])
    return path_length

def compute_orientation_path_score(path_length: float, reference_lengths: list[float], percentile: float = 95, steepness: float = 1.0) -> float:
    """
    Score orientation path length using a smooth exponential decay.

    This function provides a score that decreases smoothly as the path length
    exceeds a threshold derived from human reference data.

    Args:
        path_length (float): Orientation path length to score (in radians).
        reference_lengths (list[float]): List of human orientation path lengths.
        percentile (float): The acceptable upper bound percentile from human data.
        steepness (float): Controls how quickly the score decreases. A higher
                           value results in a steeper penalty.

    Returns:
        float: A score in the range (0.0, 1.0].
    """
    if not reference_lengths:
        return 0.0

    threshold = np.percentile(reference_lengths, percentile)
    # print(f'orientation_path_length: {path_length}')
    if path_length <= threshold:
        return 1.0
    else:
        # Calculate the score using an exponential decay based on the
        # relative deviation from the threshold.
        relative_deviation = (path_length - threshold) / threshold
        score = np.exp(-steepness * relative_deviation)
        return float(score)


def compute_orientation_score(eef_quat: np.ndarray, human_stats: dict) -> float:
    path_length = compute_orientation_path_length(eef_quat)
    return compute_orientation_path_score(
        path_length,
        reference_lengths=human_stats.get("orientation_path_lengths", []),
        percentile=95,
        steepness=1.0
    )
