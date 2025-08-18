import numpy as np


def compute_cartesian_path_length(eef_pos: np.ndarray) -> float:
    """
    Compute total Cartesian path length from a trajectory.

    Args:
        eef_pos (np.ndarray): End-effector positions [T, 3]

    Returns:
        float: Total path length
    """
    deltas = np.diff(eef_pos, axis=0)  # [T-1, 3]
    segment_lengths = np.linalg.norm(deltas, axis=1)
    total_path_length = np.sum(segment_lengths)
    # print(f"Total Cartesian path length: {total_path_length}")
    return total_path_length 


def compute_joint_path_length(joint_pos: np.ndarray) -> float:
    """
    Compute total joint-space path length from a trajectory.

    Args:
        joint_pos (np.ndarray): Joint positions [T, J]

    Returns:
        float: Total joint path length
    """
    deltas = np.diff(joint_pos, axis=0)  # [T-1, J]
    segment_lengths = np.linalg.norm(deltas, axis=1)
    total_path_length = np.sum(segment_lengths)
    # print(f"Total joint path length: {total_path_length}")
    return total_path_length


def compute_path_score(value: float, human_values: list[float], percentile: float = 95, steepness: float = 0.5) -> float:
    """
    Compute a score using a smooth exponential decay based on deviation from the human percentile.

    Args:
        value (float): The path length to score.
        human_values (list of float): List of human path lengths.
        percentile (float): The acceptable upper bound percentile from human data.
        steepness (float): Controls how quickly the score decreases. Higher is steeper.

    Returns:
        float: Score in (0.0, 1.0]
    """
    if not human_values:
        return 0.0

    threshold = np.percentile(human_values, percentile)

    if value <= threshold:
        return 1.0
    else:
        # The score decreases exponentially based on how much the value exceeds the threshold.
        relative_deviation = (value - threshold) / threshold
        score = np.exp(-steepness * relative_deviation)
        return float(score)


def compute_path_scores(eef_pos: np.ndarray, joint_pos: np.ndarray, human_stats: dict) -> dict:
    """
    Compute path length metrics and scores for Cartesian and joint-space trajectories.

    Args:
        eef_pos (np.ndarray): End-effector positions [T, 3]
        joint_pos (np.ndarray): Joint positions [T, J]
        human_stats (dict): Dict with keys 'eef_path_lengths', 'joint_path_lengths'

    Returns:
        dict: Scores for cartesian_path and joint_path
    """
    eef_path = compute_cartesian_path_length(eef_pos)
    joint_path = compute_joint_path_length(joint_pos)
    score_eef = compute_path_score(eef_path, human_stats.get("path", {}).get("eef", []))
    score_joint = compute_path_score(
        joint_path,
        human_stats.get("path", {}).get("joint", []),
        percentile=95,
        steepness=1.0
)
    
    return {
        "cartesian_path": score_eef,
        "joint_path": score_joint,
    }
