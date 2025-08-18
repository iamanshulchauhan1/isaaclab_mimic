
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_jerk import compute_jerk_metrics, score_jerk_against_human
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_joint_limit import compute_joint_limit_score
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_path import compute_path_scores
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_orientation_path_length import compute_orientation_score
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_cartesian_curvature import compute_curvature, compute_curvature_score
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_manipulability import compute_manipulability_score
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.dqs_stats import compute_reference_stats_from_demos
class DQSEvaluator:
    def __init__(self, human_stats):
        self.human_stats = human_stats
        self.metrics = {}

    def compute(self, eef_traj, joint_traj, joint_pos, eef_quat, jacobians, joint_limits, dt=1/20):
            """
            Compute all DQS metrics comparing against human reference.

            Args:
                eef_traj (np.ndarray): End-effector trajectory [T, 3]
                joint_traj (np.ndarray): Joint position trajectory [T, DoF]
                joint_pos (np.ndarray): Joint position [DoF]
                eef_quat (np.ndarray): End-effector quaternion trajectory [T, 4]
                jacobians (np.ndarray): Jacobians [T, 6, DoF]
                joint_limits (tuple): (lower_limits [DoF], upper_limits [DoF])
            """
            jerk_metrics = compute_jerk_metrics(eef_traj, joint_pos, dt=dt)
            self.metrics["jerk"] = score_jerk_against_human(jerk_metrics, self.human_stats["jerk"])

  
            self.metrics.update(compute_path_scores(eef_traj, joint_traj, self.human_stats))


            # self.metrics["manipulability"] = compute_manipulability_score(jacobians)

            lower_limits, upper_limits = joint_limits
            limits_list = list(zip(lower_limits, upper_limits))


            self.metrics["joint_limit_violation"] = compute_joint_limit_score(
                joint_pos, lower_limits, upper_limits
            )


            curvatures = compute_curvature(eef_traj, dt=1/20)
            self.metrics["curvature"] = compute_curvature_score(curvatures, self.human_stats["curvature"])
             
            self.metrics["orientation"] = compute_orientation_score(
                eef_quat, self.human_stats["orientation"]
            )

            return self.metrics

    def overall_score(self) -> float:
        """Return the average of all metric scores."""
        return sum(self.metrics.values()) / len(self.metrics)

    def as_dict(self) -> dict:
        return self.metrics