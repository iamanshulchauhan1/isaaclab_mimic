# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained robomimic policy across multiple settings, with optional DQS scoring
   for every successful rollout.

Args (added for DQS):
    --dqs                 : compute DQS metrics for each successful rollout
    --human_h5            : path to human demos HDF5 (for reference stats)
    --num_hum             : number of human demos to use for reference stats
    --dqs_csv             : optional CSV path to append per-success DQS rows
"""

# -------------------- Launch Isaac Sim --------------------
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric / use USD I/O.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--input_dir", type=str, default=None, help="Directory containing models to evaluate.")
parser.add_argument("--start_epoch", type=int, default=100, help="Start evaluating from this checkpoint epoch.")
parser.add_argument("--horizon", type=int, default=600, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=15, help="Number of rollouts for each setting.")
parser.add_argument("--num_seeds", type=int, default=3, help="Number of random seeds to evaluate.")
parser.add_argument("--seeds", nargs="+", type=int, default=None, help="List of specific seeds to use.")
parser.add_argument("--log_dir", type=str, default="/tmp/policy_evaluation_results", help="Results directory.")
parser.add_argument("--log_file", type=str, default="results", help="Name of output file.")
parser.add_argument("--output_vis_file", type=str, default="visuals.hdf5", help="Recorded episodes file.")
parser.add_argument("--norm_factor_min", type=float, default=None, help="Action normalization min.")
parser.add_argument("--norm_factor_max", type=float, default=None, help="Action normalization max.")
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")

# ---- DQS flags ----
parser.add_argument("--dqs", action="store_true", help="Compute DQS metrics for each successful rollout.")
parser.add_argument("--human_h5", type=str, default=None, help="Human demos HDF5 for reference stats.")
parser.add_argument("--num_hum", type=int, default=10, help="Number of human demos for reference.")
parser.add_argument("--dqs_csv", type=str, default="", help="Optional CSV path to append per-success DQS rows.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------- Rest of script --------------------
import copy
import gymnasium as gym
import os
import pathlib
import random
import torch
import numpy as np

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from isaaclab_tasks.utils import parse_env_cfg

# ---- DQS imports (your existing modules) ----
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_jerk import (
    compute_jerk_metrics, score_jerk_against_human
)
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_path import compute_path_scores
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_orientation_path_length import compute_orientation_score
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_cartesian_curvature import (
    compute_curvature, compute_curvature_score
)
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.dqs_stats import compute_reference_stats_from_demos
# Use YOUR joint-limit implementation exactly as provided
from source.isaaclab_mimic.isaaclab_mimic.datagen.dqs.metrics_joint_limit import compute_joint_limit_score


# -------------------- Helpers --------------------
def _to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return np.asarray(x)

def extract_rollout_arrays(traj, dt: float):
    """
    Build arrays from recorded obs in `traj`. Uses traj['obs'] list (includes t=0).
    Assumes eef_quat is already in WXYZ order.
    Returns: eef_traj [T,3], eef_quat_wxyz [T,4], joint_traj [T,7], step_durations [T]
    """
    obs_seq = traj["obs"]
    if len(obs_seq) < 2:
        raise ValueError("Not enough samples in trajectory to compute metrics.")

    try:
        eef_pos = np.stack([_to_numpy(o["eef_pos"]) for o in obs_seq], axis=0)          # [T,3]
        eef_quat_wxyz = np.stack([_to_numpy(o["eef_quat"]) for o in obs_seq], axis=0)   # [T,4] WXYZ
    except KeyError as e:
        raise KeyError(f"Policy obs missing key {e}. Ensure 'eef_pos' and 'eef_quat' exist in observations.policy.")

    try:
        joint_pos_full = np.stack([_to_numpy(o["joint_pos"]) for o in obs_seq], axis=0)  # [T,J]
    except KeyError:
        raise KeyError("Policy obs missing 'joint_pos'. Add it to observations.policy.")

    if joint_pos_full.shape[1] < 7:
        raise ValueError(f"Expected at least 7 DoF in joint_pos, got {joint_pos_full.shape[1]}")
    joint_traj = joint_pos_full[:, :7]

    # dt vector (constant per-step from env)
    step_durations = np.full((eef_pos.shape[0],), float(dt), dtype=float)  # length T (sample-wise)
    return eef_pos, eef_quat_wxyz, joint_traj, step_durations


def compute_dqs_for_rollout(eef_traj, eef_quat_wxyz, joint_traj, human_stats, joint_limits, dt):
    """
    Inline DQS:
      - jerk vs human
      - path vs human
      - curvature vs human
      - orientation vs human
      - joint-limit score over the full trajectory (your implementation)
    """
    # jerk
    jm = compute_jerk_metrics(eef_traj, joint_traj, dt=dt)
    jerk_score = float(score_jerk_against_human(jm, human_stats["jerk"]))

    # path
    path_scores = compute_path_scores(eef_traj, joint_traj, human_stats)
    cart_path = float(path_scores["cartesian_path"])
    joint_path = float(path_scores["joint_path"])

    # curvature
    curv_series = compute_curvature(eef_traj, dt=dt)
    curvature = float(compute_curvature_score(curv_series, human_stats["curvature"]))

    # orientation (wxyz fixed)
    orientation = float(compute_orientation_score(eef_quat_wxyz, human_stats["orientation"]))

    # joint-limit — your exact function (aggregates over time; slices to 7 DoF internally)
    lower_limits, upper_limits = joint_limits
    jl = float(
        compute_joint_limit_score(
            joint_trajectory=joint_traj,   # (T,7)
            lower_limits=lower_limits,     # can be >7 DoF; function slices [:7]
            upper_limits=upper_limits,
            method="gmean",
            s_floor=1e-3,
            gamma=0.7,
        )
    )

    metrics = {
        "jerk": jerk_score,
        "cartesian_path": cart_path,
        "joint_path": joint_path,
        "curvature": curvature,
        "orientation": orientation,
        "joint_limit": jl,
    }
    overall = float(sum(metrics.values()) / len(metrics))
    return metrics, overall


# -------------------- Rollout & Evaluation --------------------
def rollout(policy, env: gym.Env, success_term, horizon: int, device: torch.device) -> tuple[bool, dict]:
    """Perform a single rollout of the policy in the environment."""
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    for _ in range(horizon):
        # Prepare policy observations
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        # If there are image observations, convert them for robomimic
        if hasattr(env.cfg, "image_obs_list"):
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        traj["obs"].append(obs)

        # Policy action
        actions = policy(obs)

        # Optional un-normalization
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = ((actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Step env
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        # Record
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        # Success?
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


def evaluate_model(
    model_path: str,
    env: gym.Env,
    device: torch.device,
    success_term,
    num_rollouts: int,
    horizon: int,
    seed: int,
    output_file: str,
    *,
    do_dqs: bool = False,
    human_stats: dict | None = None,
    joint_limits: tuple | None = None,
    dqs_csv: str = "",
) -> float:
    """Evaluate a single model checkpoint across multiple rollouts. DQS is computed for successful rollouts."""
    # Seeds
    torch.manual_seed(seed)
    env.seed(seed)
    random.seed(seed)

    # Load policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=model_path, device=device, verbose=False)

    results = []
    model_name = os.path.basename(model_path)

    for trial in range(num_rollouts):
        print(f"[Model: {model_name}] Starting trial {trial}")
        terminated, traj = rollout(policy, env, success_term, horizon, device)
        results.append(terminated)
        with open(output_file, "a") as file:
            file.write(f"[Model: {model_name}] Trial {trial}: {terminated}\n")
        print(f"[Model: {model_name}] Trial {trial}: {terminated}")

        # --- DQS on success ---
        if do_dqs and terminated:
            try:
                eef_traj, eef_quat_wxyz, joint_traj, _ = extract_rollout_arrays(traj, dt=env.step_dt)
                metrics, overall = compute_dqs_for_rollout(
                    eef_traj=eef_traj,
                    eef_quat_wxyz=eef_quat_wxyz,
                    joint_traj=joint_traj,
                    human_stats=human_stats,
                    joint_limits=joint_limits,
                    dt=float(env.step_dt),
                )
                pretty = " | ".join([f"{k}:{v:.3f}" for k, v in metrics.items()]) + f" | overall:{overall:.3f}"
                print(f"[DQS] {pretty}")
                with open(output_file, "a") as file:
                    file.write(f"[DQS] {pretty}\n")

                # Optional CSV append
                if dqs_csv:
                    import csv
                    csv_exists = os.path.exists(dqs_csv)
                    with open(dqs_csv, "a", newline="") as f:
                        w = csv.DictWriter(
                            f,
                            fieldnames=["model", "seed", "trial", "jerk", "cartesian_path", "joint_path",
                                        "curvature", "orientation", "joint_limit", "overall"],
                        )
                        if not csv_exists:
                            w.writeheader()
                        w.writerow({
                            "model": model_name,
                            "seed": seed,
                            "trial": trial,
                            "jerk": metrics["jerk"],
                            "cartesian_path": metrics["cartesian_path"],
                            "joint_path": metrics["joint_path"],
                            "curvature": metrics["curvature"],
                            "orientation": metrics["orientation"],
                            "joint_limit": metrics["joint_limit"],
                            "overall": overall,
                        })
            except Exception as e:
                print(f"[DQS] failed: {e}")
                with open(output_file, "a") as file:
                    file.write(f"[DQS] failed: {e}\n")

    # Results summary
    success_rate = results.count(True) / len(results)
    with open(output_file, "a") as file:
        file.write(
            f"[Model: {model_name}] Successful trials: {results.count(True)}, out of {len(results)} trials\n"
        )
        file.write(f"[Model: {model_name}] Success rate: {success_rate}\n")
        file.write(f"[Model: {model_name}] Results: {results}\n")
        file.write("-" * 80 + "\n\n")

    print(
        f"\n[Model: {model_name}] Successful trials: {results.count(True)}, out of {len(results)} trials"
    )
    print(f"[Model: {model_name}] Success rate: {success_rate}\n")
    print(f"[Model: {model_name}] Results: {results}\n")

    return success_rate


def main() -> None:
    """Run evaluation across models and settings, with optional DQS on successes."""
    # Parse env cfg
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Robomimic expects dict observations
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None    # we control horizon
    env_cfg.recorders = None                # disable recorder
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.eval_mode = True

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device = TorchUtils.get_torch_device(try_to_use_cuda=False)

    # DQS setup once
    if args_cli.dqs:
        if args_cli.human_h5 is None:
            raise ValueError("--human_h5 is required when --dqs is set.")
        print("[DQS] Computing human reference stats...")
        human_stats = compute_reference_stats_from_demos(
            dataset_path=args_cli.human_h5, device="cpu", num_demos=args_cli.num_hum
        )
        # Joint limits from sim
        robot = env.scene["robot"]
        jl = robot.data.joint_pos_limits[0].detach().cpu().numpy()  # [DoF, 2]
        lower_limits = jl[:, 0]
        upper_limits = jl[:, 1]
        joint_limits = (lower_limits, upper_limits)
        print("[DQS] Ready.")
    else:
        human_stats = None
        joint_limits = None

    # Collect checkpoints
    model_checkpoints = [f.name for f in os.scandir(args_cli.input_dir) if f.is_file()]

    # Seeds
    seeds = random.sample(range(0, 10000), args_cli.num_seeds) if args_cli.seeds is None else args_cli.seeds

    # Settings (extend if you have others)
    settings = ["vanilla"]

    # Ensure log dir exists
    os.makedirs(args_cli.log_dir, exist_ok=True)

    # Evaluate
    for seed in seeds:
        output_path = os.path.join(args_cli.log_dir, f"{args_cli.log_file}_seed_{seed}")
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        results_summary = {"overall": {}}
        for setting in settings:
            results_summary[setting] = {}

        with open(output_path, "w") as file:
            for setting in settings:
                env.cfg.eval_type = setting
                file.write(f"Evaluation setting: {setting}\n")
                file.write("=" * 80 + "\n\n")

                print(f"Evaluation setting: {setting}")
                print("=" * 80)

                for model in model_checkpoints:
                    # Skip early checkpoints
                    try:
                        model_epoch = int(model.split(".")[0].split("_")[-1])
                    except Exception:
                        # if your filenames differ, remove this guard or adapt parsing
                        model_epoch = 0
                    if model_epoch < args_cli.start_epoch:
                        continue

                    model_path = os.path.join(args_cli.input_dir, model)
                    success_rate = evaluate_model(
                        model_path=model_path,
                        env=env,
                        device=device,
                        success_term=success_term,
                        num_rollouts=args_cli.num_rollouts,
                        horizon=args_cli.horizon,
                        seed=seed,
                        output_file=output_path,
                        do_dqs=args_cli.dqs,
                        human_stats=human_stats,
                        joint_limits=joint_limits,
                        dqs_csv=args_cli.dqs_csv,
                    )

                    results_summary[setting][model] = success_rate
                    if model not in results_summary["overall"]:
                        results_summary["overall"][model] = 0.0
                    results_summary["overall"][model] += success_rate

                    env.reset()

                file.write("=" * 80 + "\n\n")
                env.reset()

            # Average across settings
            for model in results_summary["overall"].keys():
                results_summary["overall"][model] /= len(settings)

            # Final summary
            file.write("\nResults Summary (success rate):\n")
            for setting in results_summary.keys():
                file.write(f"\nSetting: {setting}\n")
                for model in results_summary[setting].keys():
                    file.write(f"{model}: {results_summary[setting][model]}\n")
                if results_summary[setting]:
                    max_key = max(results_summary[setting], key=results_summary[setting].get)
                    file.write(
                        f"\nBest model for setting {setting} is {max_key} with success rate "
                        f"{results_summary[setting][max_key]}\n"
                    )

        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
