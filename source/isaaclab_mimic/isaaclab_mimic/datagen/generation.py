# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import defaultdict
import contextlib
import torch
from typing import Any
import numpy as np

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, TerminationTermCfg

from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool
from isaaclab_mimic.datagen.threshold import auto_compute_thresholds_from_demos

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_mimic.datagen.dqs.dqs_stats import compute_reference_stats_from_demos
from isaaclab_mimic.datagen.dqs.dqs_module import DQSEvaluator
from isaaclab_mimic.datagen.dqs.human_dqs import score_humans_loo_with_dqs_inline

# global variable to keep track of the data generation statistics
num_success = 0
num_failures = 0
num_attempts = 0
import csv
import os
from datetime import datetime
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
# Generate file name with date-time stamp (once per script run)
import csv
from datetime import datetime

# Create timestamp-based file name once at script start
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DQS_CSV_PATH = f"dqs_scores_{timestamp}.csv"
CSV_LOCK = asyncio.Lock()
DQS_HEADER_KEYS = None
async def run_data_generator(
    env: ManagerBasedRLMimicEnv,
    env_id: int,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    data_generator: DataGenerator,
    success_term: TerminationTermCfg,
    pause_subtask: bool = False,

):
    """Run mimic data generation from the given data generator in the specified environment index.

    Args:
        env: The environment to run the data generator on.
        env_id: The environment index to run the data generation on.
        env_reset_queue: The asyncio queue to send environment (for this particular env_id) reset requests to.
        env_action_queue: The asyncio queue to send actions to for executing actions.
        data_generator: The data generator instance to use.
        success_term: The success termination term to use.
        pause_subtask: Whether to pause the subtask during generation.
    """

    global num_success, num_failures, num_attempts, num_saved_demos
    robot = env.scene["robot"]
    num_saved_demos = 0 
    while True:
        results = await data_generator.generate(
            env_id=env_id,
            success_term=success_term,
            env_reset_queue=env_reset_queue,
            env_action_queue=env_action_queue,
            pause_subtask=pause_subtask,
        )
        if bool(results["success"]):
            try:
                # === Extract data ===
                observations = results.get("observations")
                eef_pos = np.stack([
                    obs["policy"]["eef_pos"][env_id].cpu().numpy()
                    for obs in results["observations"]
                ])
                eef_quat = np.stack([
                    obs["policy"]["eef_quat"][env_id].cpu().numpy()
                    for obs in results["observations"]
                ])
                joint_pos = np.stack([
                    obs["policy"]["joint_pos"][env_id].cpu().numpy()
                    for obs in results["observations"]
                ])

                if len(eef_pos) < 4 or len(joint_pos) < 4:
                    print(f"[ENV {env_id}] Skipping: too short")
                    return  # Original return statement

                lower_limits, upper_limits = env.joint_limits

                # === Compute and Print Score ===
                dqs_scores = env.dqs_evaluator.compute(
                    eef_traj=eef_pos,
                    joint_pos=joint_pos,
                    joint_traj=joint_pos,
                    eef_quat=eef_quat,
                    jacobians=None,
                    joint_limits=(lower_limits, upper_limits),
                )
                overall_score = env.dqs_evaluator.overall_score()
                async with env.csv_lock:
                    if env.dqs_header_keys is None:
                        env.dqs_header_keys = list(dqs_scores.keys())
                        with open(env.csv_path, mode="w", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(["demo_number", "env_id", "overall_score"] + env.dqs_header_keys)

                    num_saved_demos += 1  # increment only when actually saving
                    with open(env.csv_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        row = [num_saved_demos, env_id, overall_score] + [dqs_scores[k] for k in env.dqs_header_keys]
                        writer.writerow(row)
                # print(f"[ENV {env_id}] DQS Scores: {dqs_scores}, Overall Score: {overall_score:.4f}")

            except Exception as e:
                print(f"[ENV {env_id}] DQS computation failed: {e}")
                return  # Original return statement

        if bool(results["success"]):
            num_success += 1
        else:
            num_failures += 1
        num_attempts += 1


def env_loop(
    env: ManagerBasedRLMimicEnv,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    shared_datagen_info_pool: DataGenInfoPool,
    asyncio_event_loop: asyncio.AbstractEventLoop,
):
    """Main asyncio loop for the environment."""
    global num_success, num_failures, num_attempts
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    prev_num_attempts = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            while env_action_queue.qsize() != env.num_envs:
                asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                while not env_reset_queue.empty():
                    env_id_tensor[0] = env_reset_queue.get_nowait()
                    env.reset(env_ids=env_id_tensor)
                    env_reset_queue.task_done()

            actions = torch.zeros(env.action_space.shape)

            for i in range(env.num_envs):
                env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                actions[env_id] = action

            env.step(actions)
            for i in range(env.num_envs):
                env_action_queue.task_done()

            if prev_num_attempts != num_attempts:
                prev_num_attempts = num_attempts
                generated_sucess_rate = 100 * num_success / num_attempts if num_attempts > 0 else 0.0
                print("")
                print("*" * 50, "\033[K")
                print(f"{num_success}/{num_attempts} ({generated_sucess_rate:.1f}%) successful demos generated by mimic\033[K")
                print("*" * 50, "\033[K")

                generation_guarantee = env.cfg.datagen_config.generation_guarantee
                generation_num_trials = env.cfg.datagen_config.generation_num_trials
                check_val = num_success if generation_guarantee else num_attempts
                if check_val >= generation_num_trials:
                    print(f"Reached {generation_num_trials} successes/attempts. Exiting.")
                    break

            if env.sim.is_stopped():
                break
    env.close()


def setup_env_config(
    env_name: str,
    output_dir: str,
    output_file_name: str,
    num_envs: int,
    device: str,
    generation_num_trials: int | None = None,
) -> tuple[Any, Any]:
    """Configure the environment for data generation."""
    env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)

    if generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = generation_num_trials

    env_cfg.env_name = env_name

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def setup_async_generation(
    env: Any, num_envs: int, input_file: str, success_term: Any, pause_subtask: bool = False,
) -> dict[str, Any]:
    """Setup async data generation tasks."""
    asyncio_event_loop = asyncio.get_event_loop()
    env_reset_queue = asyncio.Queue()
    env_action_queue = asyncio.Queue()
    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = DataGenInfoPool(env, env.cfg, env.device, asyncio_lock=shared_datagen_info_pool_lock)
    shared_datagen_info_pool.load_from_dataset_file(input_file)
    thresholds = auto_compute_thresholds_from_demos(input_file, device=env.device, num_demos=10)
    print(f"Computed thresholds: {thresholds}")
    human_stats = compute_reference_stats_from_demos(input_file, device=env.device, num_demos=10)
    robot = env.scene["robot"]
    jl = robot.data.joint_pos_limits[0].cpu().numpy()
    lower_limits, upper_limits = jl[:, 0], jl[:, 1]
    env.csv_path = DQS_CSV_PATH
    env.csv_lock = CSV_LOCK
    env.dqs_header_keys = None
    # LOO DQS for your 10 human demos
    try:
        human_loo_rows = score_humans_loo_with_dqs_inline(
            dataset_path=input_file,
            device=str(env.device),
            num_demos=10,
            stored_quat_order="wxyz",
            q_min_full=lower_limits,
            q_max_full=upper_limits,
        )
        for r in human_loo_rows[:10]:
            print({k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in r.items()})
    except Exception as e:
        print(f"[WARN] Human LOO DQS failed: {e}")

    print(f"Loaded {shared_datagen_info_pool.num_datagen_infos} to datagen info pool")

    data_generator = DataGenerator(env=env, src_demo_datagen_info_pool=shared_datagen_info_pool)
    joint_limits = robot.data.joint_pos_limits[0]
    lower_limits = joint_limits[:, 0].cpu().numpy()
    upper_limits = joint_limits[:, 1].cpu().numpy()

    # Store necessary components in the env for access within the coroutine
    env.joint_limits = (lower_limits, upper_limits)
    env.demo_thresholds = thresholds
    env.dqs_evaluator = DQSEvaluator(human_stats)

    data_generator_asyncio_tasks = []
    for i in range(num_envs):
        task = asyncio_event_loop.create_task(
            run_data_generator(
                env, i, env_reset_queue, env_action_queue, data_generator, success_term, pause_subtask=pause_subtask
            )
        )
        data_generator_asyncio_tasks.append(task)

    return {
        "tasks": data_generator_asyncio_tasks,
        "event_loop": asyncio_event_loop,
        "reset_queue": env_reset_queue,
        "action_queue": env_action_queue,
        "info_pool": shared_datagen_info_pool,
    }