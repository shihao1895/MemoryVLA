import os
from dataclasses import dataclass
from typing import List, Union
import draccus
import numpy as np
import tqdm

os.environ['MUJOCO_GL'] = 'osmesa'
# apt install -y libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev

from libero.libero import benchmark

from libero_utils import (
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from robot_utils import DATE_TIME, set_seed_everywhere

# let tensorflow only see CPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


@dataclass
class GenerateConfig:
    # fmt: off
    task_suite_name: str = "libero_spatial" # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10 # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50 # Number of rollouts per task
    spcial_task_id: Union[List[int], int, None] = None # List of task IDs to evaluate (default: None, evaluates all tasks in the suite)
    run_id_note: str = ""
    local_log_dir: str = "./logs/eval_libero" # Local directory for eval logs
    seed: int = 7 # Random Seed (for reproducibility)
    resolution: Union[int, tuple] = 256 # Image resolution for model input
    port: int = 6800
    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    if cfg.spcial_task_id is not None and isinstance(cfg.spcial_task_id, int):
        cfg.spcial_task_id = [cfg.spcial_task_id]

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize local logging
    run_id = f"{cfg.task_suite_name}-{cfg.num_trials_per_task}trials-seed{cfg.seed}-{cfg.run_id_note}-{DATE_TIME}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = cfg.resolution # TODO need to be done in the cfg

    ################################################################
    ### import Policy
    from vla_policy import LLaVAClient
    policy = LLaVAClient(base_url=f'http://localhost:{cfg.port}')
    ################################################################

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Skip tasks if specified
        if cfg.spcial_task_id is not None and task_id not in cfg.spcial_task_id:
            print(f"Skipping task {task_id}...")
            continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()
            policy.reset()
            episode_first_frame = 'True'

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    observation = {
                        "image": img,
                        "states": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    ###############################################################################
                    ### here we need to use flask to get the action from the model
                    ### we need a policy to get the action
                    action = policy.process_frame(text=task_description,
                                                  episode_first_frame=episode_first_frame,
                                                  **observation)

                    episode_first_frame = 'False'

                    # Adjust gripper action values according to your data's gripper definition
                    for i in range(len(action)):
                        if action[i][-1] == 1.0:
                            action[i][-1] = -1.0
                        elif action[i][-1] == 0.0:
                            action[i][-1] = 1.0

                    done_flag = False
                    for i in range(len(action)):
                        # Execute action in environment
                        obs, reward, done, info = env.step(action[i])
                        if done:
                            task_successes += 1
                            total_successes += 1
                            done_flag = True
                            break
                        t += 1

                    if done_flag:
                        break

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            rollout_dir = os.path.join(cfg.local_log_dir, run_id + "_videos")
            save_rollout_video(
                replay_images, total_episodes,
                success=done, task_description=task_description,
                log_file=log_file,
                rollout_dir=rollout_dir,
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    log_file.close()

if __name__ == "__main__":
    eval_libero()
