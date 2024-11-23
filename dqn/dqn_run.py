# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
import importlib
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = "dqn_run"
    """the name of this experiment"""
    load_path: str = ""
    """the path to the trained model"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False                         ######## WANDB
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "DQN"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    #env_id: str = "ALE/Entombed-v5"
    env_id: str = "entombed_cooperative_v3"
    """the id of the environment"""
    total_timesteps: int = 2000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    buffer_size: int = 100
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 0
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""


def make_env(env_id):
    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env(render_mode="human")
    #env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
    obs, _ = env.reset()
    print(obs['first_0'].shape)
    env = ss.max_observation_v0(env, 2)
    obs, _ = env.reset()
    print(obs['first_0'].shape)
    env = ss.frame_skip_v0(env, 4)
    obs, _ = env.reset()
    print(obs['first_0'].shape)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    obs, _ = env.reset()
    print(obs['first_0'].shape)
    env = ss.color_reduction_v0(env, mode="B")
    obs, _ = env.reset()
    print(obs['first_0'].shape)
    env = ss.resize_v1(env, x_size=84, y_size=84)
    obs, _ = env.reset()
    print(obs['first_0'].shape)
    env = ss.frame_stack_v1(env, 4)
    obs, _ = env.reset()
    print(obs['first_0'].shape)
    env = ss.agent_indicator_v0(env, type_only=False)
    obs, _ = env.reset()
    print(obs['first_0'].shape)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    obs, _ = env.reset()
    print(obs.shape)
    return env


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(6, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def get_unique_run_name(base_run_name):
    # Check if folder with base_run_name exists
    run_name = base_run_name
    counter = 1

    # Append a number if the folder already exists
    while os.path.exists(f"runs/{run_name}"):
        run_name = f"{base_run_name}_{counter}"
        counter += 1
    
    return run_name

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    #args = tyro.cli(Args)
    args = tyro.cli(Args)

    print(args.load_path)

    assert os.path.exists(args.load_path)

    if args.env_id == "entombed_cooperative_v3":
        run_name = f"coop_{args.exp_name}_{args.num_envs // 2}_{args.total_timesteps}"
    else:
        run_name = f"comp_{args.exp_name}_{args.num_envs // 2}_{args.total_timesteps}"

    run_name = get_unique_run_name(run_name)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args.env_id)
    # env setup
    envs = ss.concat_vec_envs_v1(env, args.num_envs // 2, num_cpus=0, base_class="gymnasium")
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    start_time = time.time()

    episodic_returns = [[0, 0] for _ in range(args.num_envs // 2)]
    episodic_lengths = [[0, 0] for _ in range(args.num_envs // 2)]


    # LOAD MODEL
    model = QNetwork(envs).to(device)
    model.load_state_dict(torch.load(args.load_path))
    model.eval()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step % 100 == 0:
            print(f"global_step = {global_step}")
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device).permute(0, 3, 1, 2))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        dones = torch.maximum(torch.tensor(terminations).to(device), torch.tensor(truncations).to(device))
        
        rewards = list(map(lambda x: -5.0 if x == 1 else 0.01, rewards))  ##WE ADD A SMALL REWARD JUST FOR SURVIVAL

        for idx, rew in enumerate(rewards):
                player_idx = idx % 2
                game = idx // 2
                episodic_returns[game][player_idx] += rew
                episodic_lengths[game][player_idx] += 1

                # If life is lost, log the episodic return and reset trackers
                if rewards[game*2 + player_idx] < 0:
                    #print(rewards[game*2 + player_idx])
                    print(
                        f"game={game}, global_step={global_step}, player{player_idx}-episodic_return={episodic_returns[game][player_idx]}-episodic_length={episodic_lengths[game][player_idx]}"
                    )
                    writer.add_scalar(
                        f"Rewards/R-G{game}-P{player_idx}",
                        episodic_returns[game][player_idx],
                        global_step,
                    )
                    writer.add_scalar(
                        f"GameTime/T-G{game}-P{player_idx}",
                        episodic_lengths[game][player_idx],
                        global_step,
                    )

                    # Reset for the next episode
                    episodic_returns[game][player_idx] = 0
                    episodic_lengths[game][player_idx] = 0

        #print("OBS:", obs)
        #print("REAL_NEXT_OBS:", real_next_obs)
        #print("ACTIONS:", actions)
        #print("REWARDS:", rewards)
        #print("TERMINATIONS:", terminations)
        #print("INFOS:", infos)

        obs = next_obs

    envs.close()
    writer.close()
