# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
import importlib
import matplotlib.pyplot as plt
import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
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
    #exp_name: str = os.path.basename(__file__)[: -len(".py")]
    exp_name: str = "_vsrun_"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False                         ######## WANDB
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "VS_Run"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    ### GENERIC
    env_id: str = "entombed_cooperative_v3"
    """the id of the environment"""
    total_timesteps: int = 1000
    """total timesteps of the experiments"""
    num_envs: int = 2
    """the number of parallel game environments"""

    load_path0: str = None      # Load path for model 0
    load_path1: str = None      # Load path for model 1

def parse_args():
    parser = argparse.ArgumentParser(description="Run script with optional dataclass arguments")
    parser.add_argument("--load_path0", type=str, help="Path to the first model")
    parser.add_argument("--load_path1", type=str, help="Path to the second model")
    args = parser.parse_args()
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env():
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

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        return self.critic(self.network(x.permute((0, 3, 1, 2))))

    def get_action_and_value(self, x, action=None):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        hidden = self.network(x.permute((0, 3, 1, 2)))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

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

    parsed_args = parse_args()

    args = Args(
        load_path0=parsed_args.load_path0 if parsed_args.load_path0 else None,
        load_path1=parsed_args.load_path1 if parsed_args.load_path1 else None
    )

    if args.load_path0 == None: args.load_path0 = "runs/coop_dqn_vs_ppo_8_50000/coop_P0_dqn_model_8_50000"
    if args.load_path1 == None: args.load_path1 = "runs/coop_dqn_vs_ppo_8_50000/coop_P1_ppo_model_8_50000"
    
    print(args.load_path0)
    print(args.load_path1)

    first_model = "dqn"
    second_model= "ppo"

    if args.env_id == "entombed_cooperative_v3":
        run_name = f"coop_{first_model + args.exp_name + second_model}_{args.num_envs // 2}_{args.total_timesteps}"
    else:
        run_name = f"comp_{first_model + args.exp_name + second_model}_{args.num_envs // 2}_{args.total_timesteps}"
    
    run_name = get_unique_run_name(run_name)

    args.load_path0 = os.path.join(os.getcwd(), args.load_path0)
    args.load_path1 = os.path.join(os.getcwd(), args.load_path1)

    print(args.load_path0)
    print(args.load_path1)
    
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

    env = make_env()
    # env setup
    envs = ss.concat_vec_envs_v1(env, args.num_envs // 2, num_cpus=0, base_class="gymnasium")
    obs, _ = envs.reset()
    #print(list(obs.keys()))
    print("SECOND", obs.shape)
    envs.single_observation_space = envs.observation_space
    obs, _ = envs.reset()
    print(obs.shape)
    envs.single_action_space = envs.action_space
    obs, _ = envs.reset()
    print(obs.shape)
    envs.is_vector_env = True
    obs, _ = envs.reset()
    print(obs.shape)

    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    episodic_returns = [[0, 0] for _ in range(args.num_envs // 2)]
    episodic_lengths = [[0, 0] for _ in range(args.num_envs // 2)]

    # Start the game
    obs, _ = envs.reset(seed=args.seed)

    ###
    ###### MODEL LOAD
    ###

    if first_model == "dqn":
        q_network0 = QNetwork(envs).to(device)
        q_network0.load_state_dict(torch.load(args.load_path0))
        q_network0.eval()
    elif first_model == "ppo":
        agent0 = Agent(envs).to(device)
        agent0.load_state_dict(torch.load(args.load_path0))
        agent0.eval()

    else:
        raise Exception("Sorry, first model doesn't have a correct model name")
    
    if second_model == "dqn":
        q_network1 = QNetwork(envs).to(device)
        q_network1.load_state_dict(torch.load(args.load_path1))
        q_network1.eval()

    elif second_model == "ppo":
        agent1 = Agent(envs).to(device)
        agent1.load_state_dict(torch.load(args.load_path1))
        agent1.eval()

    else:
        raise Exception("Sorry, second model doesn't have a correct model name") 

    ###
    ######
    ###
    
    start_time = time.time()

    for global_step in range(args.total_timesteps):
        if global_step % 100 == 0:
            print(f"global_step = {global_step}")

        if global_step % 10000 == 1:
            print("ACTIONS", actions)

        if first_model == "dqn":
            q_values0 = q_network0(torch.Tensor(obs[::2]).to(device).permute(0, 3, 1, 2))
            step_actions0 = torch.argmax(q_values0, dim=1).cpu().numpy()

        elif first_model == "ppo":
            with torch.no_grad():
                step_actions0, _, _, _ = agent0.get_action_and_value(torch.Tensor(obs[::2]).to(device))

        if second_model == "dqn":
            q_values1 = q_network1(torch.Tensor(obs[1::2]).to(device).permute(0, 3, 1, 2))
            step_actions1 = torch.argmax(q_values1, dim=1).cpu().numpy()

        elif second_model == "ppo":
            with torch.no_grad():
                step_actions1, _, _, _ = agent1.get_action_and_value(torch.Tensor(obs[1::2]).to(device))

        
        actions = np.ravel(np.column_stack((step_actions0, step_actions1)))
        
        #if (global_step < 1000):
        #    actions = np.array([12 for _ in range(args.num_envs)])
        #else:
        #    print("NOW")
        #    actions = np.array([13 for _ in range(args.num_envs)])
        print("ACTIONS", actions)
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        dones = torch.maximum(torch.tensor(terminations).to(device), torch.tensor(truncations).to(device))
        
        rewards = list(map(lambda x: -5.0 if x == 1 else 0.01, rewards))  ## COOP REWARDS
        #rewards = list(map(lambda x: 0.01 if x == 0 else 5*x, rewards))  ## COMP REWARDS

        #print("DONES", dones)

        for idx, act in enumerate(actions):
            if (act == 5):
                rewards[idx] += 0.02
            elif (act == 13): 
                rewards[idx] += 0.01
            if (act == 2):
                rewards[idx] -= 0.02
            elif (act == 10): 
                rewards[idx] -= 0.01

        for idx, rew in enumerate(rewards):
                player_idx = idx % 2
                game = idx // 2
                episodic_returns[game][player_idx] += rew
                episodic_lengths[game][player_idx] += 1

                writer.add_scalar(
                    f"Rewards/R-G{game}-P{player_idx}",
                    episodic_returns[game][player_idx],
                    global_step,
                )

                # If done, log the episodic return and reset trackers
                if rewards[game*2 + player_idx] < 0:
                    #print(rewards[game*2 + player_idx])
                    print(
                        f"game={game}, global_step={global_step}, player{player_idx}-episodic_return={episodic_returns[game][player_idx]}-episodic_length={episodic_lengths[game][player_idx]}"
                    )
                    #writer.add_scalar(
                    #    f"Rewards/R-G{game}-P{player_idx}",
                    #    episodic_returns[game][player_idx],
                    #    global_step,
                    #)
                    writer.add_scalar(
                        f"GameTime/T-G{game}-P{player_idx}",
                        episodic_lengths[game][player_idx],
                        global_step,
                    )

                    # Reset for the next episode
                    episodic_returns[game][player_idx] = 0
                    episodic_lengths[game][player_idx] = 0

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

    envs.close()
    writer.close()
