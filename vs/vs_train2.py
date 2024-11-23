# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
import importlib
import matplotlib.pyplot as plt
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
    exp_name: str = "_vs2_"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False                         ######## WANDB
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "VS2_Train"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_models: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    ### GENERIC
    env_id: str = "entombed_competitive_v3"
    """the id of the environment"""
    total_timesteps: int = 5000
    """total timesteps of the experiments"""
    num_envs: int = 2
    """the number of parallel game environments"""

    ### DQN0
    learning_rate_d0: float = 1e-4           # The learning rate of the optimizer
    buffer_size0: int = 100                  # The replay memory buffer size
    gamma_d0: float = 0.99                   # The discount factor gamma
    tau0: float = 1.0                        # The target network update rate
    target_network_frequency0: int = 1000    # The timesteps it takes to update the target network
    batch_size_d0: int = 32                  # The batch size of sample from the reply memory
    start_e0: float = 1                      # The starting epsilon for exploration
    end_e0: float = 0.01                     # The ending epsilon for exploration
    exploration_fraction0: float = 0.10      # The fraction of `total-timesteps` it takes from start-e to go end-e
    learning_starts0: int = 0                # Timestep to start learning
    train_frequency0: int = 4                # The frequency of training

    ### PPO0
    learning_rate_p0: float = 2.5e-4        # The learning rate of the optimizer                    # Number of parallel game environments
    num_steps0: int = 128                    # Steps per policy rollout
    anneal_lr0: bool = True
    gamma_p0: float = 0.99                     # Discount factor gamma
    gae_lambda0: float = 0.95                # Lambda for general advantage estimation
    num_minibatches0: int = 4
    update_epochs0: int = 4
    norm_adv0: bool = True                   # Advantage normalization
    clip_coef0: float = 0.1                  # Surrogate clipping coefficient
    clip_vloss0: bool = True                 # Clipped loss for the value function
    ent_coef0: float = 0.01                  # Coefficient of entropy
    vf_coef0: float = 0.5                    # Coefficient of the value function
    max_grad_norm0: float = 0.5              # Maximum norm for gradient clipping
    target_kl0: float = None
    batch_size_p0: int = None
    minibatch_size0: int = None


    ### DQN1
    learning_rate_d1: float = 1e-4           # The learning rate of the optimizer
    buffer_size1: int = 100                  # The replay memory buffer size
    gamma_d1: float = 0.99                   # The discount factor gamma
    tau1: float = 1.0                        # The target network update rate
    target_network_frequency1: int = 1000    # The timesteps it takes to update the target network
    batch_size_d1: int = 32                  # The batch size of sample from the reply memory
    start_e1: float = 1                      # The starting epsilon for exploration
    end_e1: float = 0.01                     # The ending epsilon for exploration
    exploration_fraction1: float = 0.10      # The fraction of `total-timesteps` it takes from start-e to go end-e
    learning_starts1: int = 0                # Timestep to start learning
    train_frequency1: int = 4                # The frequency of training

    ### PPO1
    learning_rate_p1: float = 2.5e-4        # The learning rate of the optimizer                    # Number of parallel game environments
    num_steps1: int = 128                    # Steps per policy rollout
    anneal_lr1: bool = True
    gamma_p1: float = 0.99                     # Discount factor gamma
    gae_lambda1: float = 0.95                # Lambda for general advantage estimation
    num_minibatches1: int = 4
    update_epochs1: int = 4
    norm_adv1: bool = True                   # Advantage normalization
    clip_coef1: float = 0.1                  # Surrogate clipping coefficient
    clip_vloss1: bool = True                 # Clipped loss for the value function
    ent_coef1: float = 0.01                  # Coefficient of entropy
    vf_coef1: float = 0.5                    # Coefficient of the value function
    max_grad_norm1: float = 0.5              # Maximum norm for gradient clipping
    target_kl1: float = None
    batch_size_p1: int = None
    minibatch_size1: int = None

    def __post_init__(self):
        # Compute derived fields
        print("HELLO")
        self.batch_size_p0 = (self.num_envs // 2) * self.num_steps0
        self.minibatch_size0 = self.batch_size_p0 // self.num_minibatches0
        self.batch_size_p1 = (self.num_envs // 2) * self.num_steps1
        self.minibatch_size1 = self.batch_size_p1 // self.num_minibatches1


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
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.network(x / 255.0)

def dqn_training(num_agent, global_step, rb, q_network, target_network, writer, optimizer):
    # ALGO LOGIC: training.
    if global_step > getattr(args, f'learning_starts{num_agent}'):
        if global_step % getattr(args, f'train_frequency{num_agent}') == 0:
            data = rb.sample(getattr(args, f'batch_size_d{num_agent}'))
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations.permute(0, 3, 1, 2)).max(dim=1)
                td_target = data.rewards.flatten() + getattr(args, f'gamma_d{num_agent}') * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations.permute(0, 3, 1, 2)).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar(f"losses/P{num_agent}-td_loss", loss, global_step)
                writer.add_scalar(f"losses/P{num_agent}-q_values", old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                if (num_agent == 0): writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update target network
        if global_step % getattr(args, f'target_network_frequency{num_agent}') == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    getattr(args, f'tau{num_agent}') * q_network_param.data + (1.0 - getattr(args, f'tau{num_agent}')) * target_network_param.data
                )


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
        self.actor = layer_init(nn.Linear(512, 10), std=0.01)
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

def ppo_training(num_agent, global_step, agent, optimizer, next_obs, next_termination, next_truncation, rewards, terminations, truncations, values, obs, logprobs, actions):
    # bootstrap value if not done
    with torch.no_grad():
        print("Entro no_grad()")
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        next_done = torch.maximum(next_termination, next_truncation)
        dones = torch.maximum(terminations, truncations)
        #print("next_value =", next_value)
        #print("Rewards =", rewards)
        for t in reversed(range(getattr(args, f'num_steps{num_agent}'))):
            if t == getattr(args, f'num_steps{num_agent}') - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            #print("reward[t] =", rewards[t])
            #print(rewards[t], args.gamma, nextvalues, nextnonterminal, values[t])

            delta = (
                rewards[t] + getattr(args, f'gamma_p{num_agent}') * nextvalues * nextnonterminal - values[t]
            )
            #print(delta, args.gamma, args.gae_lambda, nextnonterminal, lastgaelam)
            #print("Advantages:", advantages[t])
            #print("Lastgaelam:", lastgaelam)
            advantages[t] = lastgaelam = (
                delta + getattr(args, f'gamma_p{num_agent}') * getattr(args, f'gae_lambda{num_agent}') * nextnonterminal * lastgaelam
            )
            #print("Advantages:", advantages[t])
            #print("Lastgaelam:", lastgaelam)
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(getattr(args, f'batch_size_p{num_agent}'))
    clipfracs = []
    for epoch in range(getattr(args, f'update_epochs{num_agent}')):
        np.random.shuffle(b_inds)
        for start in range(0, getattr(args, f'batch_size_p{num_agent}'), getattr(args, f'minibatch_size{num_agent}')):
            end = start + getattr(args, f'minibatch_size{num_agent}')
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions.long()[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [
                    ((ratio - 1.0).abs() > getattr(args, f'clip_coef{num_agent}')).float().mean().item()
                ]

            mb_advantages = b_advantages[mb_inds]
            if getattr(args, f'norm_adv{num_agent}'):
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - getattr(args, f'clip_coef{num_agent}'), 1 + getattr(args, f'clip_coef{num_agent}')
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if getattr(args, f'clip_vloss{num_agent}'):
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -getattr(args, f'clip_coef{num_agent}'),
                    getattr(args, f'clip_coef{num_agent}'),
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - getattr(args, f'ent_coef{num_agent}') * entropy_loss + v_loss * getattr(args, f'vf_coef{num_agent}')

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), getattr(args, f'max_grad_norm{num_agent}'))
            optimizer.step()

        if getattr(args, f'target_kl{num_agent}') is not None:
            if approx_kl > getattr(args, f'target_kl{num_agent}'):
                break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar(
        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    writer.add_scalar(f"losses/P{num_agent}-value_loss", v_loss.item(), global_step)
    writer.add_scalar(f"losses/P{num_agent}-policy_loss", pg_loss.item(), global_step)
    writer.add_scalar(f"losses/P{num_agent}-entropy", entropy_loss.item(), global_step)
    writer.add_scalar(f"losses/P{num_agent}-old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar(f"losses/P{num_agent}-approx_kl", approx_kl.item(), global_step)
    writer.add_scalar(f"losses/P{num_agent}-clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar(f"losses/P{num_agent}-explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar(
        "charts/SPS", int(global_step / (time.time() - start_time)), global_step
    )


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


action_mapping = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]

def translate_action(new_action):
    return action_mapping[new_action]


if __name__ == "__main__":

    args = Args()
    
    first_model = "dqn"
    second_model= "dqn"

    if args.env_id == "entombed_cooperative_v3":
        run_name = f"coop_{first_model + args.exp_name + second_model}_{args.num_envs // 2}_{args.total_timesteps}"
    else:
        run_name = f"comp_{first_model + args.exp_name + second_model}_{args.num_envs // 2}_{args.total_timesteps}"
    
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
    #assert 1 == 0
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
    ###### MODEL INITIALIZATION
    ###

    if first_model == "dqn":
        q_network0 = QNetwork(envs).to(device)
        optimizer0 = optim.Adam(q_network0.parameters(), lr=args.learning_rate_d0)
        target_network0 = QNetwork(envs).to(device)
        target_network0.load_state_dict(q_network0.state_dict())
        #print(envs.single_observation_space)  # Check original observation space details
        rb0 = ReplayBuffer(
            args.buffer_size0,
            #gym.spaces.Box(low=0, high=255, shape=(6, 84, 84), dtype=np.uint8),
            envs.single_observation_space,
            gym.spaces.Discrete(10),
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
            n_envs= args.num_envs // 2
        )
    elif first_model == "ppo":
        agent0 = Agent(envs).to(device)
        optimizer0 = optim.Adam(agent0.parameters(), lr=args.learning_rate_p0, eps=1e-5)
        obs0 = torch.zeros((args.num_steps0, args.num_envs // 2) + envs.single_observation_space.shape).to(device)
        actions0 = torch.zeros((args.num_steps0, args.num_envs //2) + envs.single_action_space.shape).to(device)
        logprobs0 = torch.zeros((args.num_steps0, args.num_envs // 2)).to(device)
        rewards0 = torch.zeros((args.num_steps0, args.num_envs // 2)).to(device)
        terminations0 = torch.zeros((args.num_steps0, args.num_envs // 2)).to(device)
        truncations0 = torch.zeros((args.num_steps0, args.num_envs // 2)).to(device)
        values0 = torch.zeros((args.num_steps0, args.num_envs // 2)).to(device)
        next_obs0 = torch.Tensor(obs[::2]).to(device)
        next_termination0 = torch.zeros(args.num_envs // 2).to(device)
        next_truncation0 = torch.zeros(args.num_envs // 2).to(device)
        num_updates0 = args.total_timesteps // args.batch_size_p0

    else:
        raise Exception("Sorry, first model doesn't have a correct model name")
    
    if second_model == "dqn":
        q_network1 = QNetwork(envs).to(device)
        optimizer1 = optim.Adam(q_network1.parameters(), lr=args.learning_rate_d1)
        target_network1 = QNetwork(envs).to(device)
        target_network1.load_state_dict(q_network1.state_dict())
        #print(envs.single_observation_space)  # Check original observation space details
        rb1 = ReplayBuffer(
            args.buffer_size1,
            #gym.spaces.Box(low=0, high=255, shape=(6, 84, 84), dtype=np.uint8),
            envs.single_observation_space,
            gym.spaces.Discrete(10),
            device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
            n_envs= args.num_envs // 2
        )
    elif second_model == "ppo":
        agent1 = Agent(envs).to(device)
        optimizer1 = optim.Adam(agent1.parameters(), lr=args.learning_rate_p1, eps=1e-5)
        obs1 = torch.zeros((args.num_steps1, args.num_envs // 2) + envs.single_observation_space.shape).to(device)
        actions1 = torch.zeros((args.num_steps1, args.num_envs //2) + envs.single_action_space.shape).to(device)
        logprobs1 = torch.zeros((args.num_steps1, args.num_envs // 2)).to(device)
        rewards1 = torch.zeros((args.num_steps1, args.num_envs // 2)).to(device)
        terminations1 = torch.zeros((args.num_steps1, args.num_envs // 2)).to(device)
        truncations1 = torch.zeros((args.num_steps1, args.num_envs // 2)).to(device)
        values1 = torch.zeros((args.num_steps1, args.num_envs // 2)).to(device)
        next_obs1 = torch.Tensor(obs[1::2]).to(device)
        next_termination1 = torch.zeros(args.num_envs // 2).to(device)
        next_truncation1 = torch.zeros(args.num_envs // 2).to(device)
        num_updates1 = args.total_timesteps // args.batch_size_p1

    else:
        raise Exception("Sorry, second model doesn't have a correct model name") 

    ###
    ######
    ###
    
    start_time = time.time()

    for global_step in range(args.total_timesteps):
        if global_step % 100 == 0:
            print(f"global_step = {global_step}")

        if global_step % 100000 == 1:
            print("ACTIONS", actions)

        if first_model == "dqn":
            
            epsilon = linear_schedule(args.start_e0, args.end_e0, args.exploration_fraction0 * args.total_timesteps, global_step)
            if random.random() < epsilon:
                #print("RANDOM", epsilon)
                step_actions0 = np.array([gym.spaces.Discrete(10).sample() for _ in range(envs.num_envs // 2)])
            else:
                #print("NORMAL", epsilon)
                q_values0 = q_network0(torch.Tensor(obs[::2]).to(device).permute(0, 3, 1, 2))
                step_actions0 = torch.argmax(q_values0, dim=1).cpu().numpy()

        elif first_model == "ppo":

            if args.anneal_lr0:
                frac = 1.0 - (global_step // num_updates0) / num_updates0
                lrnow = frac * args.learning_rate_p0
                optimizer0.param_groups[0]["lr"] = lrnow

            obs0[global_step % args.num_steps0] = next_obs0
            terminations0[global_step % args.num_steps0] = next_termination0
            truncations0[global_step % args.num_steps0] = next_truncation0

            # ALGO LOGIC: action logic
            with torch.no_grad():
                step_actions0, logprob, _, value = agent0.get_action_and_value(next_obs0)
                values0[global_step % args.num_steps0] = value.flatten()
            actions0[global_step % args.num_steps0] = step_actions0
            logprobs0[global_step % args.num_steps0] = logprob

        if second_model == "dqn":
            epsilon = linear_schedule(args.start_e1, args.end_e1, args.exploration_fraction1 * args.total_timesteps, global_step)
            if random.random() < epsilon:
                #print("RANDOM", epsilon)
                step_actions1 = np.array([gym.spaces.Discrete(10).sample() for _ in range(envs.num_envs // 2)])
            else:
                #print("NORMAL", epsilon)
                q_values1 = q_network1(torch.Tensor(obs[1::2]).to(device).permute(0, 3, 1, 2))
                step_actions1 = torch.argmax(q_values1, dim=1).cpu().numpy()

        elif second_model == "ppo":

            if args.anneal_lr1:
                frac = 1.0 - (global_step // num_updates1) / num_updates1
                lrnow = frac * args.learning_rate_p1
                optimizer1.param_groups[0]["lr"] = lrnow

            obs1[global_step % args.num_steps1] = next_obs1
            terminations1[global_step % args.num_steps1] = next_termination1
            truncations1[global_step % args.num_steps1] = next_truncation1

            # ALGO LOGIC: action logic
            with torch.no_grad():
                step_actions1, logprob, _, value = agent1.get_action_and_value(next_obs1)
                values1[global_step % args.num_steps1] = value.flatten()
            actions1[global_step % args.num_steps1] = step_actions1
            logprobs1[global_step % args.num_steps1] = logprob

        
        actions = np.ravel(np.column_stack((step_actions0, step_actions1)))

        #if (global_step < 300):
        #    actions = np.array([5 for _ in range(args.num_envs)])
        #else:
        #    print("NOW")
        #    actions = np.array([1 for _ in range(args.num_envs)])
        print("ACTIONS", actions)
        
        next_obs, rewards, terminations, truncations, infos = envs.step(np.array([translate_action(action) for action in actions]))

        dones = torch.maximum(torch.tensor(terminations).to(device), torch.tensor(truncations).to(device))
        
        #print(dones)

        #rewards = list(map(lambda x: -5.0 if x == 1 else 0.01, rewards))  ## COOP REWARDS
        rewards = list(map(lambda x: -2 if x == -1 else 0.01, rewards))  ## COMP REWARDS
        #print("MINIMUM REWARD", min(rewards))

        for idx, act in enumerate(actions):
            #if (act == 5):
            #    rewards[idx] += 0.02
            #if (act == 9): 
            #    rewards[idx] += 0.01
            if (act == 2):
                rewards[idx] -= 0.02
            if (act == 6): 
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

                # If any of the two players died, write game_length
                if (rewards[game*2 + player_idx] < -1) or (rewards[game*2 + (1-player_idx)] < -1):
                    #print(rewards[game*2 + player_idx])
                    print(f"game={game}, global_step={global_step}, player{player_idx}-episodic_return={episodic_returns[game][player_idx]}-episodic_length={episodic_lengths[game][player_idx]}")
                    
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

        #print("OBS:", obs)
        #print("REAL_NEXT_OBS:", real_next_obs)
        #print("ACTIONS:", actions)
        #print("REWARDS:", rewards)
        #print("TERMINATIONS:", terminations)
        #print("INFOS:", infos)

        if first_model == "dqn":

            real_next_obs0 = next_obs[::2].copy()
            rb0.add(obs[::2], real_next_obs0, actions[::2], rewards[::2], terminations[::2], infos[::2])
            dqn_training(0, global_step, rb0, q_network0, target_network0, writer, optimizer0)

        elif first_model == "ppo":
            
            next_obs0, next_termination0, next_truncation0 = (
                torch.Tensor(next_obs[::2]).to(device),
                torch.Tensor(terminations[::2]).to(device),
                torch.Tensor(truncations[::2]).to(device),
            )
            rewards0[global_step % args.num_steps0] = torch.tensor(rewards[::2]).to(device).view(-1)
            
            if (global_step > 0) and (global_step % args.num_steps0 == 0):
                ppo_training(0, global_step, agent0, optimizer0, next_obs0, next_termination0, next_truncation0, rewards0, terminations0, truncations0, values0, obs0, logprobs0, actions0)
            

        if second_model == "dqn":

            real_next_obs1 = next_obs[1::2].copy()
            rb1.add(obs[1::2], real_next_obs1, actions[1::2], rewards[1::2], terminations[1::2], infos[1::2])
            dqn_training(1, global_step, rb1, q_network1, target_network1, writer, optimizer1)

        elif second_model == "ppo":
            
            next_obs1, next_termination1, next_truncation1 = (
                torch.Tensor(next_obs[1::2]).to(device),
                torch.Tensor(terminations[1::2]).to(device),
                torch.Tensor(truncations[1::2]).to(device),
            )
            rewards1[global_step % args.num_steps1] = torch.tensor(rewards[1::2]).to(device).view(-1)

            if (global_step > 0) and (global_step % args.num_steps1 == 0):
                ppo_training(1, global_step, agent1, optimizer1, next_obs1, next_termination1, next_truncation1, rewards1, terminations1, truncations1, values1, obs1, logprobs1, actions1)


        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

    if args.save_models:
        if args.env_id == "entombed_cooperative_v3":
            model0_path = f"runs/{run_name}/coop_P0_{first_model}_model_{args.num_envs // 2}_{args.total_timesteps}"
            model1_path = f"runs/{run_name}/coop_P1_{second_model}_model_{args.num_envs // 2}_{args.total_timesteps}"
        else:
            model0_path = f"runs/{run_name}/comp_P0_{first_model}_model_{args.num_envs // 2}_{args.total_timesteps}"
            model1_path = f"runs/{run_name}/comp_P1_{second_model}_model_{args.num_envs // 2}_{args.total_timesteps}"

        if first_model == "dqn":
            torch.save(q_network0.state_dict(), model0_path)
            print(f"{first_model} model saved to {model0_path}")

        elif first_model == "ppo":
            torch.save(agent0.state_dict(), model0_path)
            print(f"{first_model} model saved to {model0_path}")

        if second_model == "dqn":
            torch.save(q_network1.state_dict(), model1_path)
            print(f"{second_model} model saved to {model1_path}")

        elif second_model == "ppo":
            torch.save(agent1.state_dict(), model1_path)
            print(f"{second_model} model saved to {model1_path}")

    envs.close()
    writer.close()
