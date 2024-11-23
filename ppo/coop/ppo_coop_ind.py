"""Advanced training script adapted from CleanRL's repository: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_pettingzoo_ma_atari.py.

This is a full training script including CLI, logging and integration with TensorBoard and WandB for experiment tracking.

Full documentation and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy.

Note: default value for total-timesteps has been changed from 2 million to 8000, for easier testing.

Authors: Costa (https://github.com/vwxyzjn), Elliot (https://github.com/elliottower)
"""

# flake8: noqa

import argparse
import importlib
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,    ###### CANVIAR AIXO SI VULL CREAR PROJECTE AL COMPTE DE WANDB
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Entombed-Coop-Ind",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="entombed_cooperative_v3",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=4000,  # CleanRL default: 2000000
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=2,                          #### NOMBRE D'AGENTS (PER TANT NOMBRE ENTORNS /= 2)
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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


def computation(number, global_step, agent, optimizer, obs, actions, logprobs, rewards, values, terminations, truncations, next_obs, next_termination, next_truncation, envs):
        # bootstrap value if not done
        with torch.no_grad():
            print("Entro no_grad()")
            next_value = agent.get_value(next_obs).reshape(1, -1)
            print("Next value", next_value)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            print("NTerm i Ntrun", next_termination, next_truncation)
            next_done = torch.maximum(next_termination, next_truncation)
            print("Next done", next_done)
            dones = torch.maximum(terminations, truncations)
            #print("next_value =", next_value)
            #print("Rewards =", rewards)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    print("IFFFFFF")
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    print("ELSEEEE")
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                #print("reward[t] =", rewards[t])
                print(rewards[t], args.gamma, nextvalues, nextnonterminal, values[t])

                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                print(delta, args.gamma, args.gae_lambda, nextnonterminal, lastgaelam)
                print("Advantages:", advantages[t])
                print("Lastgaelam:", lastgaelam)
                lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
                
                print("Advantages:", advantages[t])
                print("Lastgaelam:", lastgaelam)

                # If lastgaelam is a scalar and advantages[t] has a different shape, you need to broadcast or expand it
                #lastgaelam = lastgaelam.expand_as(advantages[t])  # Expand scalar to match the shape of advantages[t]

                print("Advantages:", advantages[t])
                print("Lastgaelam:", lastgaelam)

                # Assign to advantages[t] and lastgaelam
                advantages[t] = lastgaelam

                print("Advantages:", advantages[t])
                print("Lastgaelam:", lastgaelam)

            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                print("Raroraro", b_obs.size(), b_actions.size())
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
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/Player{number}-learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("Player{number}-losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("Player{number}-losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("Player{number}-losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("Player{number}-losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("Player{number}-losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("Player{number}-losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("Player{number}-losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        if (number == 1): writer.add_scalar(
            "charts/Player{number}SPS", int(global_step / (time.time() - start_time)), global_step
        )


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # env setup
    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env(render_mode="human")
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(
        env, args.num_envs // 2, num_cpus=0, base_class="gymnasium"
    )
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Two agents, one for each player
    agent1 = Agent(envs).to(device)
    agent2 = Agent(envs).to(device)

    optimizer1 = optim.Adam(agent1.parameters(), lr=args.learning_rate, eps=1e-5)
    optimizer2 = optim.Adam(agent2.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage for each agent
    obs1 = torch.zeros((args.num_steps, args.num_envs // 2) + envs.single_observation_space.shape).to(device)
    obs2 = torch.zeros((args.num_steps, args.num_envs // 2) + envs.single_observation_space.shape).to(device)
    actions1 = torch.zeros((args.num_steps, args.num_envs // 2) + envs.single_action_space.shape).to(device)
    actions2 = torch.zeros((args.num_steps, args.num_envs // 2) + envs.single_action_space.shape).to(device)
    logprobs1 = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    logprobs2 = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    rewards1 = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    rewards2 = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    values1 = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    values2 = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    terminations1 = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    terminations2 = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    truncations1 = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)
    truncations2 = torch.zeros((args.num_steps, args.num_envs // 2)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, info = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_termination = torch.zeros(args.num_envs).to(device)
    next_truncation = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer1.param_groups[0]["lr"] = lrnow
            optimizer2.param_groups[0]["lr"] = lrnow

        episodic_returns = [[0, 0] for _ in range(args.num_envs // 2)]
        episodic_lengths = [[0, 0] for _ in range(args.num_envs // 2)]
        
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs1[step] = torch.tensor(next_obs[::2]).to(device)
            obs2[step] = torch.tensor(next_obs[1::2]).to(device)
            terminations1[step] = torch.tensor(next_termination[::2]).to(device).view(-1)
            terminations2[step] = torch.tensor(next_termination[1::2]).to(device).view(-1)
            truncations1[step] = torch.tensor(next_truncation[::2]).to(device).view(-1)
            truncations2[step] = torch.tensor(next_truncation[1::2]).to(device).view(-1)

            # Fetch actions and values from agent1 for player 1
            with torch.no_grad():
                action1, logprob1, _, value1 = agent1.get_action_and_value(next_obs[::2])
                values1[step] = value1.flatten()
            actions1[step] = action1
            logprobs1[step] = logprob1

            # Fetch actions and values from agent2 for player 2
            with torch.no_grad():
                action2, logprob2, _, value2 = agent2.get_action_and_value(next_obs[1::2])
                values2[step] = value2.flatten()
            actions2[step] = action2
            logprobs2[step] = logprob2

            # Combine actions from both agents and step the environment
            #print(f"action1 shape: {action1.shape}, action2 shape: {action2.shape}")

            #actions_combined = torch.stack([action1, action2], dim=1).cpu().numpy()
            actions_combined = torch.cat([action1, action2], dim=0).cpu().numpy()
            next_obs, reward, termination, truncation, info = envs.step(actions_combined)

            next_obs, next_termination, next_truncation = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(termination).to(device),
                torch.Tensor(truncation).to(device),
            )

            dones = torch.maximum(torch.tensor(termination).to(device), torch.tensor(truncation).to(device))

            #print("Before", reward)
            reward = list(map(lambda x: -5.0 if x == 1 else 0.1, reward))  ##WE ADD A SMALL REWARD JUST FOR SURVIVAL
            #print("After", reward)
            
            for idx, rew in enumerate(reward):
                player_idx = idx % 2
                game = idx // 2
                episodic_returns[game][player_idx] += rew
                episodic_lengths[game][player_idx] += 1

                # If done, log the episodic return and reset trackers
                if dones[game*2 + player_idx]:
                    print(reward[game*2 + player_idx])
                    print(
                        f"game={game}, global_step={global_step}, player{player_idx}-episodic_return={episodic_returns[game][player_idx]}"
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

            # Process rewards separately for each player
            rewards1[step] = torch.tensor(reward[::2]).to(device).view(-1)
            rewards2[step] = torch.tensor(reward[1::2]).to(device).view(-1)
        
        #print("Agent", agent1, agent2)
        #print("Optimizer", optimizer1, optimizer2)
        #print("Obs", obs1.size(), obs2.size())
        #print("Actions", actions1.size(), actions2.size())
        #print("Logprobs", logprobs1.size(), logprobs2.size())
        #print("Rewards", rewards1.size(), rewards2.size())
        #print("Values", values1.size(), values2.size())

        #for num, agent, optimizer, obs, actions, logprobs, rewards, values in zip(
        #    [0,1],
        #    [agent1, agent2],
        #    [optimizer1, optimizer2],
        #    [obs1, obs2],
        #    [actions1, actions2],
        #    [logprobs1, logprobs2],
        #    [rewards1, rewards2],
        #    [values1, values2],
        #    [terminations1, terminations2],
        #    [truncations1, truncations2],
        #):
        computation(0, global_step, agent1, optimizer1, obs1, actions1, logprobs1, rewards1, values1, terminations1, truncations1, next_obs, next_termination[0::2], next_truncation[0::2], envs)
        computation(1, global_step, agent2, optimizer2, obs2, actions2, logprobs2, rewards2, values2, terminations2, truncations2, next_obs, next_termination[1::2], next_truncation[1::2], envs)

    envs.close()
    writer.close()
