import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo.atari import entombed_competitive_v3
from torch.distributions import Categorical
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pettingzoo.utils.conversions")

# Suppress all warnings
# warnings.filterwarnings("ignore")


# Hyperparameters
learning_rate = 0.0003
gamma = 0.99
clip_epsilon = 0.2
K_epochs = 4
entropy_coef = 0.01
value_loss_coef = 0.5

# CNN-based Policy Network
class CNNPolicyNetwork(nn.Module):
    def __init__(self, action_space):
        super(CNNPolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 22 * 16, 512)  # Adjusted based on conv layer outputs
        self.fc2 = nn.Linear(512, action_space.n)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch, channels, height, width]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

# CNN-based Value Network
class CNNValueNetwork(nn.Module):
    def __init__(self):
        super(CNNValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 22 * 16, 512)  # Adjusted based on conv layer outputs
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Rearrange dimensions to [batch, channels, height, width]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        state_value = self.fc2(x)
        return state_value

# PPO Agent
class PPOAgent:
    def __init__(self, action_space):
        self.policy_net = CNNPolicyNetwork(action_space).to(device)
        self.value_net = CNNValueNetwork().to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.old_policy_net = CNNPolicyNetwork(action_space).to(device)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device) / 255.0
        action_probs = self.old_policy_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def update(self, trajectory):
        states, actions, rewards, log_probs, entropies, dones = trajectory
        states = torch.tensor(states, dtype=torch.float).to(device) / 255.0
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        log_probs = torch.tensor(log_probs).to(device)
        print("ABANS", dones)
        dones = torch.tensor(dones).to(device)
        #dones = torch.tensor(dones).to(device).tolist() if isinstance(dones, torch.Tensor) else dones
        #dones = torch.tensor(dones, dtype=torch.bool).to(device)
        
        # Compute returns and advantages
        returns = []
        discounted_return = 0
        print("DESP", dones)
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_return = 0
            discounted_return = reward + gamma * discounted_return
            returns.insert(0, discounted_return)
        
        returns = torch.tensor(returns).to(device)
        advantages = returns - self.value_net(states).squeeze()

        # Optimize policy for K epochs
        for _ in range(K_epochs):
            action_probs = self.policy_net(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Ratio for PPO
            ratio = torch.exp(new_log_probs - log_probs)

            # Surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = value_loss_coef * (returns - self.value_net(states).squeeze()).pow(2).mean()

            # Total loss
            loss = policy_loss + value_loss - entropy_coef * entropy

            # Update policy and value networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

        # Update the old policy network
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())

# Environment Setup
env = entombed_competitive_v3.parallel_env(render_mode="human")
env.reset()
obs_spaces = env.observation_spaces
action_spaces = env.action_spaces

# Initialize agents
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agents = {agent: PPOAgent(action_spaces[agent]) for agent in env.agents}

maxgames = 2
maxsteps = 2

ngames = 0

while env.agents and ngames < 2:
    observations, _ = env.reset()
    done = {agent: False for agent in env.agents}
    log_probs = {agent: [] for agent in env.agents}
    entropies = {agent: [] for agent in env.agents}
    rewards = {agent: [] for agent in env.agents}
    actions = {agent: [] for agent in env.agents}
    states = {agent: [] for agent in env.agents}
    done_list = {agent: [] for agent in env.agents}

    
    steps = 0
    while not all(done.values()) and steps < maxsteps:
        for agent in env.agents:
            if not done[agent]:
                action, log_prob, entropy = agents[agent].select_action(observations[agent])
                actions[agent].append(action)
                log_probs[agent].append(log_prob)
                entropies[agent].append(entropy)
                states[agent].append(observations[agent])

        actions_to_take = {agent: actions[agent][-1] for agent in env.agents if not done[agent]}
        observations, rewards_, terminations, truncations, _ = env.step(actions_to_take)
        steps += 1
        
        for agent in env.agents:
            rewards[agent].append(rewards_[agent])
            done[agent] = terminations[agent] or truncations[agent]
            done_list[agent].append(done[agent])

    # At the end of the episode, update the policy
    print("STATES", states)
    print("ACTIONS", actions)
    print("REWARDS", rewards)
    print("LOG_PROBS", log_probs)
    print("ENTROPIES", entropies)
    print("DONE", done)
    print("DONE_LIST", done_list)

    for agent in env.agents:
        agents[agent].update((states[agent], actions[agent], rewards[agent], log_probs[agent], entropies[agent], done_list[agent]))
    
    ngames += 1

    print("Game ended", ngames)

print("All games ended", ngames)

env.close()