import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from envs.darkroom_env import DarkroomEnv

# --- Environment Definition (from your prompt) ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- PPO Implementation (Identical to before) ---

class ReplayBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = ReplayBuffer()
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

# --- Main Training Loop ---

if __name__ == '__main__':
    # --- Hyperparameters ---
    # Environment
    DIM = 9
    GOAL = [7, 8]
    ## CHANGED: Increased horizon to give the agent more time to explore
    HORIZON = 50 

    # PPO
    ## CHANGED: Increased total timesteps and batch size for more stable learning
    TOTAL_TIMESTEPS = 50000
    TIMESTEPS_PER_BATCH = 2048
    K_EPOCHS = 80 # A few more epochs can help with the larger batch size
    EPS_CLIP = 0.2
    GAMMA = 0.99
    LR_ACTOR = 0.0003
    LR_CRITIC = 0.001

    # --- Setup ---
    env = DarkroomEnv(dim=DIM, goal=GOAL, horizon=HORIZON)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP)
    
    print(f"Training on {device}...")

    time_step = 0
    i_episode = 0
    all_rewards = []
    
    while time_step <= TOTAL_TIMESTEPS:
        state, _ = env.reset()
        current_ep_reward = 0
        
        for t in range(1, HORIZON + 1):
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if len(agent.buffer.rewards) == TIMESTEPS_PER_BATCH:
                agent.update()

            if done:
                break
        
        all_rewards.append(current_ep_reward)
        if i_episode % 50 == 0:
            avg_reward = np.mean(all_rewards[-50:]) if len(all_rewards) > 0 else 0.0
            print(f"Episode {i_episode}\tTime Step: {time_step}\tAverage Reward: {avg_reward:.2f}")
        
        i_episode += 1

    print("\nTraining finished.")

    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards)
    if len(all_rewards) > 100:
        moving_avg = np.convolve(all_rewards, np.ones(100)/100, mode='valid')
        plt.plot(moving_avg, linewidth=3, label='Moving Average (100 episodes)')
    plt.title('Total Reward per Episode during Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nEvaluating trained agent...")
    eval_episodes = 20
    for ep in range(eval_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        path = [state.astype(int)]
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            path.append(state.astype(int))
            done = terminated or truncated
            ep_reward += reward
        print(f"Evaluation episode {ep+1}: Reward = {ep_reward:.2f}, Path Length = {len(path)}")
