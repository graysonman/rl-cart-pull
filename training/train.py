import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np

from model import QNetwork
from replay_buffer import ReplayBuffer
from utils import epsilon_step_by_step
from collections import deque

# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 64
LR = 5e-4
NUM_EPISODES = 500
TARGET_UPDATE_FREQ = 1000

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_net = QNetwork(state_dim, action_dim).to(device)

target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer()

# This is to track the average reward over the last 50 episodes to save a best model
reward_window = deque(maxlen=50)
best_avg_reward = -float("inf")

global_step = 0
solved = False

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        global_step += 1
        epsilon = epsilon_step_by_step(global_step)

        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).to(device)
                action = torch.argmax(q_net(s)).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # Learn only if buffer is large enough
        if len(replay_buffer) < BATCH_SIZE:
            continue

        batch = replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Q(s,a)
        q_values = q_net(states).gather(1, actions).squeeze()

        # Bellman target
        with torch.no_grad():
            next_actions = q_net(next_states).argmax(1)
            max_next_q = target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze()
            target = rewards + GAMMA * max_next_q * (1 - dones)

        loss = F.mse_loss(q_values, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # stabilization
        if global_step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())
        
    # Use reward tracking to save best model
    reward_window.append(episode_reward)

    if len(reward_window) == reward_window.maxlen:
        avg_reward = sum(reward_window) / len(reward_window)

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(q_net.state_dict(), "model/best_dqn_cartpole.pth")
            print(f"🔥 New best model saved (avg reward = {avg_reward:.1f})")

        # CartPole is solved when it averages 200 reward over 50 consecutive episodes
        if avg_reward >= 200:
            print("✅ Environment solved. Stopping training.")
            solved = True
            break

    print(f"Episode {episode} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.3f}")

    if solved:
        break

if not solved:
    torch.save(q_net.state_dict(), "model/last_dqn_cartpole.pth")
    print("Model saved to model/last_dqn_cartpole.pth")
env.close()
