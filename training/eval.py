import gymnasium as gym
import torch
import numpy as np
from model import QNetwork

MODEL_PATH = "model/best_dqn_cartpole.pth"
NUM_EPISODES = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = QNetwork(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    rewards = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                action = torch.argmax(model(state_tensor)).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Mean reward: {np.mean(rewards):.1f}")
    print(f"Max reward:  {np.max(rewards):.1f}")

    env.close()

if __name__ == "__main__":
    evaluate()