import gymnasium as gym
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from training.model import QNetwork
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

app = FastAPI()

DEVICE = torch.device("cpu")

# Load model once at startup
model = QNetwork(4, 2).to(DEVICE)
model.load_state_dict(torch.load(BASE_DIR / "model" / "best_dqn_cartpole.pth", map_location=DEVICE))
model.eval()

@app.get("/", response_class=HTMLResponse)
def index():
    with open(BASE_DIR / "api" / "frontend.html") as f:
        return f.read()

@app.post("/run_episode")
def run_episode():
    env = gym.make("CartPole-v1")
    state, _ = env.reset()

    done = False
    total_reward = 0
    trajectory = []

    while not done:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32)
            action = torch.argmax(model(s)).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        trajectory.append({
            "cart_position": float(state[0]),
            "cart_velocity": float(state[1]),
            "pole_angle": float(state[2]),
            "pole_velocity": float(state[3]),
            "action": int(action),
            "reward": float(reward),
        })

        state = next_state
        total_reward += reward

    env.close()

    return {
        "total_reward": float(total_reward),
        "steps": int(len(trajectory)),
        "trajectory": trajectory
    }
