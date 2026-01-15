# 🧠 Reinforcement Learning from Scratch — DQN on CartPole

This project is a **from-scratch implementation of Deep Q-Learning (DQN)** applied to the **CartPole-v1** environment using **PyTorch** and **Gymnasium**.

The goal of this project is to **explore how reinforcement learning works in practice**, including:
- Learning through interaction instead of labeled data
- Exploration vs exploitation tradeoffs
- Temporal difference learning
- Instability in value-based methods

---

## 🎯 Project Goals

- Implement **DQN from scratch** (no RL libraries)
- Understand the **Bellman optimality equation**
- Use **experience replay** and a **target network**
- Apply **Double DQN** to reduce overestimation bias
- Diagnose and fix training instability
- Achieve consistent performance above **200 average reward**

---

## 🧩 Environment

- **Environment:** CartPole-v1
- **Observation space:** 4 continuous values
- **Action space:** 2 discrete actions
- **Reward:** +1 per timestep
- **Max episode length:** 500 steps

> While CartPole-v1 allows up to 500 reward, an average reward ≥ 200 over 50 episodes is commonly used as a “solved” benchmark (v0 uses 200 as the threshold).

---

## 🏗️ Project Structure

```text
.
├── training/
│   ├── train.py          # Main DQN / Double DQN training loop
│   ├── eval.py           # Evaluation (ε = 0, greedy policy)
│   ├── model.py          # Q-network definition
│   ├── replay_buffer.py  # Experience replay buffer
│   ├── utils.py          # Epsilon decay schedule
│
├── model/
│   ├── best_dqn_cartpole.pth   # Best saved model (ignored by git)
│   └── .gitkeep                # Keeps directory tracked
│
├── requirements.txt
└── README.md
```

## 🧠 Algorithm Overview

### Deep Q-Learning (DQN)

We approximate the Q-function with a neural network:

$$Q(s,a) \approx Q_\theta(s,a)$$

Training minimizes the Bellman error:

$$L = \mathbb{E}[(Q(s,a) - (r + \gamma \max_{a'} Q(s',a)))^2]$$

### Key Components Implemented

- ✔ Experience Replay
- ✔ Target Network
- ✔ Double DQN
- ✔ ε-greedy Exploration
- ✔ Reward-based Early Stopping
- ✔ Best Model Checkpointing

### ⚙️ Hyperparameters (Final)

```python
GAMMA = 0.99
BATCH_SIZE = 64
LR = 5e-4
NUM_EPISODES = 500
TARGET_UPDATE_FREQ = 1000

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 10_000
```

> **Tweaking Tip:** Lowering the epsilon floor prevents random actions from destabilizing a near-optimal policy. Changing these values can significantly impact training stability. Go ahead and try different values to see how they affect performance! Epsilon_decay controls how quickly exploration decreases. Epsilon_end is the minimum exploration rate (random action probability).

## 🚀 Training

From the project root:

```bash
python training/train.py
```

During training, the script:
- Logs episode reward and epsilon
- Tracks average reward over the last 50 episodes
- Saves the best model
- Stops early once the environment is solved

## 📊 Evaluation

Evaluate the trained policy with no exploration:

```bash
python training/eval.py
```

Evaluation runs with ε = 0 to measure the true policy performance.

## 📈 Results

- Initial DQN plateaued around ~180 average reward
- Performance degraded with long training due to instability
- Double DQN + improved exploration schedule:
  - Consistently achieved >200 average reward
  - Triggered early stopping
  - Demonstrates why value-based RL methods are unstable and why modern methods exist

## 🧪 Key Lessons Learned

- Exploration scheduling matters more than network size
- Training metrics ≠ evaluation performance
- Value-based RL can regress after learning a good policy
- Reward tracking must occur at the episode level
- Double DQN improves stability but does not fully solve it

## References

- Sutton & Barto — Reinforcement Learning: An Introduction
- Mnih et al. (2015) — Human-level control through deep reinforcement learning
- Gymnasium Documentation