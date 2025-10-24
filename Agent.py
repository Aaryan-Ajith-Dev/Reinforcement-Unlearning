import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import entropy


# ===================================================
# 1. Custom FrozenLake-like Gymnasium Environment
# ===================================================

class CustomFrozenLakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_states=16, n_actions=4, stochasticity=0.2, reward_bias=0.0, seed=None):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Discrete(n_states)

        rng = np.random.default_rng(seed)
        self.P = np.zeros((n_states, n_actions, n_states))
        for s in range(n_states):
            for a in range(n_actions):
                base = np.eye(n_states)[s]
                noise = rng.random(n_states)
                probs = (1 - stochasticity) * base + stochasticity * noise / noise.sum()
                self.P[s, a] = probs / probs.sum()

        self.R = rng.uniform(-0.1, 1.0, (n_states, n_actions)) + reward_bias
        self.terminal_states = set(rng.choice(n_states, size=2, replace=False))
        self.state = 0

    def reset(self, *, seed=None, options=None):
        self.state = np.random.randint(0, self.n_states)
        return self.state, {}

    def step(self, action):
        s = self.state
        next_state = np.random.choice(self.n_states, p=self.P[s, action])
        reward = self.R[s, action]
        done = next_state in self.terminal_states
        self.state = next_state
        return next_state, reward, done, False, {}

    def get_true_transition(self):
        return self.P.copy()

    def get_true_reward(self):
        return self.R.copy()


# ===================================================
# 2. Actor and Critic Networks
# ===================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x)


# ===================================================
# 3. Helper Functions
# ===================================================

def one_hot_state(state, dim):
    v = torch.zeros(dim)
    v[state] = 1.0
    return v

def one_hot_action(action, dim):
    v = torch.zeros(dim)
    v[action] = 1.0
    return v

def select_action(actor, state):
    probs = actor(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


# ===================================================
# 4. Training Functions
# ===================================================

def train_actor_critic(envs, actor, critic, iterations=3000, gamma=0.99):
    """Standard training across all environments."""
    device = next(actor.parameters()).device
    opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    for it in range(iterations):
        total_reward = 0.0
        for env in envs:
            state, _ = env.reset()
            done = False
            while not done:
                s = one_hot_state(state, env.n_states).to(device)
                action, log_prob = select_action(actor, s)
                next_state, reward, done, _, _ = env.step(action)

                ns = one_hot_state(next_state, env.n_states).to(device)
                a_onehot = one_hot_action(action, env.n_actions).to(device)

                with torch.no_grad():
                    next_action, _ = select_action(actor, ns)
                    next_a_onehot = one_hot_action(next_action, env.n_actions).to(device)
                    q_next = critic(ns, next_a_onehot)
                q_value = critic(s, a_onehot)

                td_target = torch.tensor(reward, device=device) + gamma * q_next * (1 - int(done))
                td_error = td_target - q_value

                opt_critic.zero_grad()
                td_error.pow(2).mean().backward()
                opt_critic.step()

                opt_actor.zero_grad()
                actor_loss = -log_prob * q_value.detach()
                actor_loss.backward()
                opt_actor.step()

                total_reward += reward
                state = next_state

        if (it + 1) % 100 == 0:
            print(f"[Train] Iter {it+1}/{iterations} | Total reward: {total_reward:.2f}")


def unlearn_environment(envs, actor, critic, unlearn_idx, iterations=2000, gamma=0.99):
    """Fine-tune to unlearn a specific environment."""
    device = next(actor.parameters()).device
    opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    for it in range(iterations):
        total_reward = 0.0
        for i, env in enumerate(envs):
            state, _ = env.reset()
            done = False
            while not done:
                s = one_hot_state(state, env.n_states).to(device)
                action, log_prob = select_action(actor, s)
                next_state, reward, done, _, _ = env.step(action)

                ns = one_hot_state(next_state, env.n_states).to(device)
                a_onehot = one_hot_action(action, env.n_actions).to(device)

                with torch.no_grad():
                    next_action, _ = select_action(actor, ns)
                    next_a_onehot = one_hot_action(next_action, env.n_actions).to(device)
                    q_next = critic(ns, next_a_onehot)
                q_value = critic(s, a_onehot)

                td_target = torch.tensor(reward, device=device) + gamma * q_next * (1 - int(done))
                td_error = td_target - q_value

                opt_critic.zero_grad()
                td_error.pow(2).mean().backward()
                opt_critic.step()

                opt_actor.zero_grad()
                q_val = critic(s, a_onehot).detach()
                actor_loss = -log_prob * q_val
                if i == unlearn_idx:
                    actor_loss = -actor_loss  # reverse gradient
                actor_loss.backward()
                opt_actor.step()

                total_reward += reward
                state = next_state

        if (it + 1) % 100 == 0:
            print(f"[Unlearn] Iter {it+1}/{iterations} | Total reward: {total_reward:.2f}")


# ===================================================
# 5. Approximation and Comparison
# ===================================================

def approximate_env(env, actor, n_samples=5000):
    """Approximate P_hat(s'|s,a) and R_hat(s,a) using policy rollouts."""
    device = next(actor.parameters()).device
    n_states, n_actions = env.n_states, env.n_actions
    P_hat = np.zeros((n_states, n_actions, n_states))
    R_hat = np.zeros((n_states, n_actions))
    counts = np.zeros((n_states, n_actions))

    for _ in range(n_samples):
        state, _ = env.reset()
        s_vec = one_hot_state(state, n_states).to(device)
        action, _ = select_action(actor, s_vec)
        next_state, reward, done, _, _ = env.step(action)

        P_hat[state, action, next_state] += 1
        R_hat[state, action] += reward
        counts[state, action] += 1

    for s in range(n_states):
        for a in range(n_actions):
            if counts[s, a] > 0:
                P_hat[s, a] /= counts[s, a]
                R_hat[s, a] /= counts[s, a]
            else:
                P_hat[s, a] = np.ones(n_states) / n_states
                R_hat[s, a] = 0.0

    return P_hat, R_hat


def compare_envs(true_P, true_R, P_hat, R_hat):
    """Compute divergence between true and reconstructed models."""
    kl_total = 0
    valid = 0
    for s in range(true_P.shape[0]):
        for a in range(true_P.shape[1]):
            if np.all(P_hat[s, a] > 0):
                kl_total += entropy(true_P[s, a], P_hat[s, a])
                valid += 1
    kl_mean = kl_total / valid if valid > 0 else float("nan")
    mse_reward = np.mean((true_R - R_hat) ** 2)
    print(f"\n=== Environment Reconstruction Metrics ===")
    print(f"Mean KL Divergence (P vs P̂): {kl_mean:.6f}")
    print(f"Reward MSE (R vs R̂): {mse_reward:.6f}")
    return kl_mean, mse_reward


# ===================================================
# 6. Run Full Experiment
# ===================================================

if __name__ == "__main__":
    n_envs = 5
    n_states, n_actions = 16, 4
    stochasticities = np.linspace(0.1, 0.5, n_envs)
    reward_biases = np.linspace(-0.2, 0.2, n_envs)

    envs = [
        CustomFrozenLakeEnv(n_states, n_actions, stochasticities[i], reward_biases[i], seed=i)
        for i in range(n_envs)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(n_states, n_actions).to(device)
    critic = Critic(n_states, n_actions).to(device)

    print("\n=== Phase 1: Joint Training ===")
    train_actor_critic(envs, actor, critic, iterations=2000)

    unlearn_idx = 2
    print(f"\n=== Phase 2: Unlearning Environment {unlearn_idx} ===")
    unlearn_environment(envs, actor, critic, unlearn_idx=unlearn_idx, iterations=1500)

    print("\n=== Phase 3: Reconstruction and Comparison ===")
    P_true = envs[unlearn_idx].get_true_transition()
    R_true = envs[unlearn_idx].get_true_reward()
    P_hat, R_hat = approximate_env(envs[unlearn_idx], actor, n_samples=3000)
    compare_envs(P_true, R_true, P_hat, R_hat)
    print("\nExperiment complete.")