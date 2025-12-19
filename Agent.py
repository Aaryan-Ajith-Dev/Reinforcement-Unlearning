from collections import deque
from scipy.stats import entropy
import gymnasium as gym
from gymnasium import spaces
# from tqdm import tqdm

import numpy as np
import math

import torch 
import torch.nn as nn

from mi_grad_estimator import MIGradEstimator, MIGradEstimatorDiscrete
import torch.optim as optim
import torch.nn.functional as F

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

# Make a queue based buffer using deque


# ----------------------------
# Buffer (stores numpy arrays; log_probs NOT stored for gradient reasons)
# ----------------------------
class Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.states = deque(maxlen=capacity)

    def push(self, state_np, action_int, reward, next_state_np, done):
        # state_np and next_state_np should be numpy arrays (one-hot)
        self.buffer.append((state_np, action_int, reward, next_state_np, done))
        self.states.append(state_np)
        self.actions.append(action_int)
        self.next_states.append(next_state_np)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


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
    opt_actor = optim.Adam(actor.parameters(), lr=1e-5)
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
    opt_actor = optim.Adam(actor.parameters(), lr=1e-5)
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


def encode_state_action(state_idx, action_idx, n_states, n_actions, device):
    s_oh = torch.nn.functional.one_hot(
        torch.tensor(state_idx, device=device),
        num_classes=n_states
    ).float()
    a_oh = torch.nn.functional.one_hot(
        torch.tensor(action_idx, device=device),
        num_classes=n_actions
    ).float()
    return torch.cat([s_oh, a_oh], dim=-1)  # [n_states + n_actions]


def unlearn_environment_mi(envs, actor, critic, unlearn_idx, buffer, mi_grad_estimator,
                           iterations=2000, gamma=0.99, mi_train_steps=1):
    """
    mi_train_steps: how many gradient steps to take on the MI estimator (on detached data)
    """
    device = next(actor.parameters()).device
    opt_actor = optim.Adam(actor.parameters(), lr=1e-5)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)
    opt_mi = optim.Adam(mi_grad_estimator.parameters(), lr=1e-3)

    n_actions = envs[0].n_actions
    n_states = envs[0].n_states

    for it in range(iterations):
        total_reward = 0.0
        for i, env in enumerate(envs):
            state, _ = env.reset()
            done = False
            # make sure opt_mi is defined (optimizer for mi_grad_estimator)

            while not done:
                s = one_hot_state(state, env.n_states).to(device)
                action, log_prob = select_action(actor, s)
                next_state, reward, done, _, _ = env.step(action)

                ns = one_hot_state(next_state, env.n_states).to(device)
                a_onehot = one_hot_action(action, env.n_actions).to(device)

                # Critic update (Q-learning style you had)
                with torch.no_grad():
                    next_action, _ = select_action(actor, ns)
                    next_a_onehot = one_hot_action(next_action, env.n_actions).to(device)
                    q_next = critic(ns, next_a_onehot)
                q_value = critic(s, a_onehot)

                td_target = torch.tensor(reward, device=device) + gamma * q_next * (1 - int(done))
                td_error = td_target - q_value

                opt_critic.zero_grad()
                (td_error.pow(2).mean()).backward()
                opt_critic.step()

                # Default actor loss (actor-critic)
                q_val = critic(s, a_onehot).detach()             # treat critic output as baseline/value
                actor_loss = (-log_prob * q_val)                 # REINFORCE-style PG term

                # If this is the unlearn environment, collect data & apply MI-based update
                if i == unlearn_idx:
                    # store raw indices (not one-hot) in buffer
                    buffer.push(
                        state,       # int state index
                        action,      # int action index
                        reward,
                        next_state,  # int next-state index
                        done
                    )

                    if len(buffer) == buffer.capacity:
                        # --- build tensors ---
                        states_tensor = torch.tensor(buffer.states, dtype=torch.long, device=device)      # (N,)
                        actions_tensor = torch.tensor(buffer.actions, dtype=torch.long, device=device)    # (N,)
                        next_states_tensor = torch.tensor(buffer.next_states, dtype=torch.long, device=device)  # (N,)

                        # --- one-hot encodings ---
                        states_onehot = F.one_hot(states_tensor, num_classes=n_states).float()          # (N, n_states)
                        actions_onehot = F.one_hot(actions_tensor, num_classes=n_actions).float()       # (N, n_actions)
                        sa_embed = torch.cat([states_onehot, actions_onehot], dim=-1)                  # (N, n_states + n_actions)

                        next_states_onehot = F.one_hot(next_states_tensor, num_classes=n_states).float()  # (N, n_states)

                        # --- recompute log_probs for actor gradient ---
                        with torch.enable_grad():
                            probs = actor(states_onehot)  # or use states_tensor if actor expects indices
                            dist = torch.distributions.Categorical(probs)
                            log_probs_buf = dist.log_prob(actions_tensor)  # (N,) connected to actor

                        # --- Train MI estimator (detached data) ---
                        for p in mi_grad_estimator.parameters():
                            p.requires_grad = True
                        
                        for _ in range(mi_train_steps):
                            opt_mi.zero_grad()
                            mi_est_loss = mi_grad_estimator.learning_loss(sa_embed.detach(), next_states_onehot.detach())
                            mi_est_loss.backward()
                            opt_mi.step()

                        # --- Compute actor MI loss ---
                        for p in mi_grad_estimator.parameters():
                            p.requires_grad = False
                        mi_weighted_loss, mi = mi_grad_estimator.forward(
                            sa_embed.detach(),
                            next_states_onehot.detach(),
                            log_probs_buf
                        )

                        # override actor loss
                        actor_loss = mi_weighted_loss

                        # restore MI params
                        for p in mi_grad_estimator.parameters():
                            p.requires_grad = True


                # Backprop actor and step (single place)
                opt_actor.zero_grad()
                actor_loss.backward()

                opt_actor.step()

                total_reward += reward
                state = next_state

        if (it + 1) % 100 == 0:
            # try to show latest mi estimate safely
            mi_val = float("nan")
            try:
                mi_val = mi.item()
            except Exception:
                pass
            print(f"[Unlearn] Iter {it+1}/{iterations} | Total reward: {total_reward:.2f} | MI (last): {mi_val:.6f}")

# ====================================
# 5. Approximation and Comparison
# ====================================

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

def exact_mutual_information(env, actor):
    """
    Computes exact I(S'; A | S) using true transition probabilities.
    Assumes uniform distribution over states.
    """
    device = next(actor.parameters()).device
    P = env.get_true_transition()   # shape: (S, A, S')
    n_states, n_actions, _ = P.shape

    mi_total = 0.0

    for s in range(n_states):
        # π(a|s)
        s_vec = one_hot_state(s, n_states).to(device)
        with torch.no_grad():
            pi = actor(s_vec).cpu().numpy()  # shape: (A,)

        # p(s'|s) = Σ_a π(a|s) P(s'|s,a)
        p_sprime = np.sum(pi[:, None] * P[s], axis=0)  # shape: (S',)

        mi_s = 0.0
        for a in range(n_actions):
            if pi[a] == 0:
                continue

            P_sa = P[s, a]  # P(s'|s,a)

            # KL(P(.|s,a) || p(.|s))
            mask = (P_sa > 0) & (p_sprime > 0)
            kl = np.sum(P_sa[mask] * np.log(P_sa[mask] / p_sprime[mask]))

            mi_s += pi[a] * kl

        mi_total += mi_s

    return mi_total / n_states


if __name__ == "__main__":
    n_envs = 1
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
    # train_actor_critic(envs, actor, critic, iterations=200)

    # # save checkpoint 
    # torch.save({
    #     'actor_state_dict': actor.state_dict(),
    #     'critic_state_dict': critic.state_dict(),
    # }, 'joint_trained_checkpoint.pth')
    mi_value = exact_mutual_information(envs[0], actor)
    print(f"Exact I(S'; A | S) after joint training: {mi_value:.6f}")

    unlearn_idx = 0
    # print(f"\n=== Phase 2: Unlearning Environment {unlearn_idx} ===")

    # unlearn_environment(envs, actor, critic, unlearn_idx=unlearn_idx, iterations=500)
    # mi_value = exact_mutual_information(envs[unlearn_idx], actor)
    # print(f"Exact I(S'; A | S) after unlearning: {mi_value:.6f}")

    print("\n=== Unlearning Via MI minimisation ===")
    unlearn_environment_mi(
        envs, actor, critic, unlearn_idx=unlearn_idx,
        buffer=Buffer(capacity=64),
        mi_grad_estimator=MIGradEstimatorDiscrete(
            x_dim=n_actions + n_states,
            y_dim=n_states,
            hidden_size=128
        ).to(device),
        iterations=500
    )
    print("\n=== Exact Mutual Information After Unlearning ===")
    mi_value = exact_mutual_information(envs[unlearn_idx], actor)
    print(f"Exact I(S'; A | S) after unlearning: {mi_value:.6f}")

    # print("\n=== Phase 3: Reconstruction and Comparison ===")
    # P_true = envs[unlearn_idx].get_true_transition()
    # R_true = envs[unlearn_idx].get_true_reward()
    # P_hat, R_hat = approximate_env(envs[unlearn_idx], actor, n_samples=3000)
    # compare_envs(P_true, R_true, P_hat, R_hat)
    # print("\nExperiment complete.")