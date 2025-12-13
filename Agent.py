from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import entropy
from mi_grad_estimator import MIGradEstimator

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
    """Value function approximator V(s)."""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

# Make a queue based buffer using deque


class Buffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.log_probs = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))
        self.actions.append(action)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done, log_prob = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, log_prob

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

def estimate_mutual_information(env, actor, n_samples=5000):
    """
    Estimates I(S'; A | S) using empirical counts.
    """
    device = next(actor.parameters()).device
    n_states, n_actions = env.n_states, env.n_actions

    # Count tables
    joint_counts = np.zeros((n_states, n_actions, n_states))  # p(a, s' | s)
    state_counts = np.zeros(n_states)

    for _ in range(n_samples):
        s, _ = env.reset()
        s_vec = one_hot_state(s, n_states).to(device)

        with torch.no_grad():
            probs = actor(s_vec)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample().item()

        s_next, _, _, _, _ = env.step(a)

        joint_counts[s, a, s_next] += 1
        state_counts[s] += 1

    mi_total = 0.0
    valid_states = 0

    for s in range(n_states):
        if state_counts[s] == 0:
            continue

        # Normalize
        p_a_sprime = joint_counts[s] / joint_counts[s].sum()
        p_a = p_a_sprime.sum(axis=1, keepdims=True)
        p_sprime = p_a_sprime.sum(axis=0, keepdims=True)

        mi_s = 0.0
        for a in range(n_actions):
            for sp in range(n_states):
                p = p_a_sprime[a, sp]
                if p > 0 and p_a[a, 0] > 0 and p_sprime[0, sp] > 0:
                    mi_s += p * np.log(p / (p_a[a, 0] * p_sprime[0, sp]))

        mi_total += mi_s
        valid_states += 1

    return mi_total / max(valid_states, 1)

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
                with torch.no_grad():
                    v_next = critic(ns)
                v_s = critic(s)

                td_target = torch.tensor(reward, device=device) + gamma * v_next * (1 - int(done))
                advantage = td_target - v_s

                # Critic update
                opt_critic.zero_grad()
                advantage.pow(2).mean().backward()
                opt_critic.step()

                # Actor update (policy gradient with advantage)
                opt_actor.zero_grad()
                actor_loss = -log_prob * advantage.detach()
                actor_loss.backward()
                opt_actor.step()

                total_reward += reward
                state = next_state

        # if (it + 1) % 100 == 0:
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
                with torch.no_grad():
                    v_next = critic(ns)
                v_s = critic(s)

                td_target = torch.tensor(reward, device=device) + gamma * v_next * (1 - int(done))
                advantage = td_target - v_s

                # Critic update
                opt_critic.zero_grad()
                advantage.pow(2).mean().backward()
                opt_critic.step()

                # Actor update with advantage; reverse for unlearning target env
                opt_actor.zero_grad()
                actor_loss = -log_prob * advantage.detach()
                if i == unlearn_idx:
                    actor_loss = -actor_loss
                actor_loss.backward()
                opt_actor.step()

                total_reward += reward
                state = next_state

        # if (it + 1) % 100 == 0:
        print(f"[Unlearn] Iter {it+1}/{iterations} | Total reward: {total_reward:.2f}")

def unlearn_environment_mi(envs, actor, critic, unlearn_idx, buffer, mi_grad_estimator, iterations=2000, gamma=0.99):
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
                # Value-based TD target and advantage
                with torch.no_grad():
                    v_next = critic(ns)
                v_s = critic(s)
                td_target = torch.tensor(reward, device=device) + gamma * v_next * (1 - int(done))
                advantage = td_target - v_s

                # Critic update
                opt_critic.zero_grad()
                advantage.pow(2).mean().backward()
                opt_critic.step()

                # Actor update
                opt_actor.zero_grad()
                actor_loss = -log_prob * advantage.detach()
                if i == unlearn_idx:
                    # Populate buffer for MI-based unlearning
                    buffer.push(s.cpu().numpy(), action, reward, ns.cpu().numpy(), done, log_prob.item())
                    if len(buffer) == buffer.capacity:
                        actions_buf = torch.tensor(np.array(buffer.actions), dtype=torch.float32).unsqueeze(-1).to(device)
                        # Use full next-state vectors
                        next_states_buf = torch.tensor(np.array(buffer.next_states), dtype=torch.float32).to(device)
                        log_probs_buf = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32).to(device)

                        # MI gradient estimator first backward pass
                        mi_grad_estimator.learning_loss(
                            actions_buf,
                            next_states_buf
                        ).backward()

                        # Replace actor loss with MI loss (overrides policy gradient)
                        mi_loss = mi_grad_estimator.forward(
                            actions_buf,
                            next_states_buf,
                            log_probs_buf
                        )
                        actor_loss = mi_loss
                actor_loss.backward()
                opt_actor.step()

                total_reward += reward
                state = next_state

        # if (it + 1) % 100 == 0:
        print(f"[Unlearn] Iter {it+1}/{iterations} | Total reward: {total_reward:.2f}")


# ====================================
# 5. Model Save/Load Utilities
# ====================================

def save_models(actor, critic, filepath="trained_policy.pt"):
    """Save actor and critic state dicts to a file."""
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict()
    }, filepath)
    print(f"Models saved to {filepath}")

def load_models(actor, critic, filepath="trained_policy.pt", device=None):
    """Load actor and critic state dicts from a file."""
    if device is None:
        device = next(actor.parameters()).device
    checkpoint = torch.load(filepath, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"Models loaded from {filepath}")

def compute_policy_kl_divergence(actor1, actor2, n_states, n_samples=1000):
    """
    Compute KL divergence between two policies: KL(actor1 || actor2).
    
    Args:
        actor1: First policy (e.g., trained policy)
        actor2: Second policy (e.g., unlearned policy)
        n_states: Number of states in the environment
        n_samples: Number of state samples to average over
        
    Returns:
        mean_kl: Average KL divergence across sampled states
        state_kls: KL divergence for each state
    """
    device = next(actor1.parameters()).device
    actor1.eval()
    actor2.eval()
    
    state_kls = []
    
    with torch.no_grad():
        for state_idx in range(n_states):
            # Create one-hot encoded state
            state = one_hot_state(state_idx, n_states).to(device)
            
            # Get action probabilities from both policies
            probs1 = actor1(state)
            probs2 = actor2(state)
            
            # Compute KL divergence: KL(P1 || P2) = sum(P1 * log(P1/P2))
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            kl = torch.sum(probs1 * (torch.log(probs1 + eps) - torch.log(probs2 + eps)))
            state_kls.append(kl.item())
    
    mean_kl = np.mean(state_kls)
    return mean_kl, np.array(state_kls)

# ====================================
# 6. Approximation and Comparison
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
# 7. Run Full Experiment
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

    unlearn_idx = 2  # Index of environment to unlearn
    
    # ===== Phase 1: Joint Training =====
    # print("\n=== Phase 1: Joint Training ===")
    # train_actor_critic(envs, actor, critic, iterations=2000)
    # mi_train = estimate_mutual_information(envs[unlearn_idx], actor)
    # print(f"I(S';A|S) = {mi_train:.6f}")
    
    # # Save the trained policy
    # save_models(actor, critic, "trained_policy.pt")

    # ===== Phase 2: Standard Gradient Reversal Unlearning =====
    # print(f"\n=== Phase 2: Standard Unlearning Environment {unlearn_idx} ===")
    # # Create fresh models and load trained weights
    # actor_std = Actor(n_states, n_actions).to(device)
    # critic_std = Critic(n_states, n_actions).to(device)
    # load_models(actor_std, critic_std, "trained_policy.pt", device)
    
    # unlearn_environment(envs, actor_std, critic_std, unlearn_idx=unlearn_idx, iterations=1500)
    # mi_unlearn = estimate_mutual_information(envs[unlearn_idx], actor_std)
    # print(f"I(S';A|S) after standard unlearning = {mi_unlearn:.6f}")

    # ===== Phase 3: MI-based Unlearning =====
    print(f"\n=== Phase 3: MI-based Unlearning Environment {unlearn_idx} ===")
    # Create fresh models and load trained weights again
    actor_mi = Actor(n_states, n_actions).to(device)
    critic_mi = Critic(n_states, n_actions).to(device)
    load_models(actor_mi, critic_mi, "trained_policy.pt", device)
    
    buffer_capacity = 5000
    buffer = Buffer(capacity=buffer_capacity)
    mi_grad_estimator = MIGradEstimator(
        x_dim=1,  # Action dimension
        y_dim=n_states,  # Next state dimension
        hidden_size=64
    ).to(device)
    unlearn_environment_mi(envs, actor_mi, critic_mi, unlearn_idx=unlearn_idx, buffer=buffer, mi_grad_estimator=mi_grad_estimator, iterations=1500)
    mi_unlearn_mi = estimate_mutual_information(envs[unlearn_idx], actor_mi)
    print(f"I(S';A|S) after MI-based unlearning = {mi_unlearn_mi:.6f}")
    
    # ===== Optional: Reconstruction and Comparison =====
    # print("\n=== Phase 4: Reconstruction and Comparison ===")
    # P_true = envs[unlearn_idx].get_true_transition()
    # R_true = envs[unlearn_idx].get_true_reward()
    # P_hat_std, R_hat_std = approximate_env(envs[unlearn_idx], actor_std, n_samples=3000)
    # P_hat_mi, R_hat_mi = approximate_env(envs[unlearn_idx], actor_mi, n_samples=3000)
    # print("\nStandard Unlearning:")
    # compare_envs(P_true, R_true, P_hat_std, R_hat_std)
    # print("\nMI-based Unlearning:")
    # compare_envs(P_true, R_true, P_hat_mi, R_hat_mi)
    
    print("\nExperiment complete.")