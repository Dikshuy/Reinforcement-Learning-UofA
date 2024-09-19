import gymnasium
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.unwrapped.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

P = P * (1.0 - T[..., None])  # next state probability for terminal transitions is 0

def bellman_q(pi, gamma, max_iter=1000):
    delta = np.inf
    iter = 0
    Q = np.zeros((n_states, n_actions))
    be = np.zeros((max_iter))
    while delta > 1e-5 and iter < max_iter:
        Q_new = R + (np.dot(P, gamma * (Q * pi)).sum(-1))
        delta = np.abs(Q_new - Q).sum()
        be[iter] = delta
        Q = Q_new
        iter += 1
    return Q

def episode(env, Q, eps, seed):
    data = dict()
    data["s"] = []
    data["a"] = []
    data["r"] = []
    s, _ = env.reset(seed=seed)
    done = False
    while not done:
        a = eps_greedy_action(Q, s, eps)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)
        s = s_next
    return data

def eps_greedy_probs(Q, s, eps):
    # return action probabilities
    p_action = np.zeros((n_states, n_actions))
    best_action = np.random.choice(np.where(Q[s] == np.max(Q[s]))[0])
    for a in range(n_actions):
        if a == best_action:
            p_action[s][a] = 1 - eps + eps/n_actions
        else:
            p_action[s][a] = eps/n_actions

    return p_action

def eps_greedy_action(Q, s, eps):
    # return action drawn according to eps-greedy policy
    if np.random.rand() < eps:
        action = np.random.choice(n_actions)
    else:
        action = np.random.choice(np.where(Q[s] == np.max(Q[s]))[0])

    return action


def monte_carlo(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration, use_is):
    # return Q, be
    steps = 0
    eps = 1
    returns = np.zeros((n_states, n_actions))
    bellman_error = []

    while steps < max_steps:
        for _ in range(episodes_per_iteration):
            data = episode(env, Q, eps, int(seed))
            steps += len(data["s"])

            G = 0
            for t in reversed(data["s"]):
                s, a, r = data["s"][t], data["a"][t], data["r"][t]
                G = gamma * G + r
                returns[s,a] += G
                Q[s,a] = np.average(returns[s,a])

            eps = max(eps - eps_decay * steps, 0.01)
            pi = eps_greedy_probs(Q, s, eps)
            Q_original = bellman_q(pi, G)
            error = np.abs(Q-Q_original)
            bellman_error.append(error)
    print(Q)
    return Q, np.array(bellman_error)
            


def error_shade_plot(ax, data, stepsize, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

init_value = 0.0
gamma = 0.9
max_steps = 2000
horizon = 10

episodes_per_iteration = [1, 10, 50]
decays = [1, 2, 5]
seeds = np.arange(10)   # change to 50

results = np.zeros((
    len(episodes_per_iteration),
    len(decays),
    len(seeds),
    max_steps,
))

fig, axs = plt.subplots(1, 2)
plt.ion()
plt.show()

use_is = False  # repeat with True
for ax, reward_noise_std in zip(axs, [0.0, 3.0]):
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Absolute Bellman Error")
    env = gymnasium.make(
        "Gym-Gridworlds/Penalty-3x3-v0",
        max_episode_steps=horizon,
        reward_noise_std=reward_noise_std,
    )
    for j, episodes in enumerate(episodes_per_iteration):
        for k, decay in enumerate(decays):
            for seed in seeds:
                np.random.seed(seed)
                Q = np.zeros((n_states, n_actions)) + init_value
                Q, be = monte_carlo(env, Q, gamma, decay / max_steps, max_steps, episodes, use_is)
                results[j, k, seed] = be
            error_shade_plot(
                ax,
                results[j, k],
                stepsize=1,
                label=f"Episodes: {episodes}, Decay: {decay}",
            )
            ax.legend()
            plt.draw()
            plt.pause(0.001)

plt.ioff()
plt.show()