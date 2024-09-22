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

# next state probability for terminal transitions is 0
P = P * (1.0 - T[..., None])


def bellman_q(pi, gamma):
    I = np.eye(n_states * n_actions)
    P_under_pi = (
        P[..., None] * pi[None, None]
    ).reshape(n_states * n_actions, n_states * n_actions)
    return (
        R.ravel() * np.linalg.inv(I - gamma * P_under_pi)
    ).sum(-1).reshape(n_states, n_actions)


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


def eps_greedy_probs(Q, eps):
    # return action probabilities
    pi = np.ones_like(Q) * (eps / n_actions)
    best_actions = np.argmax(Q, axis=1)
    for s in range(Q.shape[0]):
        pi[s, best_actions[s]] += (1 - eps)
    return pi


def eps_greedy_action(Q, s, eps):
    # return action drawn according to eps-greedy policy
    if np.random.rand() < eps:
        action = np.random.choice(n_actions)
    else:
        action = np.random.choice(np.where(Q[s] == np.max(Q[s]))[0])

    return action


def monte_carlo(env, Q, gamma, eps_decay, max_steps, episodes_per_iteration, use_is, _seed):
    # return Q, be
    eps_behavior = 1
    eps_target = 0.01
    C = np.zeros((n_states, n_actions))
    bellman_error = np.zeros(max_steps)

    pi = eps_greedy_probs(Q, eps_target)
    Q_true = bellman_q(pi, gamma)
    bellman_error[0] = np.abs(Q - Q_true).sum()
    total_steps = 1

    while total_steps < max_steps:
        current_step = total_steps
        episodes_data = []

        for _ in range(episodes_per_iteration):
            data = episode(env, Q, eps_behavior, seed=_seed)
            episodes_data.append(data)
            episode_steps = len(data["s"])
            total_steps += episode_steps

            eps_behavior = max(eps_behavior - eps_decay * episode_steps, 0.01)

            if total_steps >= max_steps:
                break

        for data in episodes_data:
            G = 0
            W = 1

            for t in reversed(range(len(data["s"]))):
                s, a, r = data["s"][t], data["a"][t], data["r"][t]
                G = gamma * G + r 
                C[s, a] += W
                Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
                IS = eps_greedy_probs(Q, eps_target)[s][a] / eps_greedy_probs(Q, eps_behavior)[s][a]
                W *= IS

        pi = eps_greedy_probs(Q, eps_target)
        Q_true = bellman_q(pi, gamma)
        error = np.abs(Q-Q_true).sum()

        bellman_error[current_step: total_steps] = error

    return Q, bellman_error


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
seeds = np.arange(50)

results = np.zeros((
    len(episodes_per_iteration),
    len(decays),
    len(seeds),
    max_steps,
))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
plt.ion()
plt.show()

use_is = False  # repeat with True
for ax, reward_noise_std in zip(axs, [0.0, 3.0]):
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black", "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps", fontsize=10)
    ax.set_ylabel("Absolute Bellman Error", fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.minorticks_on()
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
                Q, be = monte_carlo(env, Q, gamma, decay / max_steps, max_steps, episodes, use_is, int(seed))
                results[j, k, seed] = be
            error_shade_plot(
                ax,
                results[j, k],
                stepsize=1,
                label=f"Episodes: {episodes}, Decay: {decay}",
            )
            ax.legend()
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)

plt.savefig("MC_with_IS.png", dpi=300)
plt.ioff()
plt.show()