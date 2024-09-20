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


def eps_greedy_probs(Q, eps):
    # return action probabilities
    pi = np.ones_like(Q) * (eps / n_actions)
    best_actions = np.argmax(Q, axis=1)
    for s in range(Q.shape[0]):
        pi[s, best_actions[s]] += (1 - eps)
    return pi


def eps_greedy_probs_s(Q, s, eps):
    p_s = np.ones(n_actions) * eps / n_actions
    best_actions = np.where(Q[s] == np.max(Q[s]))[0]
    best_action = np.random.choice(best_actions) 
    
    p_s[best_action] += 1.0 - eps
    
    return p_s

def eps_greedy_action(Q, s, eps):
    # return action drawn according to eps-greedy policy
    if np.random.rand() < eps:
        action = np.random.choice(n_actions)
    else:
        action = np.random.choice(np.where(Q[s] == np.max(Q[s]))[0])
    return action


def expected_return(env, Q, gamma, episodes=10):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            a = eps_greedy_action(Q, s, 0.0)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()


def td(env, env_eval, Q, gamma, eps, alpha, max_steps, alg):
    be = []
    exp_ret = []
    tde = np.zeros(max_steps)
    Q1, Q2 = Q, Q
    # eps_decay = eps / max_steps
    # alpha_decay = alpha / max_steps
    tot_steps = 0
    while tot_steps < max_steps:
        s, _ = env.reset()
        done = False
        while not done and tot_steps < max_steps:
            a = eps_greedy_action(Q1+Q2, s, eps)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # TD learning with if ... else for the 3 algorithms
            # log TD error at every timestep
            if alg == "Double SARSA":
                if np.random.rand() < eps:
                    a_next = np.random.choice(n_actions)
                else:
                    a_next = np.argmax((Q1[s_next] + Q2[s_next])/2)

                if np.random.rand() > 0.5:
                    td_err = r + gamma * Q2[s_next, a_next] * (1 - done) - Q1[s, a]
                    Q1[s,a] += alpha * td_err
                else:
                    td_err = r + gamma * Q1[s_next, a_next] * (1 - done) - Q2[s, a]
                    Q2[s,a] += alpha * td_err
                
            elif alg == "Double QL":
                if np.random.rand() > 0.5:
                    a_next = eps_greedy_action(Q1, s_next, eps)
                    td_err = r + gamma * Q2[s_next, a_next] * (1 - done) - Q1[s, a]
                    Q1[s,a] += alpha * td_err
                else:
                    a_next = eps_greedy_action(Q2, s_next, eps)
                    td_err = r + gamma * Q1[s_next, a_next] * (1 - done) - Q2[s, a]
                    Q2[s,a] += alpha * td_err
                    
            elif alg == "Double Exp_SARSA":
                pi = eps_greedy_probs_s((Q1+Q2)/2, s_next, eps)
                if np.random.rand() > 0.5:
                    td_err = r + gamma * np.dot(Q2[s_next], pi) * (1 - done) - Q1[s, a]
                    Q1[s,a] += alpha * td_err
                else:
                    td_err = r + gamma * np.dot(Q1[s_next], pi) * (1 - done) - Q2[s, a]
                    Q2[s,a] += alpha * td_err

            tde[tot_steps] = abs(td_err)

            # log B error only every 100 steps
            # expected_return(env_eval, Q, gamma) only every
            if tot_steps % 100 == 0:
                if alg == "QL":
                    Q_true = bellman_q(eps_greedy_probs(Q, 0), gamma)
                else:
                    Q_true = bellman_q(eps_greedy_probs(Q, eps), gamma)
                be.append(np.mean(np.abs(Q - Q_true)))
                exp_ret.append(expected_return(env_eval, Q, gamma))

            eps = max(eps - 1 / max_steps, 0.01)
            alpha = max(alpha - 0.1 / max_steps, 0.001)

            s = s_next
            tot_steps += 1

    return Q, be, tde, exp_ret


def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span:])
    return re


def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())


gamma = 0.99
alpha = 0.1
eps = 1.0
max_steps = 10000
horizon = 10

init_values = [-10, 0.0, 10]
algs = ["Double QL", "Double SARSA", "Double Exp_SARSA"]
seeds = np.arange(50)

results_be = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps // 100,
))
results_tde = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps,
))
results_exp_ret = np.zeros((
    len(init_values),
    len(algs),
    len(seeds),
    max_steps // 100,
))

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
plt.ion()
plt.show()

reward_noise_std = 0.0  # re-run with 3.0

for ax in axs:
    ax.set_prop_cycle(
        color=["red", "green", "blue", "black",
               "orange", "cyan", "brown", "gray", "pink"]
    )
    ax.set_xlabel("Steps", fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.minorticks_on()

env = gymnasium.make(
    "Gym-Gridworlds/Penalty-3x3-v0",
    max_episode_steps=horizon,
    reward_noise_std=reward_noise_std,
)

env_eval = gymnasium.make(
    "Gym-Gridworlds/Penalty-3x3-v0",
    max_episode_steps=horizon,
)

for i, init_value in enumerate(init_values):
    for j, alg in enumerate(algs):
        for seed in seeds:
            np.random.seed(seed)
            Q = np.zeros((n_states, n_actions)) + init_value
            Q, be, tde, exp_ret = td(
                env, env_eval, Q, gamma, eps, alpha, max_steps, alg)
            results_be[i, j, seed] = be
            results_tde[i, j, seed] = tde
            results_exp_ret[i, j, seed] = exp_ret
            print(i, j, seed)
        label = f"$Q_0$: {init_value}, Alg: {alg}"
        axs[0].set_title("TD Error", fontsize=12)
        error_shade_plot(
            axs[0],
            results_tde[i, j],
            stepsize=1,
            smoothing_window=20,
            label=label,
        )
        axs[0].legend()
        axs[0].set_ylim([0, 5])
        axs[1].set_title("Bellman Error", fontsize=12)
        error_shade_plot(
            axs[1],
            results_be[i, j],
            stepsize=100,
            smoothing_window=20,
            label=label,
        )
        axs[1].legend()
        axs[1].set_ylim([0, 50])
        axs[2].set_title("Expected Return", fontsize=12)
        error_shade_plot(
            axs[2],
            results_exp_ret[i, j],
            stepsize=100,
            smoothing_window=20,
            label=label,
        )
        axs[2].legend()
        axs[2].set_ylim([-5, 1])
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

plt.savefig("Double TD(0).png", dpi=300)
plt.ioff()
plt.show()
