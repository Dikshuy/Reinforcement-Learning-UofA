import gymnasium
# import gym_gridworlds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x, y):
    return int(0.5 * (x + y) * (x + y + 1) + y)

def rbf_features(x: np.array, c: np.array, s: np.array) -> np.array:
    return np.exp(-(((x[:, None] - c[None]) / s[None])**2).sum(-1) / 2.0)

def expected_return(env, weights, gamma, episodes=100):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            # a = np.dot(phi, weights)
            # a_clip = np.clip(a, env.action_space.low, env.action_space.high)  # this is for the Pendulum
            a = eps_greedy_action(phi, weights, 0)  # this is for the Gridworld
            # s_next, r, terminated, truncated, _ = env.step(a_clip)  # replace with a for Pendulum
            s_next, r, terminated, truncated, _ = env.step(a)  # replace with a for Gridworld
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()

def collect_data(env, weights, n_episodes, var):
    data = dict()
    data["phi"] = []
    data["a"] = []
    data["r"] = []
    data["done"] = []
    for ep in range(n_episodes):
        episode_seed = cantor_pairing(ep, seed)
        s, _ = env.reset(seed=episode_seed)
        done = False
        while not done:
            phi = get_phi(s)
            # a = gaussian_action(phi, weights, var)
            a = softmax_action(phi, weights, var)
            # a_clip = np.clip(a, env.action_space.low, env.action_space.high)  # only for Gaussian policy
            # s_next, r, terminated, truncated, _ = env.step(a_clip)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            data["phi"].append(phi)
            data["a"].append(a)
            data["r"].append(r)
            data["done"].append(terminated or truncated)
            s = s_next
    return data

def eps_greedy_action(phi, weights, eps):
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    else:
        Q = np.dot(phi, weights).ravel()
        best = np.argwhere(Q == Q.max())
        i = np.random.choice(range(best.shape[0]))
        return best[i][0]

def softmax_probs(phi, weights, eps):
    q = np.dot(phi, weights)
    # this is a trick to make it more stable
    # see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    q_exp = np.exp((q - np.max(q, -1, keepdims=True)) / max(eps, 1e-12))
    probs = q_exp / q_exp.sum(-1, keepdims=True)
    return probs

def softmax_action(phi, weights, eps):
    probs = softmax_probs(phi, weights, eps)
    return np.random.choice(weights.shape[1], p=probs.ravel())

def dlog_softmax_probs(phi, weights, eps, act):
    # implement log-derivative of pi
    probs = softmax_probs(phi, weights, eps)
    dlog_pi = phi[:, np.newaxis] * (np.eye(n_actions)[act] - probs)
    return dlog_pi

def gaussian_action(phi, weights, sigma: np.array):
    mu = np.dot(phi, weights)
    return np.random.normal(mu, sigma**2)

def dlog_gaussian_probs(phi, weights, sigma, action: np.array):
    # implement log-derivative of pi with respect to the mean only
    mu = np.dot(phi, weights)
    return phi * (action - mu) / (sigma**2)

def compute_mc_returns(rewards, gamma, dones):
    G = np.zeros_like(rewards)
    g = 0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            g = 0
        g = rewards[t] + gamma * g
        G[t] = g
    return G

def reinforce(baseline="none"):
    # weights = np.zeros((phi_dummy.shape[1], action_dim))  # PENDULUM
    weights = np.zeros((phi_dummy.shape[1], n_actions))     # GRIDWORLD
    sigma = 2.0  # for Gaussian
    eps = 1.0  # softmax temperature, DO NOT DECAY
    tot_steps = 0
    exp_return_history = np.zeros(max_steps)
    exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
    pbar = tqdm(total=max_steps)

    while tot_steps < max_steps:
        # collect data
        # data = collect_data(env, weights, episodes_per_update, sigma)   # PENDULUM
        data = collect_data(env, weights, episodes_per_update, eps)   # GRIDWORLD
        phi = np.vstack(data["phi"])
        a = np.vstack(data["a"])
        r = np.vstack(data["r"])
        done = np.vstack(data["done"])

        # compute MC return
        G = compute_mc_returns(r, gamma, done)

        # compute gradient of all samples (with/without baseline) (PENDULUM)
        # dlog_pi = dlog_gaussian_probs(phi, weights, sigma, a)

        # compute gradient of all samples (with/without baseline)  (GRIDWORLD)
        dlog_pi = np.array([dlog_softmax_probs(phi[t], weights, eps, a[t]) for t in range(len(a))])

        if baseline == "none":
            b = 0
        elif baseline == "mean_return":
            b = np.mean(G)
        else:
            # b = np.sum(np.sum(dlog_pi**2, axis=1) * G) / np.sum(np.sum(dlog_pi**2, axis=1))
            b = np.sum((dlog_pi**2) * G) / np.sum(dlog_pi**2)           # CHECK WHETHER THIS IS CORRECT OR NOT

        # average gradient over all samples
        gradient = np.zeros_like(weights)
        for t in range(len(G)):
            gradient += dlog_pi[t].reshape(-1, 1) * (G[t] - b)
        gradient /= len(G)

        # update weights
        weights += alpha * gradient

        T = len(G) # steps taken while collecting data
        exp_return_history[tot_steps : tot_steps + T] = exp_return
        tot_steps += T
        exp_return = expected_return(env_eval, weights, gamma, episodes_eval)
        sigma = max(sigma - T / max_steps, 0.1)

        pbar.set_description(
            f"G: {exp_return:.3f}"
        )
        pbar.update(T)

    pbar.close()
    return exp_return_history


# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
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


# UNCOMMENT TO SOLVE THE PENDULUM
# env_id = "Pendulum-v1"
# env = gymnasium.make(env_id)
# env_eval = gymnasium.make(env_id)
# episodes_eval = 100
# # you'll solve the Pendulum when the empirical expected return is higher than -150
# # but it can get even higher, eg -120
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]

# UNCOMMENT TO SOLVE THE GRIDWORLD
env_id = "Gym-Gridworlds/Penalty-3x3-v0"
env = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=10000)
env_eval = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=10)  # 10 steps only for faster eval
episodes_eval = 1  # max expected return will be 0.941
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# automatically set centers and sigmas
n_centers = [7] * state_dim
state_low = env.observation_space.low
state_high = env.observation_space.high
centers = np.array(
    np.meshgrid(*[
        np.linspace(
            state_low[i] - (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
            state_high[i] + (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
            n_centers[i],
        )
        for i in range(state_dim)
    ])
).reshape(state_dim, -1).T
sigmas = (state_high - state_low) / np.asarray(n_centers) * 0.75 + 1e-8  # change sigmas for more/less generalization
get_phi = lambda state : rbf_features(state.reshape(-1, state_dim), centers, sigmas)  # reshape because feature functions expect shape (N, S)
phi_dummy = get_phi(env.reset()[0])  # to get the number of features

# hyperparameters
gamma = 0.99
alpha = 0.1
episodes_per_update = 10
# max_steps = 1000000  # 1000000 for the Pendulum
max_steps = 100000  # 100000 for the Gridworld
baselines = ["none", "mean_return", "min_variance"]
n_seeds = 10
results_exp_ret = np.zeros((
    len(baselines),
    n_seeds,
    max_steps,
))

fig, axs = plt.subplots(1, 1)
axs.set_prop_cycle(color=["red", "green", "blue"])
axs.set_xlabel("Steps")
axs.set_ylabel("Expected Return")

for i, baseline in enumerate(baselines):
    for seed in range(n_seeds):
        np.random.seed(seed)
        exp_return_history = reinforce(baseline)
        results_exp_ret[i, seed] = exp_return_history
        print(baseline, seed)

    plot_args = dict(
        stepsize=1,
        smoothing_window=20,
        label=baseline,
    )
    error_shade_plot(
        axs,
        results_exp_ret[i],
        **plot_args,
    )
    axs.legend()

plt.show()