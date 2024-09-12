import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def policy_evaluation(V, pi, R, P, T, gamma, theta, history):
    bellman_error = np.inf
    while bellman_error > theta:
        V, bellman_error = bellman_updates(V, pi, R, P, T, gamma)
        history.append(bellman_error)
    return V, history


def bellman_updates(V, pi, R, P, T, gamma):
    v = V.copy()                                    # copying initial policy
    pi_V = np.multiply(np.dot(P, V), (1-T))         # pi*V and removing the terminal state
    V = np.sum(pi * (R + gamma * pi_V),axis=1)      # updating state value function

    bellman_error = np.sum(np.abs(V - v))

    return V, bellman_error


def policy_improvement(V, pi, R, P, T, gamma):
    policy_stable = True
    for s in range(n_states):
        old = pi[s].copy()
        greedy_policy(s, V, pi, R, P, T, gamma)
        if not np.array_equal(pi[s], old):
            policy_stable = False

    return pi, policy_stable

def greedy_policy(s, V, pi, R, P, T, gamma):
    Q_s = R[s] + gamma * np.multiply(np.dot(P[s], V), (1-T[s]))
    best = np.argmax(Q_s)
    pi[s] = np.eye(n_actions)[best]


def policy_iteration(V, R, P, T, gamma, theta, history):
    pi = np.ones((n_states, n_actions)) / n_actions
    policy_stable = False

    while not policy_stable:
        V, history = policy_evaluation(V, pi, R, P, T, gamma, theta, history)
        pi, policy_stable = policy_improvement(V, pi, R, P, T, gamma)

    return V, pi, history


def optimal_policy():
    '''
     |----------------------|
     |   |   |      |       |
     |   v   |  ->  |   o   |
     |----------------------|
     |   |   |      |   ^   |
     |   v   |  ->  |   |   |
     |----------------------|
     |       |      |   ^   |
     |   ->  |  ->  |   |   |
     |----------------------|
    '''
    policy = np.zeros((n_states, n_actions))
    policy[0, 1] = 1  # down
    policy[1, 2] = 1  # right
    policy[2, 4] = 1  # stay
    policy[3, 1] = 1  # down
    policy[4, 2] = 1  # right
    policy[5, 3] = 1  # up
    policy[6, 2] = 1  # right
    policy[7, 2] = 1  # right
    policy[8, 3] = 1  # up

    return policy


def plot_v_function(V, axs, gamma, grid_size):
    V_grid = V.reshape((grid_size, grid_size))
    sns.heatmap(V_grid, annot=True, cmap='viridis', ax=axs, cbar=True)
    axs.set_title(f'$\gamma$ = {gamma}')
    axs.set_xticks([])
    axs.set_yticks([])


def plot_q_function(Q, axs, a, gamma, grid_size):
    action_labels = ['Left', 'Down', 'Right', 'Up', 'Stay']
    Q_grid = Q[:, a].reshape((grid_size, grid_size))
    sns.heatmap(Q_grid, annot=True, cmap='viridis', ax=axs, cbar=True)
    axs.set_title(f'action: {action_labels[a]}, $\gamma$ = {gamma}')
    axs.set_xticks([])
    axs.set_yticks([])


if __name__ == "__main__":

    gammas = [0.99]
    initial_values = [-10, 0, 10]
    theta = 1e-10
    max_iterations = 10000

    env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
    env.reset()

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    R = np.zeros((n_states, n_actions))
    P = np.zeros((n_states, n_actions, n_states))
    T = np.zeros((n_states, n_actions))

    # print("Reward Matrix (R):\n", R)
    # print("Transition Matrix (P):\n", P)
    # print("Termination Matrix (T):\n", T)

    for s in range(n_states):
        for a in range(n_actions):
            env.set_state(s)
            s_next, r, terminated, _, _ = env.step(a)
            R[s, a] = r
            P[s, a, s_next] = 1.0
            T[s, a] = terminated

    pi_opt = optimal_policy()

    for init_value in initial_values:

        # plot for V-function and convergence history
        fig1, axs1 = plt.subplots(2, len(gammas), figsize=(15, 10))
        fig1.suptitle(f"State-Value Function $V_0$: {init_value}")

        for i, gamma in enumerate(gammas):
            V = np.full(n_states, init_value)
            history = []
            V, pi_learnt, history = policy_iteration(V, R, P, T, gamma, theta, history)

            assert np.allclose(pi_learnt, pi_opt)

            grid_size = int(np.sqrt(len(V)))

            # plot V function
            plot_v_function(V, axs1[0][i], gamma, grid_size)

            # plot convergence history
            axs1[1][i].plot(history)
            axs1[1][i].set_title(f'Convergence History, $\gamma$ = {gamma}')
            axs1[1][i].set_xlabel('Iteration')
            axs1[1][i].set_ylabel('Bellman Error')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename_v = f'plots_v0_{init_value}.png'
        plt.savefig(filename_v)
        plt.close(fig1)
