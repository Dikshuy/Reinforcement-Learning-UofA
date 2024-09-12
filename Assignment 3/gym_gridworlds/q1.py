import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def bellman_updates(V, pi, R, P, T, gamma):
    v = V.copy()                                    # copying initial policy
    P_V = np.multiply(np.dot(P, V), (1-T))         # pi*V and removing the terminal state
    V = np.sum(pi * (R + gamma * P_V),axis=1)      # updating state value function

    bellman_error = np.sum(np.abs(V - v))

    return V, bellman_error


def policy_evaluation(V, pi, R, P, T, gamma, theta, history):
    bellman_error = 1
    while True:
        V, bellman_error = bellman_updates(V, pi, R, P, T, gamma)
        history.append(bellman_error)
        if bellman_error < theta:
            break
    return V, history


def greedy_policy(s, V, pi, R, P, T, gamma):
    Q_s = R[s] + gamma * np.multiply(np.dot(P[s], V), (1-T[s]))
    best = np.argmax(Q_s)
    pi[s] = np.eye(n_actions)[best]


def policy_improvement(V, pi, R, P, T, gamma):
    policy_stable = True
    for s in range(n_states):
        old = pi[s].copy()
        greedy_policy(s, V, pi, R, P, T, gamma)
        if not np.array_equal(pi[s], old):
            policy_stable = False

    return pi, policy_stable


def policy_iteration(V, R, P, T, gamma, theta, history):
    pi = np.ones((n_states, n_actions)) / n_actions
    policy_stable = False

    while not policy_stable:
        V, history = policy_evaluation(V, pi, R, P, T, gamma, theta, history)
        pi, policy_stable = policy_improvement(V, pi, R, P, T, gamma)

    return V, pi, history


def optimality_update(s, V, R, P, T, gamma):
    P_V = np.multiply(np.dot(P[s], V), (1-T[s]))
    V_s = R[s] + gamma * P_V
    return V_s


def value_iteration(V, R, P, T, gamma, theta, history):
    while True:
        delta = 0
        for s in range(n_states):
            v = V.copy()
            V_s = optimality_update(s, V, R, P, T, gamma)
            V[s] = np.max(V_s)
            delta = max(delta, np.abs(v[s]- V[s]))
        history.append(delta)
        if delta < theta:
                break
        
    pi = np.ones((n_states, n_actions)) / n_actions

    for s in range(n_states):
        V_s = optimality_update(s, V, R, P, T, gamma)
        best = np.argmax(V_s)
        pi[s] = np.eye(n_actions)[best]
    
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
        fig2, axs2 = plt.subplots(2, len(gammas), figsize=(15, 10))
        fig1.suptitle(f"State-Value Function $V_0$: {init_value}")
        fig2.suptitle(f"State-Value Function $V_0$: {init_value}")

        for i, gamma in enumerate(gammas):
            V = np.full(n_states, init_value)
            history_PI = []
            V, pi_learnt_PI, history_PI = policy_iteration(V, R, P, T, gamma, theta, history_PI)
            
            if np.all(pi_learnt_PI == pi_opt):
                print("optimal policy found using policy iteration")

            V = np.full(n_states, init_value)
            history_VI = []
            V, pi_learnt_VI, history_VI = value_iteration(V, R, P, T, gamma, theta, history_VI)
            
            if np.all(pi_learnt_VI == pi_opt):
                print("optimal policy found using value iteration")

            grid_size = int(np.sqrt(n_states))

            axs1 = np.expand_dims(axs1, axis=1)
            axs2 = np.expand_dims(axs2, axis=1)

            # plot V function
            plot_v_function(V, axs1[0][i], gamma, grid_size)
            plot_v_function(V, axs2[0][i], gamma, grid_size)

            # plot convergence history
            axs1[1][i].plot(history_PI)
            axs1[1][i].set_title(f'Convergence History, $\gamma$ = {gamma}')
            axs1[1][i].set_xlabel('Iteration')
            axs1[1][i].set_ylabel('Bellman Error')

            # plot convergence history
            axs2[1][i].plot(history_VI)
            axs2[1][i].set_title(f'Convergence History, $\gamma$ = {gamma}')
            axs2[1][i].set_xlabel('Iteration')
            axs2[1][i].set_ylabel('Bellman Error')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        # filename_v = f'plots_v0_{init_value}.png'
        # plt.savefig(filename_v)
        # plt.close(fig1)
