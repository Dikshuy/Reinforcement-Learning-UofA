import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def bellman_updates(V, Q, pi, R, P, T, gamma):
    v = V.copy()                                   # copying initial state value function
    q = Q.copy()                                   # copying initial state action value function
    P_V = np.multiply(np.dot(P, V), (1-T))         # P*V and removing the terminal state
    V = np.sum(pi * (R + gamma * P_V),axis=1)      # updating state value function
    Q = R + gamma * P_V                            # updating state-action value function

    V_error = np.sum(np.abs(V - v))
    Q_error = np.sum(np.abs(Q - q))

    return V, Q, V_error, Q_error


def policy_evaluation(V, Q, pi, R, P, T, gamma, theta, history):
    Q_error = np.inf
    while Q_error > theta:
        V, Q, V_error, Q_error = bellman_updates(V, Q, pi, R, P, T, gamma)
        history.append(Q_error)

    return V, Q, history


def greedy_policy(V, R, P, T, gamma):
    Q = R + gamma * np.multiply(np.dot(P, V), (1-T))
    best = np.argmax(Q, axis=1)
    pi = np.eye(n_actions)[best]
    
    return pi


def policy_improvement(V, pi, R, P, T, gamma):
    policy_stable = True
    old = pi.copy()
    pi = greedy_policy(V, R, P, T, gamma)
    
    if not np.allclose(pi, old):
        policy_stable = False

    return pi, policy_stable


def policy_iteration(V, Q, R, P, T, gamma, theta, history):
    pi = np.ones((n_states, n_actions)) / n_actions
    policy_stable = False

    while not policy_stable:
        V, Q, history = policy_evaluation(V, Q, pi, R, P, T, gamma, theta, history)
        pi, policy_stable = policy_improvement(V, pi, R, P, T, gamma)

    return V, Q, pi, history


def optimality_update(s, V, R, P, T, gamma):
    P_V = np.multiply(np.dot(P[s], V), (1-T[s]))
    V_s = R[s] + gamma * P_V
    return V_s


def value_iteration(V, Q, R, P, T, gamma, theta, history):
    while True:
        delta = 0
        v = V.copy()
        q = Q.copy()      
        Q = R + gamma * np.multiply(np.dot(P, V), (1 - T))
        V = np.max(Q, axis=1)

        delta = max(delta, np.max(np.abs(Q - q)))
        history.append(np.max(np.abs(Q - q)))

        if delta < theta:
            break

    pi = np.ones((n_states, n_actions)) / n_actions

    for s in range(n_states):
        V_s = optimality_update(s, V, R, P, T, gamma)
        best = np.argmax(V_s)
        pi[s] = np.eye(n_actions)[best]

    return V, Q, pi, history


def generalized_policy_iteration(V, Q, R, P, T, gamma, theta, history, eval_steps=5):
    pi = np.ones((n_states, n_actions)) / n_actions
    policy_stable = False

    while not policy_stable:
        for _ in range(eval_steps):
            V, Q, V_error, Q_error = bellman_updates(V, Q, pi, R, P, T, gamma)
            history.append(Q_error)

        pi, policy_stable = policy_improvement(V, pi, R, P, T, gamma)

    return V, Q, pi, history


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


def plot_Q_function(Q, axs, gamma, grid_size):
    Q_grid = np.max(Q, axis=1).reshape((grid_size, grid_size))
    sns.heatmap(Q_grid, annot=True, cmap='viridis', ax=axs, cbar=True)
    axs.set_title(f'$\gamma$ = {gamma}')
    axs.set_xticks([])
    axs.set_yticks([])


if __name__ == "__main__":

    gammas = [0.99]
    initial_values = [-100, -10, -5, 0, 5, 10, 100]
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
        fig3, axs3 = plt.subplots(2, len(gammas), figsize=(15, 10))
        fig1.suptitle(f"State-Action Value Function $V_0$: {init_value}")
        fig2.suptitle(f"State-Action Value Function $V_0$: {init_value}")
        fig3.suptitle(f"State-Action Value Function $V_0$: {init_value}")

        for i, gamma in enumerate(gammas):
            V_PI_init = np.full(n_states, init_value)
            Q_PI_init = np.full((n_states, n_actions), init_value)
            history_PI = []
            V_PI, Q_PI, pi_learnt_PI, history_PI = policy_iteration(V_PI_init, Q_PI_init, R, P, T, gamma, theta, history_PI)

            # if np.allclose(pi_learnt_PI, pi_opt):
            #     print("optimal policy found using policy iteration")

            assert np.allclose(pi_learnt_PI, pi_opt)

            V_VI_init = np.full(n_states, init_value)
            Q_VI_init = np.full((n_states, n_actions), init_value)
            history_VI = []
            V_VI, Q_VI, pi_learnt_VI, history_VI = value_iteration(V_VI_init, Q_VI_init, R, P, T, gamma, theta, history_VI)

            # if np.allclose(pi_learnt_VI, pi_opt):
            #     print("optimal policy found using value iteration")

            np.allclose(pi_learnt_VI, pi_opt)

            V_GPI_init = np.full(n_states, init_value)
            Q_GPI_init = np.full((n_states, n_actions), init_value)
            history_GPI = []
            V_GPI, Q_GPI, pi_learnt_GPI, history_GPI = generalized_policy_iteration(V_GPI_init, Q_GPI_init, R, P, T, gamma, theta, history_GPI)

            if np.allclose(pi_learnt_GPI, pi_opt):
                print(init_value)
                print("optimal policy found using generalized policy iteration")

            assert np.allclose(pi_learnt_GPI, pi_opt)

            grid_size = int(np.sqrt(n_states))

            axs1 = np.expand_dims(axs1, axis=1)
            axs2 = np.expand_dims(axs2, axis=1)
            axs3 = np.expand_dims(axs3, axis=1)

            # plot V function
            plot_Q_function(Q_PI, axs1[0][i], gamma, grid_size)
            plot_Q_function(Q_VI, axs2[0][i], gamma, grid_size)
            plot_Q_function(Q_GPI, axs3[0][i], gamma, grid_size)

            # plot convergence history
            axs1[1][i].plot(history_PI)
            axs1[1][i].set_title(f'Convergence history of Policy Iteration , $\gamma$ = {gamma}')
            axs1[1][i].set_xlabel('Iteration')
            axs1[1][i].set_ylabel('Bellman Error')

            # plot convergence history
            axs2[1][i].plot(history_VI)
            axs2[1][i].set_title(f'Convergence history of Value Iteration, $\gamma$ = {gamma}')
            axs2[1][i].set_xlabel('Iteration')
            axs2[1][i].set_ylabel('Bellman Error')

            # plot convergence history
            axs3[1][i].plot(history_GPI)
            axs3[1][i].set_title(f'Convergence history of Generalized Policy Iteration, $\gamma$ = {gamma}')
            axs3[1][i].set_xlabel('Iteration')
            axs3[1][i].set_ylabel('Bellman Error')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        fig1.savefig(f"Q_policy_iteration_V0_{init_value}.png")
        fig2.savefig(f"Q_value_iteration_V0_{init_value}.png")
        fig3.savefig(f"Q_generalized_policy_iteration_V0_{init_value}.png")

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)