import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def policy_evaluation(V, Q, pi, R, P, T, gamma, theta):
    history = []
    bellman_error = np.inf
    while bellman_error > theta:
        V, Q, bellman_error = bellman_updates(V, Q, pi, R, P, T, gamma)
        history.append(bellman_error)
    return V, Q, history


def bellman_updates(V, Q, pi, R, P, T, gamma):
    v = V.copy()                                    # copying initial policy
    pi_V = np.multiply(np.dot(P, V), (1-T))         # pi*V and removing the terminal state
    V = np.sum(pi * (R + gamma * pi_V),axis=1)      # updating state value function
    Q = R + gamma * pi_V                            # updating state-action value function

    bellman_error = np.sum(np.abs(V - v))

    return V, Q, bellman_error


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

    gammas = [0.01, 0.5, 0.99]
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

    policy = optimal_policy()

    for init_value in initial_values:
        store_q_values = []

        # plot for V-function and convergence history
        fig1, axs1 = plt.subplots(2, len(gammas), figsize=(15, 10))
        fig1.suptitle(f"State-Value Function $V_0$: {init_value}")

        for i, gamma in enumerate(gammas):
            V = np.full(n_states, init_value)
            Q = np.full((n_states, n_actions), init_value)
            V, Q, history = policy_evaluation(V, Q, policy, R, P, T, gamma, theta)
  
            store_q_values.append(Q)

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

        # plot for Q-function
        fig2, axs2 = plt.subplots(n_actions, len(gammas), figsize=(15, 3 * n_actions))
        fig2.suptitle(f"Action-Value Function $Q_0$: {init_value}")

        for i, gamma in enumerate(gammas):
            Q = store_q_values[i]

            # plot Q-values for each action
            for a in range(n_actions):
                plot_q_function(Q, axs2[a][i], a, gamma, grid_size)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename_q = f'plots_q0_{init_value}.png'
        plt.savefig(filename_q)
        plt.close(fig2)


'''
-- old --

------ MAIN UPDATE LOOP ------

def evaluate_V_policy(V, pi, R, P, T,gamma, theta):
    history = []
    for _ in range(max_iterations):
        bellman_error = 0
        v = V.copy()
        V = np.sum(pi * (R + gamma * np.multiply(np.dot(P, V), (1-T))), axis=1)
        bellman_error = np.sum(np.abs(v-V))
        history.append(bellman_error) 
        if bellman_error < theta:
            break  
    return V, history

results = {}

for gamma in gammas:
    if gamma not in results:
        results[gamma] = {}
    for init_value in initial_values:
        V_initial = np.full(n_states, init_value)

        policy = optimal_policy()

        V_pi, history = evaluate_V_policy(V_initial, policy, R, P, terminal_state, gamma, theta)

        results[gamma][init_value] = {
            'V_pi': V_pi,
            'delta_history': history,
        }
'''
