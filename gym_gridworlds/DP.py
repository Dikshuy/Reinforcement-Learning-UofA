import gymnasium
import numpy as np
import matplotlib.pyplot as plt

def bellman_updates(V, Q, pi, R, P, T, gamma, theta):
    history = []

    for _ in range(max_iterations):
        v = V.copy()                                                            # copying initial policy
        V = np.sum(pi * (R + gamma * np.multiply(np.dot(P, V), (1-T))), axis=1) # updating state value function
        Q = R + gamma * np.multiply(np.dot(P, V), (1-T))                        # updating state-action value function

        bellman_error = np.sum(np.abs(v-V))                                     # computing the bellman error    
        history.append(bellman_error)           
        if bellman_error < theta:
            break  
    return V, Q, history

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
    policy[0,1] = 1 # down
    policy[1,2] = 1 # right
    policy[2,4] = 1 # stay
    policy[3,1] = 1 # down
    policy[4,2] = 1 # right
    policy[5,3] = 1 # up
    policy[6,2] = 1 # right
    policy[7,2] = 1 # right
    policy[8,3] = 1 # up

    return policy

def plot_v_function(V, axs, gamma, i, grid_size):
    V_grid = V.reshape((grid_size, grid_size))
    im = axs.imshow(V_grid, cmap='viridis', interpolation='none')
    axs.set_title(f'$\gamma$ = {gamma}')
    axs.set_xticks([])
    axs.set_yticks([])
    return im

def plot_q_function(Q, axs, a, gamma, i, grid_size):
    Q_grid = Q[:, a].reshape((grid_size, grid_size))
    im = axs.imshow(Q_grid, cmap='viridis', interpolation='none')
    axs.set_title(f'Action {a}, $\gamma$ = {gamma}')
    axs.set_xticks([])
    axs.set_yticks([])
    return im

if __name__ == "__main__":

    gammas = [0.01, 0.5, 0.99]
    initial_values = [-10, 0, 10]
    theta = 1e-5
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
        
        # Plot for V-function and convergence history
        fig1, axs1 = plt.subplots(2, len(gammas), figsize=(15, 10))
        fig1.suptitle(f"State-Value Function $V_0$: {init_value}")

        for i, gamma in enumerate(gammas):
            V = np.full(n_states, init_value)
            Q = np.full((n_states, n_actions), init_value)
            
            V, Q, history = bellman_updates(V, Q, policy, R, P, T, gamma, theta)
            store_q_values.append(Q)

            grid_size = int(np.sqrt(len(V)))
            
            # Plot V function
            im_v = plot_v_function(V, axs1[0][i], gamma, i, grid_size)
            fig1.colorbar(im_v, ax=axs1[0][i])

            # Plot convergence history
            axs1[1][i].plot(history)
            axs1[1][i].set_title(f'Convergence History, $\gamma$ = {gamma}')
            axs1[1][i].set_xlabel('Iteration')
            axs1[1][i].set_ylabel('Delta')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename_v = f'plots_v0_{init_value}.png'
        plt.savefig(filename_v)
        plt.close(fig1)

        # Plot for Q-function
        fig2, axs2 = plt.subplots(n_actions, len(gammas), figsize=(15, 3 * n_actions))
        fig2.suptitle(f"Action-Value Function $Q_0$: {init_value}")

        for i, gamma in enumerate(gammas):
            Q = store_q_values[i]

            # Plot Q-values for each action
            for a in range(n_actions):
                im_q = plot_q_function(Q, axs2[a][i], a, gamma, i, grid_size)
                fig2.colorbar(im_q, ax=axs2[a][i])

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