import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
env.reset()

gammas = [0.01, 0.5, 0.99]
initial_values = [-10, 0, 10]
theta = 1e-50
max_iterations = 10000

n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

for s in range(n_states):
    for a in range(n_actions):
        env.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

# print("Reward Matrix (R):\n", R)
# print("Transition Matrix (P):\n", P)
# print("Termination Matrix (T):\n", T)

def evaluate_V_policy(V, pi, R, P, terminal_state,gamma, theta):
    history = []
    for _ in range(max_iterations):
        bellman_error = 0
        v = V.copy()
        V = np.sum(pi * (R + gamma * np.matmul(P, V)), axis=1)
        V[terminal_state] = R[terminal_state, policy[terminal_state].argmax()]
        bellman_error = np.sum(np.abs(v-V))
        history.append(bellman_error) 
        if bellman_error < theta:
            break  
    return V, history

def evaluate_Q_policy(Q, pi, R, P, terminal_state, gamma, theta):
    history = []
    for _ in range(max_iterations):
        delta = 0
        total_bellman_error = 0  
        
        for s in range(n_states):
            if s == terminal_state:
                continue
            for a in range(n_actions):
                q = Q[s, a]
                Q[s, a] = R[s, a] + gamma * np.sum([P[s, a, s_next] * np.dot(pi[s_next], Q[s_next])
                                                    for s_next in range(n_states)])
                
                bellman_error = abs(q - Q[s, a])
                total_bellman_error += bellman_error
                delta = max(delta, bellman_error)

        history.append(total_bellman_error) 
        if delta < theta:
            break

    return Q, history

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

results = {}

terminal_state = 2

for gamma in gammas:
    if gamma not in results:
        results[gamma] = {}
    for init_value in initial_values:
        V_initial = np.full(n_states, init_value)
        Q_initial = np.full((n_states, n_actions), init_value)

        policy = optimal_policy()

        V_pi, history_v = evaluate_V_policy(V_initial, policy, R, P, terminal_state, gamma, theta)
        Q_pi, history_q = evaluate_Q_policy(Q_initial, policy, R, P, terminal_state, gamma, theta)

        results[gamma][init_value] = {
            'V_pi': V_pi,
            'Q_pi': Q_pi,
            'delta_history_v': history_v,
            'delta_history_q': history_q
        }

def plot_heatmap(data, title, ax):
    sns.heatmap(data.reshape(3, 3), annot=True, fmt=".2f", cbar=False, ax=ax)
    ax.set_title(title)

def plot_bellman_error(delta_history, title, ax):
    ax.plot(delta_history)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Bellman Error")

# fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# fig2, axs2 = plt.subplots(3, 3, figsize=(15, 15))

# for i, gamma in enumerate(gammas):
#     for j, init_val in enumerate(initial_values):
#         data = results[gamma][init_val]
#         plot_heatmap(data['V_pi'], f"V_pi (gamma={gamma}, init_val={init_val})", axs[i, j])
#         plot_bellman_error(data['delta_history_v'], f"V_pi Bellman Error (gamma={gamma}, init_val={init_val})", axs2[i, j])

# plt.tight_layout()
# plt.show()

# # Q-function heatmaps for each action
# fig, axs = plt.subplots(3, 5, figsize=(20, 12))
# actions = ["left", "down", "right", "up", "stay"]
# data = results[0.99][0]
# plot_heatmap(data['Q_pi'][:, 1], f"Q_pi Action={actions[1]} (gamma={0.99}, init_val={0})", axs[1, 1])
# # for i, gamma in enumerate(gammas):
# #     for j, init_val in enumerate(initial_values):
# #         data = results[gamma][init_val]
# #         for a in range(n_actions):
# #             # plot_bellman_error(data['delta_history_q'], f"Q_pi Bellman Error (gamma={gamma}, init_val={init_val})", axs2[i, j])
# #             plot_heatmap(data['Q_pi'][:, a], f"Q_pi Action={actions[a]} (gamma={gamma}, init_val={init_val})", axs[i, a])

# # plt.tight_layout()
# plt.show()

gammas = [0.01, 0.5, 0.99]
for init_value in [-10, 0, 10]:
    fig, axs = plt.subplots(2, len(gammas))
    fig.suptitle(f"$V_0$: {init_value}")
    for i, gamma in enumerate(gammas):
        V, delta_history = evaluate_V_policy(V_initial, policy, R, P, terminal_state, gamma, theta)
        grid_size = int(np.sqrt(len(V))) 
        V_grid = V.reshape((grid_size, grid_size))
        
        # Plot value function
        im = axs[0][i].imshow(V_grid, cmap='viridis', interpolation='none')
        axs[0][i].set_title(f'$\gamma$ = {gamma}')
        axs[0][i].set_xticks([])
        axs[0][i].set_yticks([])
        fig.colorbar(im, ax=axs[0][i])

        # Plot convergence history
        axs[1][i].plot(delta_history)
        axs[1][i].set_title(f'$\gamma$ = {gamma}')
        axs[1][i].set_xlabel('Iteration')
        axs[1][i].set_ylabel('Delta')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    filename = f'plots_v0_{init_value}.png'
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free memory

    fig, axs = plt.subplots(n_actions + 1, len(gammas), figsize=(15, 2 * (n_actions + 1)))
    fig.suptitle(f"$Q_0$: {init_value}")

    for i, gamma in enumerate(gammas):
        # Compute Q-values and convergence history
        Q, delta_history = evaluate_Q_policy(Q_initial, policy, R, P, terminal_state, gamma, theta)
        
        # Reshape Q-values into a 2D grid for each action
        # Plot Q-values for each action
        for a in range(n_actions):
            # Reshape Q-values into a 2D grid
            Q_grid = Q[:, a].reshape((grid_size, grid_size))
            print(f"axs shape: {axs.shape}")
            print(f"Q shape: {Q.shape}")
            print(f"Q_grid shape: {Q_grid.shape}")
            print(f"axs shape: {axs.shape}")
            im = axs[a][i].imshow(Q_grid, cmap='viridis', interpolation='none')
            axs[a][i].set_title(f'Action {a}')
            axs[a][i].set_xticks([])
            axs[a][i].set_yticks([])
            fig.colorbar(im, ax=axs[a][i])

        # Plot convergence history
        axs[-1, i].plot(delta_history)
        axs[-1, i].set_title(f'$\gamma$ = {gamma}')
        axs[-1, i].set_xlabel('Iteration')
        axs[-1, i].set_ylabel('Delta')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    filename = f'plots_q0_{init_value}.png'
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free memory