import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
env.reset()

gammas = [0.01, 0.5, 0.99]
initial_values = [-10, 0, 10]
theta = 1e-5
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

def bellman_updates(V, Q, pi, R, P, terminal_state, gamma, theta):
    history = []
    for _ in range(max_iterations):
        bellman_error = 0
        v = V.copy()
        V = np.sum(pi * (R + gamma * np.matmul(P, V)), axis=1)
        # Q = R + gamma * 


def evaluate_V_policy(V, pi, R, P, T,gamma, theta):
    history = []
    for _ in range(max_iterations):
        bellman_error = 0
        v = V.copy()
        V = np.sum(pi * (R + gamma * np.multiply(np.dot(P, V), 1-T)), axis=1)
        # V[terminal_state] = R[terminal_state, pi[terminal_state].argmax()]
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

# for gamma in gammas:
#     if gamma not in results:
#         results[gamma] = {}
#     for init_value in initial_values:
#         V_initial = np.full(n_states, init_value)
#         Q_initial = np.full((n_states, n_actions), init_value)

#         policy = optimal_policy()

#         V_pi, history_v = evaluate_V_policy(V_initial, policy, R, P, terminal_state, gamma, theta)
#         Q_pi, history_q = evaluate_Q_policy(Q_initial, policy, R, P, terminal_state, gamma, theta)

#         results[gamma][init_value] = {
#             'V_pi': V_pi,
#             'Q_pi': Q_pi,
#             'delta_history_v': history_v,
#             'delta_history_q': history_q
#         }

def plot_heatmap(data, title, ax):
    sns.heatmap(data.reshape(3, 3), annot=True, fmt=".2f", cbar=False, ax=ax)
    ax.set_title(title)

def plot_bellman_error(delta_history, title, ax):
    ax.plot(delta_history)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Bellman Error")

gammas = [0.01, 0.5, 0.99]
policy = optimal_policy()
for init_value in [-10, 0, 10]:
    fig, axs = plt.subplots(2, len(gammas))
    fig.suptitle(f"$V_0$: {init_value}")
    for i, gamma in enumerate(gammas):
        V, delta_history = evaluate_V_policy(np.full(n_states, init_value), policy, R, P, terminal_state, gamma, theta)
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
    
    filename = f'plots_v0_{init_value}.png'
    plt.savefig(filename)
    plt.close(fig)

    fig, axs = plt.subplots(n_actions + 1, len(gammas), figsize=(15, 2 * (n_actions + 1)))
    fig.suptitle(f"$Q_0$: {init_value}")

    for i, gamma in enumerate(gammas):
        Q, delta_history = evaluate_Q_policy(np.full((n_states, n_actions), init_value), policy, R, P, terminal_state, gamma, theta)
        # Plot Q-values for each action
        for a in range(n_actions):
            # Reshape Q-values into a 2D grid
            Q_grid = Q[:, a].reshape((grid_size, grid_size))
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

    filename = f'plots_q0_{init_value}.png'
    plt.savefig(filename)
    plt.close(fig)