import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def bellman_updates(Q, pi, R, P, T, gamma, num_evaluations):
    q = Q.copy()                                    # copying initial state action value function
    P_Q = np.dot(P, np.multiply(pi, Q).sum(-1))     # P*pi*Q 
    Q = R + gamma * np.multiply(P_Q, (1-T))         # updating state-action value function

    Q_error = np.sum(np.abs(Q - q))

    num_evaluations += 1

    return Q, Q_error, num_evaluations


def policy_evaluation(Q, pi, R, P, T, gamma, theta, history, num_evaluations):
    Q_error = np.inf
    while Q_error > theta:
        Q, Q_error, num_evaluations = bellman_updates(Q, pi, R, P, T, gamma, num_evaluations)
        history.append(Q_error)

    return Q, history, num_evaluations


def policy_improvement(Q, pi, R, P, T, gamma):
    policy_stable = True
    old = pi.copy()

    max_Q = np.max(Q, axis=-1)

    pi = np.zeros((n_states, n_actions))
    
    for s in range(n_states):
        best_actions = np.where(Q[s] == max_Q[s])[0]
        chosen_action = np.random.choice(best_actions)
        pi[s, chosen_action] = 1
    
    if not np.allclose(pi, old):
        policy_stable = False

    return pi, policy_stable


def policy_iteration(Q, R, P, T, gamma, theta, history, num_evaluations):
    pi = np.ones((n_states, n_actions)) / n_actions
    policy_stable = False

    while not policy_stable:
        Q, history, num_evaluations = policy_evaluation(Q, pi, R, P, T, gamma, theta, history, num_evaluations)
        pi, policy_stable = policy_improvement(Q, pi, R, P, T, gamma)

    return Q, pi, history, num_evaluations


def value_iteration(Q, R, P, T, gamma, theta, history, num_evaluations):
    while True:
        delta = 0
        q = Q.copy() 
        P_Q = np.max(np.dot(P, Q), -1)  
        Q = R + gamma * np.multiply(P_Q, (1 - T))

        delta = max(delta, np.max(np.abs(Q - q)))
        history.append(np.max(np.abs(Q - q)))

        num_evaluations += 1

        if delta < theta:
            break

    optimal_actions = np.argmax(Q, -1)

    pi = np.zeros((n_states, n_actions))
    pi[np.arange(n_states), optimal_actions] = 1

    return Q, pi, history, num_evaluations


def generalized_policy_iteration(Q, R, P, T, gamma, theta, history, num_evaluations, eval_steps=5):
    pi = np.ones((n_states, n_actions)) / n_actions
    policy_stable = False

    while not policy_stable:
        for _ in range(eval_steps):
            Q, Q_error, num_evaluations = bellman_updates(Q, pi, R, P, T, gamma, num_evaluations)
            history.append(Q_error)

        pi, policy_stable = policy_improvement(Q, pi, R, P, T, gamma)

    return Q, pi, history, num_evaluations


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


def save_data(evals_PI, evals_VI, evals_GPI):
    data = {
        "Algorithm": ["Policy Iteration", "Value Iteration", "Generalized Policy Iteration"],
        "Mean": [np.mean(evals_PI), np.mean(evals_VI), np.mean(evals_GPI)],
        "Standard Deviation": [np.std(evals_PI), np.std(evals_VI), np.std(evals_GPI)]
    }

    df = pd.DataFrame(data)
    
    df.to_csv('Q_evaluation_summary.csv', index=False)


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

    evaluations_PI = []
    evaluations_VI = []
    evaluations_GPI = []

    for init_value in initial_values:

        # plot for V-function and convergence history
        fig1, axs1 = plt.subplots(1, 1, figsize=(15, 10))
        fig2, axs2 = plt.subplots(1, 1, figsize=(15, 10))
        fig3, axs3 = plt.subplots(1, 1, figsize=(15, 10))
        fig1.suptitle(f"State-Action Value Function $V_0$: {init_value}")
        fig2.suptitle(f"State-Action Value Function $V_0$: {init_value}")
        fig3.suptitle(f"State-Action Value Function $V_0$: {init_value}")

        for i, gamma in enumerate(gammas):
            Q_PI_init = np.full((n_states, n_actions), init_value)
            history_PI = []
            num_evaluations_PI = 0
            Q_PI, pi_learnt_PI, history_PI, num_evaluations_PI = policy_iteration(Q_PI_init, R, P, T, gamma, theta, history_PI, num_evaluations_PI)
            evaluations_PI.append(num_evaluations_PI)

            # if np.allclose(pi_learnt_PI, pi_opt):
            #     print("optimal policy found using policy iteration")
            assert np.allclose(pi_learnt_PI, pi_opt)

            Q_VI_init = np.full((n_states, n_actions), init_value)
            history_VI = []
            num_evaluations_VI = 0
            Q_VI, pi_learnt_VI, history_VI, num_evaluations_VI = value_iteration(Q_VI_init, R, P, T, gamma, theta, history_VI, num_evaluations_VI)
            evaluations_VI.append(num_evaluations_VI)
            
            # if np.allclose(pi_learnt_VI, pi_opt):
            #     print("optimal policy found using value iteration")

            np.allclose(pi_learnt_VI, pi_opt)

            Q_GPI_init = np.full((n_states, n_actions), init_value)
            history_GPI = []
            num_evaluations_GPI = 0
            Q_GPI, pi_learnt_GPI, history_GPI, num_evaluations_GPI = generalized_policy_iteration(Q_GPI_init, R, P, T, gamma, theta, history_GPI, num_evaluations_GPI)
            evaluations_GPI.append(num_evaluations_GPI)

            # if np.allclose(pi_learnt_GPI, pi_opt):
            #     print("optimal policy found using generalized policy iteration")

            assert np.allclose(pi_learnt_GPI, pi_opt)

            grid_size = int(np.sqrt(n_states))

            # # plot Q function
            # plot_Q_function(Q_PI, axs1[0][i], gamma, grid_size)
            # plot_Q_function(Q_VI, axs2[0][i], gamma, grid_size)
            # plot_Q_function(Q_GPI, axs3[0][i], gamma, grid_size)

            # plot convergence history
            axs1.plot(history_PI)
            axs1.set_title(f'Convergence history of Policy Iteration')
            axs1.set_xlabel('Iteration')
            axs1.set_ylabel('Bellman Error')

            # plot convergence history
            axs2.plot(history_VI)
            axs2.set_title(f'Convergence history of Value Iteration')
            axs2.set_xlabel('Iteration')
            axs2.set_ylabel('Bellman Error')

            # plot convergence history
            axs3.plot(history_GPI)
            axs3.set_title(f'Convergence history of Generalized Policy Iteration')
            axs3.set_xlabel('Iteration')
            axs3.set_ylabel('Bellman Error')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        fig1.savefig(f"Q_policy_iteration_V0_{init_value}.png")
        fig2.savefig(f"Q_value_iteration_V0_{init_value}.png")
        fig3.savefig(f"Q_generalized_policy_iteration_V0_{init_value}.png")

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    save_data(evaluations_PI, evaluations_VI, evaluations_GPI)
    print(evaluations_PI, evaluations_VI, evaluations_GPI)