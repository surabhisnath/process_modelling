import numpy as np
from scipy.optimize import minimize

class Bandits:
    def __init__(self, num_clusters, cluster_sequence, inverse_prob):
        self.num_arms = num_clusters
        self.arms = np.arange(self.num_arms)
        self.q = np.zeros(self.num_arms)
        self.arms_sampled = cluster_sequence
        self.rewards = inverse_prob

    def epsilon_greedy(epsilon, alpha):
        likelihood = 0
        for arm, reward in zip(self.arms_sampled, self.rewards):
            # Calculate probabilities
            if np.random.rand() < epsilon:
                chosen_arm = np.random.choice(self.arms)  # Exploration
            else:
                chosen_arm = np.argmax(self.q)

            # Update Q-values using alpha (learning rate)
            self.q[arm] += alpha * (reward - self.q[arm])
            
            # Accumulate log likelihood of choosing observed arms
            prob = epsilon / num_arms + (1 - epsilon) * (chosen_arm == arm)
            likelihood += np.log(prob + 1e-9)  # Avoid log(0)

        return -likelihood  # Negative log-likelihood


    def optimise():

           
# Data
arms_sampled = [3, 4, 4, 1, 5, 3, 2]
rewards_obtained = [1, 2, 2, 0, 0, 1, 6]
num_arms = max(arms_sampled)

# Define the function to simulate the epsilon-greedy learning
def simulate_sequence(epsilon, alpha, arms_sampled, rewards_obtained, num_arms):
    Q_values = np.zeros(num_arms + 1)  # Q-values for each arm (index 0 unused)
    likelihood = 0  # Track log-likelihood
    
    for arm, reward in zip(arms_sampled, rewards_obtained):
        # Calculate probabilities
        if np.random.rand() < epsilon:
            chosen_arm = np.random.choice(range(1, num_arms + 1))  # Exploration
        else:
            chosen_arm = np.argmax(Q_values[1:]) + 1  # Exploitation
            
        # Update Q-values using alpha (learning rate)
        Q_values[arm] += alpha * (reward - Q_values[arm])
        
        # Accumulate log likelihood of choosing observed arms
        prob = epsilon / num_arms + (1 - epsilon) * (chosen_arm == arm)
        likelihood += np.log(prob + 1e-9)  # Avoid log(0)

    return -likelihood  # Negative log-likelihood

# Fit epsilon and alpha using optimization
def fit_parameters(arms_sampled, rewards_obtained, num_arms):
    # Initial guesses for epsilon and alpha
    initial_params = [0.1, 0.1]
    bounds = [(0, 1), (0, 1)]  # epsilon and alpha are in [0, 1]
    
    # Minimize negative log-likelihood
    result = minimize(
        simulate_sequence, initial_params, args=(arms_sampled, rewards_obtained, num_arms),
        bounds=bounds, method='L-BFGS-B'
    )
    epsilon, alpha = result.x
    return epsilon, alpha

# Run fitting
epsilon, alpha = fit_parameters(arms_sampled, rewards_obtained, num_arms)
print(f"Fitted Epsilon (Exploration Rate): {epsilon}")
print(f"Fitted Alpha (Learning Rate): {alpha}")
