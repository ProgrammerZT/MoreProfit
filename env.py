import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class PortfolioEnvSliding(gym.Env):
    def __init__(self, features, targets, benchmark_returns, initial_cash=1.0, window_size=10, cash_daily_return=0.02/250):
        """
        Initialize the Portfolio environment with sliding window logic.
        :param features: Standardized market features matrix (n_samples, n_features)
        :param targets: Target returns for next day for each asset (e.g., [涨幅A500, 涨幅C1000])
        :param benchmark_returns: Benchmark returns to calculate excess returns
        :param initial_cash: Initial cash value
        :param window_size: Number of past days to include in the state
        :param cash_daily_return: Fixed daily return for cash
        """
        super(PortfolioEnvSliding, self).__init__()

        self.features = features
        self.targets = targets
        self.benchmark_returns = benchmark_returns
        self.window_size = window_size
        self.cash_daily_return = cash_daily_return

        # Verify the shape of features and targets
        print("Features shape:", features.shape)
        print("Targets shape:", targets.shape)
        print("Benchmark Returns shape:", benchmark_returns.shape)

        self.n_samples, self.n_features = features.shape[0], features.shape[2]
        self.current_step = 0

        # Initial cash and weights
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.weights = np.array([1.0, 0.0, 0.0])  # Initial allocation: all cash

        # Action space: adjust cash, A500, C1000 weights (0, 1)
        self.action_space = spaces.Box(low=0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: current weights + sliding window features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 + window_size * self.n_features,),  # 3 weights + sliding window features
            dtype=np.float32
        )

    def reset(self):
        """Reset the environment."""
        self.current_step = 0
        self.cash = self.initial_cash
        self.weights = np.array([1.0, 0.0, 0.0])  # Reset to all cash
        return self._get_obs(), {}

    def step(self, action):
        """
        Execute an action and compute the new state, reward, and termination.
        """
        # Smooth weight adjustments: normalize action via softmax
        action = np.exp(action) / np.sum(np.exp(action))
        self.weights = self.weights + action - np.mean(action)
        self.weights = np.clip(self.weights, 0, 1)  # Ensure within [0, 1]
        self.weights /= np.sum(self.weights)  # Normalize to sum to 1

        # Use actual market returns for each asset
        next_market_returns = self.targets[self.current_step]  # This should be a vector with 3 elements
        next_market_returns[2] = self.cash_daily_return  # Set fixed daily return for cash

        portfolio_return = np.dot(self.weights, next_market_returns)  # This should be a scalar

        # Update cash as a scalar
        self.cash *= (1 + portfolio_return)

        # Calculate the excess return
        benchmark_return = self.benchmark_returns[self.current_step]
        excess_return = portfolio_return - benchmark_return

        # Reward is the excess return
        reward = excess_return

        # Increment step
        self.current_step += 1

        # Check if done
        done = (self.current_step >= self.n_samples - self.window_size) or (self.cash < 0.5 * self.initial_cash)
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        """Return the current state: current weights + sliding window features."""
        start = self.current_step
        end = self.current_step + self.window_size
        sliding_window_features = self.features[start:end].flatten()
        return np.concatenate([self.weights, sliding_window_features])

    def render(self):
        """Print the current state."""
        print(f"Step: {self.current_step}, Weights: {self.weights}, Cash: {self.cash}")


# Sliding window size
window_size = 10

# Create sliding window features and targets
def create_sliding_window(data, target_cols, benchmark_col, window_size=10):
    features, targets, benchmark_returns = [], [], []
    for i in range(len(data) - window_size):
        # Sliding window
        feature_window = data.iloc[i:i + window_size].values
        features.append(feature_window)

        # Next day's targets and benchmark returns
        target = data.iloc[i + window_size][target_cols].values
        targets.append(target)
        benchmark_returns.append(data.iloc[i + window_size][benchmark_col])
    return np.array(features), np.array(targets), np.array(benchmark_returns)

# Dummy data for testing
data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'target_A500': np.random.randn(100),
    'target_C1000': np.random.randn(100),
    'target_cash': np.zeros(100),  # We will set this in the step function
    'benchmark': np.random.randn(100)  # Dummy benchmark returns
})

# Create a list of target columns
target_cols = ['target_A500', 'target_C1000', 'target_cash']

features, targets, benchmark_returns = create_sliding_window(data, target_cols, 'benchmark', window_size)

# Initialize the environment
env = PortfolioEnvSliding(features, targets, benchmark_returns)

# Run a simple test
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Random action for testing
    obs, reward, done, _, _ = env.step(action)
    done = bool(done)  # Ensure done is a boolean
    total_reward += reward
    env.render()

print(f"Total Reward: {total_reward}")
