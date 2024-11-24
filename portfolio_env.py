import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    def __init__(self, features, targets, benchmark_returns, initial_cash=1.0, cash_daily_return=0.02 / 250,
                 no_bench_mark=False,
                 render_mode=None):
        """
        Initialize the Portfolio Environment.

        Args:
            features (np.ndarray): Historical market features used as observations.
            targets (np.ndarray): Target returns for each asset at each time step.
            benchmark_returns (np.ndarray): Benchmark returns for comparison.
            initial_cash (float): Initial cash amount (default is 1.0).
            cash_daily_return (float): Daily return rate for risk-free cash asset (default is ~0.02 annualized).
            render_mode (str): Optional mode for rendering (e.g., 'human').
        """
        super(PortfolioEnv, self).__init__()

        self.features = features
        self.targets = targets
        self.benchmark_returns = benchmark_returns
        self.cash_daily_return = cash_daily_return
        self.render_mode = render_mode
        self.no_bench_mark = no_bench_mark
        self.delayed_reward = None
        # Shape of input data
        self.n_samples, self.n_features = features.shape
        # Add a small constant for the cash column (target for cash allocation)
        self.targets = np.hstack((self.targets, np.full((self.targets.shape[0], 1), self.cash_daily_return)))
        # Initialize time step
        self.current_step = 0

        # Initial cash and portfolio weights
        self.initial_cash = initial_cash
        self.cash = self.initial_cash
        self.weights = np.array([0.0, 0.0, 1.0])  # Start with all cash asset

        # Action space: allocation weights for three assets
        self.action_space = spaces.Box(low=0, high=1.0, shape=(3,), dtype=np.float64)

        # Observation space: portfolio weights + market features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 + self.n_features,),
            dtype=np.float64
        )

        # Variable to store delayed reward
        self.prev_reward = 0

    def reset(self, seed=None):
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Random seed for reproducibility.

        Returns:
            tuple: Initial observation and an empty dictionary.
        """
        self.current_step = 0
        self.cash = self.initial_cash
        self.weights = np.array([0.0, 0.0, 1.0])  # Reset weights to all cash
        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (np.ndarray): Proposed portfolio allocation action.

        Returns:
            tuple: Observation, reward, termination flag, truncation flag, and additional info.
        """
        # Normalize action using softmax to ensure valid portfolio weights
        action = np.exp(action) / np.sum(np.exp(action))

        # Update portfolio weights
        self.weights = self.weights + action - np.mean(action)
        self.weights = np.clip(self.weights, 0, 1)  # Ensure weights are in [0, 1]
        self.weights /= np.sum(self.weights)  # Normalize weights to sum to 1

        # Get returns for the next time step
        if self.current_step + 1 < self.n_samples:
            next_market_returns = self.targets[self.current_step + 1]
            benchmark_return = self.benchmark_returns[self.current_step + 1]
        else:
            # Use current returns for the last step if out of bounds
            next_market_returns = self.targets[self.current_step]
            benchmark_return = self.benchmark_returns[self.current_step]

        # Compute portfolio return
        portfolio_return = np.dot(self.weights, next_market_returns)

        prev_cash = self.cash

        # Update cash value based on portfolio return
        self.cash *= (1 + portfolio_return)

        log_return = np.log(self.cash / prev_cash)

        # self.delayed_reward = log_return
        if self.no_bench_mark is True:
            self.delayed_reward = log_return

        if self.no_bench_mark is False:
            # Compute delayed reward (excess return over benchmark)
            self.delayed_reward = log_return - np.log(1 + benchmark_return)

        # Advance time step
        self.current_step += 1

        # Check termination conditions
        terminated = self.current_step >= self.n_samples  # End of data
        truncated = self.cash < 0.5 * self.initial_cash  # Cash dropped below 50% of initial value

        # Return previous step's reward (delayed reward mechanism)
        reward = getattr(self, "prev_reward")
        self.prev_reward = self.delayed_reward  # Store current reward for next step

        return self._get_obs(), reward, terminated, truncated, {"weights": self.weights, "cash": self.cash}

    def _get_obs(self):
        """
        Generate the current observation.

        Returns:
            np.ndarray: Concatenated portfolio weights and market features.
        """
        if self.current_step < self.n_samples:
            current_features = self.features[self.current_step]
        else:
            # Use the last available features if out of bounds
            current_features = self.features[-1]
        return np.concatenate([self.weights, current_features])

    def render(self):
        """
        Render the current state of the environment (optional).
        """
        if self.render_mode:
            print(f"Step: {self.current_step}, Weights: {self.weights}, Cash: {self.cash}")
