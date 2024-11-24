import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from data_preprocessor import DataPreprocessor
from portfolio_env import PortfolioEnv

preprocessor = DataPreprocessor('data/A500_test.csv', 'data/C1000_test.csv')
features, targets, benchmark_returns = preprocessor.preprocess_data()

# Initialize environment
env = PortfolioEnv(features, targets, benchmark_returns)
# Vectorized environment
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env,norm_obs=True, norm_reward=True)
# Load the pre-trained model
model = TD3.load("td3_portfolio_model.zip")

# Reset the environment
obs, _ = env.reset()
done = False
total_reward = 0
steps = 0

# Run a test episode
while not done:
    action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions for evaluation
    obs, reward, done, _, info = env.step(action)
    print(info)
    print(reward)
    total_reward = total_reward+reward
    steps += 1
    env.render()  # Optionally render the environment state

print(f"Test completed in {steps} steps. Total reward: {np.exp(total_reward)}")
