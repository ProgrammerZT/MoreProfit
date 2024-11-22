from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from data_preprocessor import DataPreprocessor
from portfolio_env import PortfolioEnv

preprocessor = DataPreprocessor('data/A500_train.csv', 'data/C1000_train.csv')
# Preprocess data
features, targets, benchmark_returns = preprocessor.preprocess_data()

# Initialize environment
env = PortfolioEnv(features, targets, benchmark_returns, render_mode='human',no_bench_mark=True)

# Vectorized environment
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy",vec_env,verbose=1)
# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_portfolio_model")

print("Model saved as 'ppo_portfolio_model'")
