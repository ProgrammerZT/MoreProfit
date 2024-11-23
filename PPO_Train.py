import numpy as np
import pandas as pd
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from portfolio_env import PortfolioEnv

# Load data
a500_data = pd.read_csv('data/A500_train.csv')
c1000_data = pd.read_csv('data/C1000_train.csv')

# Process data
a500_data['涨跌幅'] = a500_data['涨跌幅'] / 100
c1000_data['涨跌幅'] = c1000_data['涨跌幅'] / 100

merged_data = pd.merge(a500_data, c1000_data, on='日期', suffixes=('_A500', '_C1000'))

# Create features, targets, and benchmark_returns
features = merged_data[['开盘_A500',
                        '收盘_A500',
                        '最高_A500',
                        '最低_A500',
                        '成交量_A500',
                        '成交额_A500',
                        '振幅_A500',
                        '涨跌幅_A500',
                        '涨跌额_A500',
                        '换手率_A500',
                        '开盘_C1000',
                        '收盘_C1000',
                        '最高_C1000',
                        '最低_C1000',
                        '成交量_C1000',
                        '成交额_C1000',
                        '振幅_C1000',
                        '涨跌幅_C1000',
                        '涨跌额_C1000',
                        '换手率_C1000']].values

targets = merged_data[['涨跌幅_A500', '涨跌幅_C1000']].values
benchmark_returns = merged_data['涨跌幅_A500'].values

# Add cash column to targets
targets = np.hstack((targets, np.full((targets.shape[0], 1), 0.02 / 250)))

# Ensure sizes match
assert features.shape[0] == targets.shape[0] == benchmark_returns.shape[0], "Data length mismatch!"

# Initialize environment
env = PortfolioEnv(features, targets, benchmark_returns, render_mode='human')

# Vectorized environment
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy",vec_env,verbose=1,gamma=0.997)
# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_portfolio_model")

print("Model saved as 'ppo_portfolio_model'")

# Evaluate the model
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
