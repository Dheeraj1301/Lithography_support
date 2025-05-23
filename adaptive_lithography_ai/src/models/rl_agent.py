from stable_baselines3 import PPO
from src.envs.litho_env import LithoEnv

def train_rl_agent():
    env = LithoEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("models/ppo_litho")
    print("✅ RL model trained and saved.")
