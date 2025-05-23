import gym
from gym import spaces
import numpy as np
from gym.envs.registration import register

register(
    id='LithoEnv-v0',
    entry_point='src.envs.litho_env:LithoEnv',
)

class LithoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.state = None

    def reset(self):
        self.state = np.random.rand(5)
        return self.state

    def step(self, action):
        reward = -np.abs(action[0] * self.state[0] - 0.5)
        done = np.random.rand() > 0.95
        self.state = np.random.rand(5)
        return self.state, reward, done, {}
