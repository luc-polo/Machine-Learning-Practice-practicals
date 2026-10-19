from pathlib import Path

import numpy as np


class Agent:
    def __init__(self):
        self.q = np.load(Path(__file__).parent / "q_table.npy")

    def setup(self, observation_space, action_space):
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        return int(np.argmax(self.q[int(observation)]))
