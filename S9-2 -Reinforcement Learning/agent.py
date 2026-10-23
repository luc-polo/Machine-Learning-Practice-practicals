"""Compact LunarLander-v3 policy ensemble; NumPy inference only."""

from pathlib import Path

import numpy as np


class Agent:
    def __init__(self):
        with np.load(Path(__file__).with_name("policy.npz"), allow_pickle=False) as data:
            self.layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(3)]

    def setup(self, observation_space, action_space):
        from flexkit.spaces import decode_space

        self.observation_space = decode_space(observation_space)
        self.action_space = decode_space(action_space)
        return True

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        if terminated or truncated:
            return None
        x = np.broadcast_to(np.asarray(observation, dtype=np.float32), (2, 8))
        for weight, bias in self.layers[:-1]:
            x = np.tanh(np.einsum("nij,nj->ni", weight, x) + bias)
        weight, bias = self.layers[-1]
        logits = (np.einsum("nij,nj->ni", weight, x) + bias).mean(axis=0)
        if action_mask is not None:
            logits = np.where(action_mask, logits, -np.inf)
        return int(np.argmax(logits))
