import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN_BUDGET_S = 40.0
EPOCHS = 20
BATCH = 512
LR = 3e-3


class Agent:
    def __init__(self, output_dim: int = 10, seed=None):
        torch.set_num_threads(3)
        self.output_dim = output_dim
        self.seed = 0 if seed is None else seed

    @staticmethod
    def _flatten(X):
        array = np.asarray(X).reshape(len(X), -1).astype(np.float32) / 255.0
        return torch.from_numpy(array)

    def train(self, X_train, y_train):
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        X = self._flatten(X_train)
        self.mu, self.sd = X.mean(), X.std()
        X = (X - self.mu) / self.sd
        y = torch.from_numpy(np.asarray(y_train).reshape(-1).astype(np.int64))

        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LR,
            epochs=EPOCHS,
            steps_per_epoch=(len(X) + BATCH - 1) // BATCH,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0,
        )

        self.model.train()
        for _ in range(EPOCHS):
            for idx in torch.randperm(len(X)).split(BATCH):
                loss = F.cross_entropy(self.model(X[idx]), y[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if time.perf_counter() - t0 > TRAIN_BUDGET_S:
                    return

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            X = (self._flatten(X_test) - self.mu) / self.sd
            return self.model(X).argmax(1).tolist()
