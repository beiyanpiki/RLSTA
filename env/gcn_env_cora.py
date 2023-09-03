import torch
import gymnasium as gym
import numpy as np
import torch_geometric.transforms as T
from gymnasium import spaces
import random
from greatx.training import Trainer
from .my_model import MyModel


class GcnEnv(gym.Env):
    metadata = {"render_modes": ["gcn"], "datasets": ["cora", "pubmed", "karate_club"]}

    def __init__(self, data, splits, target=1, max_b=3, save_reward=False):
        self.data = data
        self.splits = splits
        assert target < self.data.num_nodes
        self.target = target

        self._agent_to_nodes = {}
        cnt = 0
        self._agent_to_nodes[cnt] = set([])
        for i in range(self.data.num_nodes - 35, self.data.num_nodes):
            cnt += 1
            self._agent_to_nodes[cnt] = set([i])
            for j in range(i + 1, self.data.num_nodes):
                cnt += 1
                self._agent_to_nodes[cnt] = set([i, j])

        self.observation_space = spaces.Discrete(cnt + 1)

        self.action_space = spaces.Discrete(35 * 2)
        self.baseline = []
        self.max_b = 10
        self.save_reward = save_reward
        if self.save_reward:
            self.reward_matrix = np.ones((cnt + 1, 35 * 2))

    def _get_obs(self):
        return self._agent_location

    def _get_location_by_x(self, x):
        for key, val in self._agent_to_nodes.items():
            if val == x:
                return key

    def _get_info(self):
        return self._get_location_by_x(self._agent_location)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._agent_location = 0
        observation = self._get_obs()
        self._count = 0
        return observation, {}

    def step(self, action):
        self._count += 1
        if isinstance(action, np.ndarray):
            action = np.int64(action)

        # while True:
        x = self._agent_to_nodes[self._agent_location]

        if action < self.data.num_nodes and action in x:
            xx = x - set([action])
            new_location = self._get_location_by_x(xx)
            self._agent_location = new_location
        elif (
            action >= self.data.num_nodes
            and (action - self.data.num_nodes) not in x
            and len(x) < 2
        ):
            xx = set([action - self.data.num_nodes]) | x
            new_location = self._get_location_by_x(xx)
            self._agent_location = new_location
        else:
            return (
                self._get_obs(),
                0,
                False,
                False,
                {"acc": 0, "mask": self._agent_to_nodes[self._agent_location]},
            )

        mask = torch.BoolTensor(self.data.num_edges)
        print(self._agent_location)
        print(self._agent_to_nodes[self._agent_location])
        for i in range(self.data.num_edges):
            u = self.data.edge_index[0][i]
            v = self.data.edge_index[1][i]
            if (
                u in self._agent_to_nodes[self._agent_location]
                or v in self._agent_to_nodes[self._agent_location]
            ):
                mask[i] = False
            else:
                mask[i] = True

        self.model = MyModel

        trainer_my_gcn = Trainer(
            self.model(self.data.x.size(-1), 4, True, True, mask),
            device="cuda",
        )
        trainer_my_gcn.fit(
            self.data, mask=(self.splits.train_nodes, self.splits.val_nodes)
        )
        logs = trainer_my_gcn.evaluate(self.data, self.splits.test_nodes)
        acc = logs["acc"]

        reward = acc - 0.4 * sum(self.baseline) / (len(self.baseline) + 0.1)
        # reward = acc
        self.baseline.append(acc)
        if len(self.baseline) > self.max_b:
            self.baseline = self.baseline[1:]

        if self.save_reward:
            self.reward_matrix[self._agent_location][action] = reward
            np.save("/home/miku/reinforce-gcn/rewardmatrix.npy", self.reward_matrix)
            print(np.count_nonzero(self.reward_matrix == 1))
        observation = self._get_obs()

        terminated = reward > 0.8
        truncated = acc > 0.8

        print(f"{self._count}\tOBS:{observation}\nREWARD: {reward}\tACC: {acc}\t")
        return (
            observation,
            reward,
            terminated,
            truncated,
            {"acc": acc, "mask": self._agent_to_nodes[self._agent_location]},
        )
