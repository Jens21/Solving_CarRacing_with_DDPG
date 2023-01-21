import torch as th
import os

class Actor(th.nn.Module):
    def __init__(self):
        self.actor_policy = th.nn.Sequential(
            th.nn.Linear(96*84+1),
            th.nn.ReLU(),
            th.nn.Linear(100, 3)
        )

        self.actor_target = th.nn.Sequential(
            th.nn.Linear(96*84+1),
            th.nn.ReLU(),
            th.nn.Linear(100, 3)
        )
        self.actor_target.load_state_dict(self.actor_policy.state_dict())

    def forward(self, x):
        return self.actor_policy(x)

    def update_networks(self):
        self.actor_target.load_state_dict(self.actor_policy.state_dict())

    def save_network(self, path):
        th.save(self.actor_policy.state_dict(), os.path.join(path, 'actor.pth'))

    def load_network(self, path):
        self.actor_policy.load_state_dict(th.load('path'))
        self.actor_target.load_state_dict(self.actor_policy.state_dict())