import torch as th
import os

class Actor(th.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.actor_policy = th.nn.Sequential(
            # th.nn.Linear(96*84+1, 100),
            th.nn.Linear(48*42+1, 100),
            # th.nn.Linear(32*28+1, 100),
            # th.nn.Linear(24*21+1, 100),
            th.nn.ReLU(),
            th.nn.Linear(100, 3)
        )

        self.actor_target = th.nn.Sequential(
            # th.nn.Linear(96*84+1, 100),
            th.nn.Linear(48*42+1, 100),
            # th.nn.Linear(32*28+1, 100),
            # th.nn.Linear(24*21+1, 100),
            th.nn.ReLU(),
            th.nn.Linear(100, 3)
        )
        self.actor_target.load_state_dict(self.actor_policy.state_dict())

    def forward(self, x):
        out = self.actor_policy(x)
        out[:, 0] = th.tanh(out[:, 0])
        out[:, 1:] = th.sigmoid(out[:, 1:])

        return out

    def update_networks(self):
        self.actor_target.load_state_dict(self.actor_policy.state_dict())

    def save_network(self, path):
        th.save(self.actor_policy.state_dict(), os.path.join(path, 'actor.pth'))

    def load_network(self, path):
        self.actor_policy.load_state_dict(th.load('path'))
        self.actor_target.load_state_dict(self.actor_policy.state_dict())