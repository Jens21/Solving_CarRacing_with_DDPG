import torch as th
import os

class Critic(th.nn.Module):
    def __init__(self):
        self.critic_policy = th.nn.Sequential(
            th.nn.Linear(96*84+1+3),
            th.nn.ReLU(),
            th.nn.Linear(100, 3)
        )

        self.critic_target = th.nn.Sequential(
            th.nn.Linear(96*84+1),
            th.nn.ReLU(),
            th.nn.Linear(100, 3)
        )
        self.critic_target.load_state_dict(self.critic_policy.state_dict())

    def forward(self, x):
        return self.critic_policy(x)

    def update_networks(self):
        self.critic_target.load_state_dict(self.critic_policy.state_dict())

    def save_network(self, path):
        th.save(self.critic_policy.state_dict(), os.path.join(path, 'actor.pth'))

    def load_network(self, path):
        self.critic_policy.load_state_dict(th.load('path'))
        self.critic_target.load_state_dict(self.critic_policy.state_dict())