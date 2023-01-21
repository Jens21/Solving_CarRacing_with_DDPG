import torch as th
import torch.nn as nn

class Network():
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()

