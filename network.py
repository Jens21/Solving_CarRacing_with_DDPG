import torch as th
import torch.nn as nn

from actor import Actor
from critic import Critic

class Network():
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()

