import numpy as np

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size