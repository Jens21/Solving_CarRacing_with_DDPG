import numpy as np

class ReplayBuffer():

    buffer = None
    buffer_idx = 0
    current_buffer_size = 0
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size

        self.mem = np.empty(self.buffer_size, dtype='object')

    def len(self):
        return self.current_buffer_size

    def add_sample(self, sample):
        self.mem[self.mem_idx] = sample

        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size

        if self.current_buffer_size != self.buffer_size:
            self.current_buffer_size += 1

    def get_samples(self, batch_size):
        indices = np.random.randint(0, self.current_buffer_size, batch_size)

        return self.buffer[indices]