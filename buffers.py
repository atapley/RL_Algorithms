import torch as T
import random

# Buffer used during training to hold transitions for learning.
# Transitions can hold different information depending on Agent


class Buffer():
    def __init__(self, buffer_size, batch_size):
        super(Buffer, self).__init__()
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def reset(self):
        self.buffer = []

    def is_full(self):
        return len(self.buffer) == self.buffer_size

    def insert_transition(self, transition):
        if self.is_full():
            self.buffer.pop(0)
            self.buffer.append(transition)
        else:
            self.buffer.append(transition)

    def sample_buffer(self):
        batch = random.sample(self.buffer, self.batch_size)
        return batch

    def get_buffer(self):
        return self.buffer
