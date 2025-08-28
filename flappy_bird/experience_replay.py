from collections import deque
import random

class ReplayMemory:
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        if seed is not None:
            random.seed(seed)
    
    def append(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)