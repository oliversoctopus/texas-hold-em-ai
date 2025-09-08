import torch
import torch.nn as nn
from collections import namedtuple, deque
import numpy as np
import random

# Experience replay with prioritization
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
    
    def push(self, *args, priority=None):
        self.buffer.append(Experience(*args))
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(priority if priority else max_priority)
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))
        
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-6
    
    def __len__(self):
        return len(self.buffer)

# Improved Neural Network with batch normalization
class PokerNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 512, 256], output_size=5, dropout_rate=0.3):
        super(PokerNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Handle batch size of 1 for batch norm
        if x.size(0) == 1:
            self.eval()
            output = self.network(x)
            self.train()
            return output
        return self.network(x)