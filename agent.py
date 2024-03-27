import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from model import DQN

class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Discount rate
        self.epsilon = 0.05  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values, axis=1).item()  # returns action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model(next_state).detach()).item())
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.model, path)