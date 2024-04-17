import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import copy

class Agent:
    def __init__(self, state_dim, action_dim, model, reward_shaping_model=None, reward_shaping_weight=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.gamma = 1.0  # Discount rate
        self.epsilon = 0.05  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        # TODO: add a separate target model that only updates onces K episode
        self.model = model
        self.reward_shaping_model = reward_shaping_model
        self.reward_shaping_weight = reward_shaping_weight
        self.reward_shaping_decay = 0.995
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
    
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
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            target = reward
            reward_shaping = None
            
            if self.reward_shaping_model is not None:
                # (1, 501)
                reward_shaping = self.reward_shaping_model(state)
            
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model(next_state).detach()).item())
            
            
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            q_pred = self.model(state)
            mse_loss = torch.nn.functional.mse_loss(target_f, q_pred)
            
            if reward_shaping is not None:
                # KL divergence loss
                rescale_factor = reward / reward_shaping[0][action]
                policy_clone_loss = torch.nn.functional.mse_loss(reward_shaping * rescale_factor, q_pred)
                loss = self.reward_shaping_weight * policy_clone_loss + mse_loss
            else:
                loss = mse_loss

            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.reward_shaping_weight *= self.reward_shaping_decay

    def save_model(self, path):
        torch.save(self.model, path)