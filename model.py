import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, previous_model=None):
        super(MLP, self).__init__()

        nn_dims = [128, 128]
        self.fc = nn.ModuleList()
        self.input_dim = input_dim

        # adaptor
        if previous_model is not None:
            previous_input_dim = previous_model.input_dim
            self.fc.append(nn.Linear(input_dim, previous_input_dim))
            self.fc.append(previous_model)
        else:
            in_dim = input_dim
            for out_dim in nn_dims:
                self.fc.append(nn.Linear(in_dim, out_dim))
                self.fc.append(nn.LeakyReLU())
                in_dim = out_dim
            self.fc.append(nn.Linear(in_dim, output_dim))
    
    def forward(self, x):
        for fc in self.fc:
            x = fc(x)
        return x
