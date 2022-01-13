import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, num_states, num_actions, hidden1=400, hidden2=400):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_actions)
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden1=400, hidden2=400):
        super(Critic, self).__init__()
        self.fc1_1 = nn.Linear(num_states, hidden1)
        self.fc1_2 = nn.Linear(num_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state_action_list):
        state, action = state_action_list
        out = self.fc1_1(state) + self.fc1_2(action)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class Agent:
    def __init__(self, batch_size, memory_size, device):
        return
    def searching_action(self, step, state):
        action = None
        return action
    def deterministic_action(self, step, state):
        action = None
        return action
    def update_memory_buffer(self, current_state, action, next_state, reward):
        return
    def update_main_network(self):
        return
    def update_target_network(self):
        return

class Replay_Memory:
    def __init__(self, num_states, num_actions, batch_size, max_size):
        return
    def give_state_to_buffer(self, current_state, action, next_state, reward):
        return
    def get_minibatch(self):
        return






























