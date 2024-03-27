import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import random
from agent import Agent


def get_discretized_action_map(action_space, n=10):
    lower = action_space.low
    higher = action_space.high

    space = (higher - lower) / n

    result = {}
    for i in range(n+1):
        result[i] = lower + i * space
    return result


# Main training loop
task_name = 'InvertedPendulum-v5'
env = gym.make(task_name)
# env = RecordVideo(env, './video', episode_trigger = lambda x: x % 10 == 0 )

# discretize action space
state_dim = env.observation_space.shape[0]
action_map = get_discretized_action_map(env.action_space, 100)
action_dim = len(action_map)

agent = Agent(state_dim, action_dim)
episodes=100
batch_size = 32

for e in range(episodes):
    state, _ = env.reset()
    for time in range(1000):  # Set to a value that suits your environment's maximum step
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action_map[action])
        # reward = reward if not done or time == 499 else -10
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    if e % 10 == 0:
        agent.save_model("./models/dqn_{}_episode_{}.pt".format(task_name, e))

env.close()



# demonstrate agent
env = gym.make(task_name, render_mode="rgb_array")
env = RecordVideo(env, "./videos/dqn_{}_episode_{}.pt".format(task_name, episodes), episode_trigger=lambda x: True)

for trial in range(10):
    state, _ = env.reset()
    for time in range(1000):  # Set to a value that suits your environment's maximum step
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action_map[action])
        state = next_state
        if done:
            break

# env.close_video_recorder()
env.close()