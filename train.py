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
from model import MLP
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

def get_discretized_action_map(action_space, n=10):
    lower = action_space.low
    higher = action_space.high

    space = (higher - lower) / n

    result = {}
    for i in range(n + 1):
        result[i] = lower + i * space
    return result

def get_adaptor_model(pretrained_path):
    old_model = torch.load(pretrained_path)
    old_input_dim = old_model.input_dims


# Main training loop
# Change mujoco config here: .venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets/inverted_pendulum.xml
prefix = "from_scratch_double_gravity_reload_from_single_pretrained"
# task_name = 'InvertedPendulum-v5'
task_name = "InvertedDoublePendulum-v4"
env = gym.make(task_name)
episodes=500
batch_size=64

run_name = "{}_dqn_{}_episode_{}".format(prefix, task_name, episodes)

writer = SummaryWriter("./log/{}".format(run_name))

data = {}

# discretize action space
state_dim = env.observation_space.shape[0]
action_map = get_discretized_action_map(env.action_space, 500)
action_dim = len(action_map)

previous_model = None
# model = MLP(state_dim, action_dim, previous_model)
model = torch.load("models/from_scratch_single_gravity_dqn_InvertedDoublePendulum-v4_episode_500.pt")

agent = Agent(state_dim, action_dim, model)


for e in range(episodes):
    state, _ = env.reset()
    reward_accu = 0
    for time in range(
        1000
    ):  # Set to a value that suits your environment's maximum step
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action_map[action])
        # reward = reward if not done or time == 499 else -10
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        reward_accu += reward

        if done:
            data[e] = reward_accu
            print(f"episode: {e}/{episodes}, score: {reward_accu}, e: {agent.epsilon:.2}")
            writer.add_scalar("reward", reward_accu, e)
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

env.close()
# Save data
df = pd.DataFrame(data.items(), columns=["episode", "total_reward"])
df.to_csv("./data/{}.csv".format(run_name))
agent.save_model("./models/{}.pt".format(run_name))


# # demonstrate agent
# env = gym.make(task_name, render_mode="rgb_array")
# env = RecordVideo(
#     env,
#     "./videos/dqn_{}_episode_{}.pt".format(task_name, episodes),
#     episode_trigger=lambda x: True,
# )

# for trial in range(10):
#     state, _ = env.reset()
#     for time in range(
#         1000
#     ):  # Set to a value that suits your environment's maximum step
#         action = agent.act(state)
#         next_state, reward, done, _, _ = env.step(action_map[action])
#         state = next_state
#         if done:
#             break

# # env.close_video_recorder()
# env.close()
