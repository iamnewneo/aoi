import math
import time
from collections import deque
from itertools import count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame
from space_invader_reward_change import SpaceInvaderGame

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

TRAINING = True
MODEL_PATHS = f"./models_fs/modified_dqn"
Path(MODEL_PATHS).mkdir(parents=True, exist_ok=True)

env_dummy = SpaceInvaderGame()

INPUT_SHAPE = (4, 84, 84)
N_EPISODES = 10000
MAX_STEPS = 50000
ACTION_SIZE = len(env_dummy.action_list)
SEED = 0
GAMMA = 0.99  # discount factor
BUFFER_SIZE = 100000  # replay buffer size
BATCH_SIZE = 64  # Update batch size
LR = 0.0001  # learning rate
TAU = 1e-3  # for soft update of target parameters
UPDATE_EVERY = 1  # how often to update the network
UPDATE_TARGET = 5000  # After which thershold replay to be started
EPS_START = 0.99  # starting value of epsilon
EPS_END = 0.01  # Ending value of epsilon
EPS_DECAY = 100  # Rate by which epsilon to be decayed

agent = DQNAgent(
    INPUT_SHAPE,
    ACTION_SIZE,
    SEED,
    device,
    BUFFER_SIZE,
    BATCH_SIZE,
    GAMMA,
    LR,
    TAU,
    UPDATE_EVERY,
    UPDATE_TARGET,
    DQNCnn,
)


def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, exclude=None, output=84)
    frames = stack_frame(frames, frame, is_new)

    return frames


start_epoch = 0
scores = []
scores_window = deque(maxlen=20)


epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(
    -1.0 * frame_idx / EPS_DECAY
)


def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    for i_episode in range(start_epoch + 1, n_episodes + 1):
        env = SpaceInvaderGame()
        env.init_game()
        state = stack_frames(None, env.get_screen(), True)
        score = 0
        total_episode_loss = 0
        action_count_dict = {_a: 0 for _a in env.action_list}
        episode_start_time = time.time()
        eps = epsilon_by_epsiode(i_episode)
        # while True:
        for step in count():
            action = agent.act(state, eps)
            action_string = env.action_list[action.item()]
            action_count_dict[action_string] += 1
            reward, done = env.step(action_string)
            next_state = env.get_screen()
            score += reward
            next_state = stack_frames(state, next_state, False)
            step_loss = agent.step(state, action, reward, next_state, done)
            if step_loss is not None:
                total_episode_loss += step_loss
            state = next_state
            if step % 1000 == 0:
                print(
                    f"Episode: {i_episode}. Step: {step}. Last Best Action: {action_string}. "
                    f"Reward: {reward}. Total Casualties: {env.total_casualties}. "
                )
            if done:
                break
            if step >= MAX_STEPS:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print(
            f"Episode: {i_episode}. Loss: {total_episode_loss/step:.3f} "
            f"Average Reward: {np.mean(scores_window):.2f}. Total Casualties: {env.total_casualties} "
            f"Total Steps: {step}. Total Time: {time.time() - episode_start_time:.2f}"
        )
        print(action_count_dict)
        if i_episode % 5 == 0:
            save_path = f"{MODEL_PATHS}/policy_net_ep_{i_episode}.pth"
            torch.save(agent.policy_net.state_dict(), save_path)
    return scores


scores = train(N_EPISODES)
