import gc
import math
import random
import time
import traceback
from collections import deque, namedtuple
from itertools import count
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import transform

from space_invader_reward_change import SpaceInvaderGame

torch.set_flush_denormal(True)
torch.cuda.empty_cache()

TRAINING = True

MEM_CAPACITY = 140000
LR = 1e-3
N_EPISODES = 10000
MAX_STEPS = 50000
BATCH_SIZE = 256
GAMMA = 0.999
TARGET_UPDATE = 25
PRETRAIN_LENGTH = BATCH_SIZE
FRAME_SKIP = 4

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# game constants
WIDTH = 120
HEIGHT = 90

STACK_SIZE = 4

MODEL_PATHS = f"./models_fs/reward_change_v2_rms_lr_{LR}_tg_{TARGET_UPDATE}"
Path(MODEL_PATHS).mkdir(parents=True, exist_ok=True)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def q_mem(mem):
    total_memory = 0
    total_state_mem = 0
    total_action_mem = 0
    total_next_state_mem = 0
    total_reward_mem = 0
    count = 0
    buffer = mem.buffer
    for x in buffer:
        temp_mem = 0
        state = x.state
        action = x.action
        next_state = x.next_state
        reward = x.reward
        state_mem = state.element_size() * state.nelement()
        action_mem = action.element_size() * action.nelement()
        next_state_mem = next_state.element_size() * next_state.nelement()
        reward_mem = reward.element_size() * reward.nelement()
        temp_mem = state_mem + action_mem + next_state_mem + reward_mem
        total_state_mem += state_mem
        total_action_mem += action_mem
        total_next_state_mem += next_state_mem
        total_reward_mem += reward_mem
        total_memory += temp_mem
        count += 1

    total_memory /= 1000000
    total_state_mem /= 1000000
    total_action_mem /= 1000000
    total_next_state_mem /= 1000000
    total_reward_mem /= 1000000
    print(
        f"Total Queue Obejcts: {count} Memory: {total_memory} MB. State Mem: {total_state_mem}"
        f" Action Mem: {total_action_mem} Next State Mem {total_next_state_mem} Reward Mem {total_reward_mem}"
    )


def preprocess_frame(frame):
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]) / 255

    preprocessed_frame = rgb2gray(frame)
    preprocessed_frame = transform.resize(preprocessed_frame, [HEIGHT, WIDTH])
    # preprocessed_frame = preprocessed_frame.transpose(2, 0, 1)
    return preprocessed_frame


# Initialize deque with zero-images one array for each image
stacked_frames = deque(
    [np.zeros((WIDTH, HEIGHT), dtype=np.int16) for i in range(STACK_SIZE)], maxlen=4
)


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque(
            [torch.zeros((WIDTH, HEIGHT)) for i in range(STACK_SIZE)], maxlen=4,
        )

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = (
            torch.stack([torch.tensor(fr) for fr in stacked_frames], axis=0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = (
            torch.stack([torch.tensor(fr) for fr in stacked_frames], axis=0)
            .unsqueeze(0)
            .float()
            .to(device)
        )
    return stacked_state, stacked_frames


# def stack_frames(stacked_frames, state, is_new_episode):
#     # Preprocess frame
#     frame = preprocess_frame(state)

#     if is_new_episode:
#         # Clear our stacked_frames
#         stacked_frames = deque(
#             [np.zeros((WIDTH, HEIGHT), dtype=np.int16) for i in range(STACK_SIZE)],
#             maxlen=4,
#         )

#         # Because we're in a new episode, copy the same frame 4x
#         stacked_frames.append(frame)
#         stacked_frames.append(frame)
#         stacked_frames.append(frame)
#         stacked_frames.append(frame)

#         # Stack the frames
#         stacked_state = np.stack(stacked_frames, axis=0)

#     else:
#         # Append frame to deque, automatically removes the oldest frame
#         stacked_frames.append(frame)

#         # Build the stacked state (first dimension specifies different frames)
#         stacked_state = np.stack(stacked_frames, axis=0)

#     stacked_state = torch.from_numpy(stacked_state).unsqueeze(0).to(device).float()
#     return stacked_state, stacked_frames


def transpose_tensors(inp):
    inp = inp.transpose(1, -1)
    inp = inp.transpose(-2, -1)
    return inp


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        # buffer_size = len(self.buffer)
        # index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        # return [self.buffer[i] for i in index]
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class NNModel(nn.Module):
    def __init__(self, height, width, num_action):
        super(NNModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=32, kernel_size=3, padding=1, stride=2
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2
        )
        self.bn3 = nn.BatchNorm2d(64)

        linear_input_size = 64 * 15 * 12
        linear_input_size = int(linear_input_size)
        self.linear_input_size = linear_input_size
        self.fc = nn.Linear(linear_input_size, num_action)

    def forward(self, x):
        # print("inp")
        # print(x.shape)
        start_time = time.time()
        x = F.relu(self.conv1(x))
        # print("conv1")
        # print(x.shape)
        x = self.bn1(x)
        # print("bn1")
        # print(x.shape)
        c_b = time.time()

        x = F.relu(self.conv2(x))
        # print("conv2")
        # print(x.shape)
        cb_time = time.time()
        # print(f"Conv2 Time: {cb_time - c_b}")
        x = self.bn2(x)
        # print("bn2")
        # print(x.shape)
        c_b_2 = time.time()
        # print(f"BN2 Time: {c_b_2 - cb_time}")
        x = F.relu(self.conv3(x))
        # print("conv3")
        # print(x.shape)
        x = self.bn3(x)
        # print("bn3")
        # print(x.shape)
        x = x.view(x.shape[0], self.linear_input_size)
        # print("final x")
        # print(x.shape)
        view_time = time.time()
        # print(f"View Time: {view_time - c_b_2}")
        x = self.fc(x)
        # print(f"Forward took { time.time() - start_time} sec")
        return x


class SpaceInvaderDQN:
    def __init__(self, height, width, num_action) -> None:
        self.num_action = num_action
        self.policy_net = NNModel(height, width, num_action=num_action).to(device)
        self.target_net = NNModel(height, width, num_action=num_action).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.steps_done = 0

    def select_action(self, state, test=False):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        if test:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.num_action)]], device=device, dtype=torch.long
            )

    def optimize_model(self, memory):
        if len(memory) < BATCH_SIZE:
            return
        self.policy_net.train()
        self.target_net.eval()
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()


def train():
    space_game = SpaceInvaderGame()
    memory = ReplayMemory(capacity=MEM_CAPACITY)
    num_action = len(space_game.action_list)
    agent = SpaceInvaderDQN(height=HEIGHT, width=WIDTH, num_action=num_action)
    stacked_frames = deque(
        [np.zeros((WIDTH, HEIGHT), dtype=np.int16) for i in range(STACK_SIZE)], maxlen=4
    )

    for i in range(PRETRAIN_LENGTH):
        if i == 0:
            space_game.init_game()
            state = space_game.get_screen()
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        best_action = random.randint(1, num_action) - 1
        best_action_string = space_game.action_list[best_action]
        step_reward, done = space_game.step(best_action_string)

        next_state = space_game.get_screen()
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        if not torch.is_tensor(next_state):
            next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)
        next_state = next_state.float()

        reward = torch.tensor([step_reward], device=device)
        best_action = torch.tensor([[best_action]], device=device)
        memory.push(
            state.detach(), best_action.detach(), next_state.detach(), reward.detach()
        )
        state = next_state

    # q_mem(memory)
    for ep_i in range(N_EPISODES):
        print(f"Episode: {ep_i}")
        action_count_dict = {_a: 0 for _a in space_game.action_list}
        episode_start_time = time.time()
        total_episode_reward = 0
        total_episode_loss = 0
        space_game = SpaceInvaderGame()
        space_game.init_game()
        state = space_game.get_screen()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        if not torch.is_tensor(state):
            state = torch.from_numpy(state).unsqueeze(0).to(device)
        state = state.float()
        for step in count():
            start_time = time.time()

            best_action = agent.select_action(state)
            best_action_string = space_game.action_list[best_action.item()]
            action_count_dict[best_action_string] += 1
            for _ in range(FRAME_SKIP):
                step_reward, done = space_game.step(best_action_string)
                total_episode_reward += step_reward
                if done:
                    break

            next_state = space_game.get_screen()
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            reward = torch.tensor([step_reward], device=device)

            if not torch.is_tensor(next_state):
                next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)
            next_state = next_state.to(device)
            next_state = next_state.float()
            # memory.push(state, best_action, next_state, reward)
            memory.push(
                state.detach(),
                best_action.detach(),
                next_state.detach(),
                reward.detach(),
            )
            state = next_state
            opt_start = time.time()

            step_loss = agent.optimize_model(memory)
            total_episode_loss += step_loss
            if step % 1000 == 0:
                print(
                    f"Episode: {ep_i}. Step: {step}. Last Best Action: {best_action_string}. "
                    f"Reward: {step_reward}. Total Casualties: {space_game.total_casualties}. "
                    f"Time: {time.time() - start_time:.2f}. Opt Time: {time.time() - opt_start:.3f}"
                )

            if done or step >= MAX_STEPS:
                break
        if ep_i % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        # gpu_usage(useOldCode=True)
        # q_mem(memory)
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"Episode: {ep_i}. Loss: {total_episode_loss/step:.2f} "
            f"Total Reward: {total_episode_reward}. Total Casualties: {space_game.total_casualties} "
            f"Total Time: {time.time() - episode_start_time:.2f}"
        )
        print(f"Action Distribution: {action_count_dict}")
        if TRAINING and ep_i % 5 == 0:
            save_path = f"{MODEL_PATHS}/policy_net_ep_{ep_i}.pth"
            torch.save(agent.policy_net.state_dict(), save_path)
    save_path = f"{MODEL_PATHS}/policy_net_model.pth"
    torch.save(agent.policy_net.state_dict(), save_path)


def simulate():
    save_path = f"{MODEL_PATHS}/policy_net_ep_500.pth"
    print(save_path)
    stacked_frames = deque(
        [torch.zeros((WIDTH, HEIGHT)) for i in range(STACK_SIZE)], maxlen=4,
    )

    space_game = SpaceInvaderGame()
    space_game.init_game()
    num_action = len(space_game.action_list)
    agent = SpaceInvaderDQN(height=HEIGHT, width=WIDTH, num_action=num_action)
    agent.policy_net.load_state_dict(
        torch.load(save_path, map_location=torch.device("cpu"))
    )
    agent.policy_net.eval()
    state = space_game.get_screen()
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    while True:
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).unsqueeze(0).to(device)
        best_action = agent.select_action(state, test=True)
        best_action_string = space_game.action_list[best_action.item()]
        # with torch.no_grad():
        #     action = agent.policy_net(state)
        # best_action_max_value, best_action_max_index = torch.max(action, 1)
        # best_action_string = space_game.action_list[best_action_max_index.item()]
        print(best_action_string)
        for _ in range(FRAME_SKIP):
            _, done = space_game.step(best_action_string)
            if done:
                break

        next_state = space_game.get_screen()
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        state = next_state
        if done:
            break


if __name__ == "__main__":
    if TRAINING:
        try:
            train()
        except Exception as e:
            print(e)
            traceback.print_exc()
    else:
        simulate()
