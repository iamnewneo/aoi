import time
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import count
import torch.optim as optim
from collections import namedtuple

from collections import deque
from skimage import transform
from skimage.color import rgb2gray
from space_invader import SpaceInvaderGame

torch.set_flush_denormal(True)

TRAINING = True

MEM_CAPACITY = 1000000
LR = 1e-3
N_EPISODES = 500
MAX_STEPS = 50000
BATCH_SIZE = 64
GAMMA = 0.999
TARGET_UPDATE = 5
PRETRAIN_LENGTH = BATCH_SIZE

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# game constants
WIDTH = 110
HEIGHT = 84

STACK_SIZE = 4
STATE_SIZE = [110, 84, 4]

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_frame(frame):
    gray = rgb2gray(frame)

    normalized_frame = gray / 255.0

    preprocessed_frame = transform.resize(normalized_frame, [110, 84])

    return preprocessed_frame


# Initialize deque with zero-images one array for each image
stacked_frames = deque(
    [np.zeros((110, 84), dtype=np.int16) for i in range(STACK_SIZE)], maxlen=4
)


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque(
            [np.zeros((110, 84), dtype=np.int16) for i in range(STACK_SIZE)], maxlen=4
        )

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


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
            in_channels=4, out_channels=32, kernel_size=8, padding=1, stride=4
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2
        )
        self.bn3 = nn.BatchNorm2d(64)

        linear_input_size = 2240
        self.linear_input_size = linear_input_size
        self.fc = nn.Linear(linear_input_size, num_action)

    def forward(self, x):
        start_time = time.time()
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        c_b = time.time()

        x = F.relu(self.conv2(x))
        cb_time = time.time()
        # print(f"Conv2 Time: {cb_time - c_b}")
        x = self.bn2(x)
        c_b_2 = time.time()
        # print(f"BN2 Time: {c_b_2 - cb_time}")
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = x.view(x.shape[0], self.linear_input_size)
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

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def select_action(self, state, steps_done):
        with torch.no_grad():
            if len(state.shape) == 3:
                state = state.unsqueeze(1)
            state = transpose_tensors(state)
            action = self.policy_net(state)

        action_max_value, action_max_index = torch.max(action, 1)
        action = action_max_index.item()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * steps_done / EPS_DECAY
        )
        if np.random.rand(1) >= eps_threshold:  # epslion greedy
            action = random.randint(0, self.num_action - 1)

        action = torch.tensor([[action]], device=device)
        return action

    def optimize_model(self, memory):
        start_time = time.time()
        # if len(memory) < BATCH_SIZE:
        #     return

        transitions = memory.sample(BATCH_SIZE)
        # print(transitions)
        ## This converts transitions into batches [input for pytroch model]
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

        cat_time = time.time()
        # print(f"cat took: {cat_time - start_time} sec")

        if len(state_batch.shape) == 3:
            state_batch = state_batch.unsqueeze(1)
        unsq_1_time = time.time()
        # print(f"unsq_1 took: {unsq_1_time - cat_time} sec")

        # print("Here 1")
        state_batch = state_batch.to(device)
        state_batch = transpose_tensors(state_batch)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        policy_time = time.time()
        # print(f"policy took: {policy_time - unsq_1_time} sec")

        if len(non_final_next_states.shape) == 3:
            non_final_next_states = non_final_next_states.unsqueeze(1)

        unsq_2_time = time.time()
        # print(f"unsq_2 took: {unsq_2_time - policy_time} sec")

        # print("Here 2")
        non_final_next_states = non_final_next_states.to(device)
        non_final_next_states = transpose_tensors(non_final_next_states)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1)[0].detach()
            )

        targ_time = time.time()
        # print(f"targ took: {targ_time - unsq_2_time} sec")

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        loss_time = time.time()
        # print(f"loss took: {loss_time - targ_time} sec")

        # print("Here 3")
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        opt_loss_time = time.time()
        # print(f"opt_loss took: {opt_loss_time - loss_time} sec")
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        opt_step_time = time.time()
        # print(f"opt_step took: {opt_step_time - opt_loss_time} sec")


def train():
    space_game = SpaceInvaderGame()
    memory = ReplayMemory(capacity=MEM_CAPACITY)
    num_action = len(space_game.action_list)
    agent = SpaceInvaderDQN(height=HEIGHT, width=WIDTH, num_action=num_action)
    stacked_frames = deque(
        [np.zeros((110, 84), dtype=np.int16) for i in range(STACK_SIZE)], maxlen=4
    )

    for i in range(PRETRAIN_LENGTH):
        if i == 0:
            space_game.init_game()
            state = space_game.get_screen()
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).unsqueeze(0).to(device)
        state = state.float()

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
        memory.push(state, best_action, next_state, reward)
        state = next_state

    for ep_i in range(N_EPISODES):
        print(f"Episode: {ep_i}")
        total_episode_reward = 0
        space_game.init_game()
        state = space_game.get_screen()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        if not torch.is_tensor(state):
            state = torch.from_numpy(state).unsqueeze(0).to(device)
        state = state.float()
        for step in count():
            start_time = time.time()

            best_action = agent.select_action(state, step)
            best_action_string = space_game.action_list[best_action.item()]
            step_reward, done = space_game.step(best_action_string)
            total_episode_reward += step_reward

            next_state = space_game.get_screen()
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            reward = torch.tensor([step_reward], device=device)

            if not torch.is_tensor(next_state):
                next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)
            next_state = next_state.to(device)
            next_state = next_state.float()
            memory.push(state, best_action, next_state, reward)
            state = next_state
            opt_start = time.time()

            agent.optimize_model(memory)
            if step % 1000 == 0:
                print(
                    f"Episode: {ep_i}. Step: {step}. Last Best Action: {best_action_string}. "
                    f"Reward: {step_reward}. Time: {time.time() - start_time}. "
                    f"Opt Time: {time.time() - opt_start}"
                )
            if done or step >= MAX_STEPS:
                break
        if ep_i % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Episode: {ep_i}. Total Reward: {total_episode_reward}")
        if TRAINING and ep_i % 5 == 0:
            save_path = f"policy_net_ep_{ep_i}.pth"
            torch.save(agent.policy_net.state_dict(), save_path)
    save_path = f"policy_net_model.pth"
    torch.save(agent.policy_net.state_dict(), save_path)


def simulate():
    save_path = f"policy_net_model.pth"
    stacked_frames = deque(
        [np.zeros((110, 84), dtype=np.int16) for i in range(STACK_SIZE)], maxlen=4
    )

    space_game = SpaceInvaderGame()
    space_game.init_game()
    num_action = len(space_game.action_list)
    agent = SpaceInvaderDQN(height=HEIGHT, width=WIDTH, num_action=num_action)
    agent.policy_net.load_state_dict(torch.load(save_path))
    agent.policy_net.eval()
    state = space_game.get_screen()
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    while True:
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).unsqueeze(0).to(device)
        state = state.float()
        state = transpose_tensors(state)
        action = agent.policy_net(state)
        best_action_max_value, best_action_max_index = torch.max(action, 1)
        best_action_string = space_game.action_list[best_action_max_index.item()]
        _, done = space_game.step(best_action_string)

        next_state = space_game.get_screen()
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        state = next_state
        if done:
            break


if __name__ == "__main__":
    if TRAINING:
        train()
    else:
        simulate()

