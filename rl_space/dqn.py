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
from space_invader import SpaceInvaderGame
import torchvision.transforms as T
from PIL import Image

torch.set_flush_denormal(True)

MEM_CAPACITY = 10000
LR = 1e-3
N_EPISODES = 1000
MAX_STEPS = 10000
BATCH_SIZE = 512
GAMMA = 0.999
TARGET_UPDATE = 10

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# game constants
WIDTH = 256
HEIGHT = 256

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resize = T.Compose(
    [
        T.ToPILImage(),
        T.Grayscale(num_output_channels=1),
        T.Resize(size=(HEIGHT, WIDTH), interpolation=Image.CUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ]
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NNModel(nn.Module):
    def __init__(self, height, width, num_action):
        super(NNModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, padding=1, stride=1
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, padding=1, stride=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        linear_input_size = 2032128
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
        # x = F.relu(self.conv3(x))
        # x = self.bn3(x)
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
        if len(memory) < BATCH_SIZE:
            return

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


def tranform_state(state):
    state = state.astype(np.uint8)
    state = resize(state)
    return state


def train():
    space_game = SpaceInvaderGame()
    memory = ReplayMemory(capacity=MEM_CAPACITY)
    num_action = len(space_game.action_list)
    agent = SpaceInvaderDQN(height=HEIGHT, width=WIDTH, num_action=num_action)
    for ep_i in range(N_EPISODES):
        total_episode_reward = 0
        space_game.init_game()
        state = space_game.get_screen()
        state = tranform_state(state)
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).unsqueeze(0).to(device)
        state = state.float()
        for step in count():
            start_time = time.time()
            best_action = agent.select_action(state, step)
            best_action_string = space_game.action_list[best_action.item()]
            step_reward = space_game.step(best_action_string)
            total_episode_reward += step_reward
            next_state = space_game.get_screen()
            next_state = tranform_state(next_state)
            reward = torch.tensor([step_reward], device=device)
            next_state = next_state.to(device)
            next_state = next_state.float()
            memory.push(state, best_action, next_state, reward)
            state = next_state
            opt_start = time.time()
            agent.optimize_model(memory)
            # print(
            #     f"Episode: {ep_i}. Step: {step}. Last Best Action: {best_action_string}."
            #     f"Reward: {step_reward}. Time: {time.time() - start_time}. "
            #     f"Opt Time: {time.time() - opt_start}"
            # )
            if step >= MAX_STEPS:
                break
        if ep_i % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Episode: {ep_i}. Total Reward: {total_episode_reward}")


if __name__ == "__main__":
    train()
