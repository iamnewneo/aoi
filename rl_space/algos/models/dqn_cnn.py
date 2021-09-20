import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class DQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, self.num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return (
            self.features(autograd.Variable(torch.zeros(1, *self.input_shape)))
            .view(1, -1)
            .size(1)
        )
