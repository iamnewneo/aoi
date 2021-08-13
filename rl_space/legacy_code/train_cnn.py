import json
import os
import pickle

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 50
N_EPOCHS = 200
LR = 1e-5
SAMPLES = 20000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=10, stride=1)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=5)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(5184, 15)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        im_out = out
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.sigmoid1(out)
        return im_out, out


def h_score(fx, gy):
    fx = fx - fx.mean(0)
    gy = gy - gy.mean(0)
    Nsamples = fx.size(0)
    covf = torch.matmul(fx.t(), fx) / Nsamples
    covg = torch.matmul(gy.t(), gy) / Nsamples
    h = -2 * torch.mean((fx * gy).sum(1)) + (covf * covg).sum()
    return h


class CNNTrainer:
    def __init__(self) -> None:
        self.u_frames = None
        self.model = None

    def get_dataloader(self):
        print("Loading Data")
        data_exists = False
        if os.path.isfile("./data/u_frames.pkl"):
            with open("./data/u_frames.pkl", "rb") as f:
                uframes = pickle.load(f)
            data_exists = True

        if not data_exists:
            print("Uframes not present, recreating.")
            uframes = []
            vidcap = cv2.VideoCapture("./data/atari_2_vehicles.avi")
            success, image = vidcap.read()
            while success:
                uframes.append(image)
                success, image = vidcap.read()

            for i, frame in enumerate(uframes):
                uframes[i] = cv2.resize(frame, (100, 100))

            # release the cap object
            vidcap.release()
            with open("./data/u_frames.pkl", "wb") as f:
                pickle.dump(uframes, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.u_frames = uframes

        data1 = []
        data2 = []
        for i in range(SAMPLES):
            data1.append(
                np.array(
                    np.transpose(np.asarray(uframes[i % len(uframes)]), (2, 0, 1)),
                    dtype=np.float32,
                )
            )
            data2.append(
                np.array(
                    np.transpose(np.asarray(uframes[(i + 1) % len(uframes)]), (2, 0, 1)),
                    dtype=np.float32,
                )
            )

        y0 = np.asarray(data1)
        y1 = np.asarray(data2)

        x = torch.from_numpy(y0)
        y = torch.from_numpy(y1)
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        torch_dataset = TensorDataset(x, y)

        loader = DataLoader(
            dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )
        return loader

    def train_cnnn(self):
        model = CNNModel()
        model.to(device=DEVICE)
        data_loader = self.get_dataloader()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

        print("Training Model")
        for epoch in range(N_EPOCHS):
            for x, y in data_loader:
                optimizer.zero_grad()
                loss = h_score(model(x)[1], model(y)[1])
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch + 1}/{N_EPOCHS}], loss:{loss.item():.4f}")

        print("Saving CNN Model")
        torch.save(model.state_dict(), "./models/cnn_model.pth")
        self.model = model
        return model

    def post_processing(self):
        print("Started Post Processing")
        tank_moving_panels = {}
        tank_reloading_panels = {}
        tmtr_panels = {}

        num_channels = 16
        buffer = 5

        # Hard Coded Frame Numbers
        tmtr = 38
        tm = 4
        tr = 46

        def val_pos(array):
            value = np.sum(array)
            position = np.unravel_index(np.argmax(array), array.shape)
            return value, position

        def image_to_numpy(img):
            data3 = []
            data3.append(np.array(np.transpose(np.asarray(img), (2, 0, 1)), dtype=np.float32))
            y3 = np.array(data3)
            x_1 = torch.from_numpy(y3)
            output = self.model(x_1[0].reshape(1, 3, 100, 100))[0].detach().numpy()
            return output

        for i in range(num_channels):

            output = image_to_numpy(self.uframes[tmtr])
            tmtr_panel = output[0][i]
            val_tmtr, pos_tmtr = val_pos(tmtr_panel)

            output = image_to_numpy(self.uframes[tm])
            tm_panel = output[0][i]
            val_tm, pos_tm = val_pos(tm_panel)

            output = image_to_numpy(self.uframes[tr])
            tr_panel = output[0][i]
            val_tr, pos_tr = val_pos(tr_panel)

            if val_tm > max(val_tmtr, val_tr):
                tank_moving_panels[i] = val_tm - buffer
            elif val_tr > max(val_tmtr, val_tm):
                tank_reloading_panels[i] = val_tr - buffer
            elif val_tmtr > max(val_tr, val_tm):
                tmtr_panels[i] = val_tmtr - buffer

        print("Saving Model Channel Data")
        with open("./data/tank_moving_panels.json", "w") as f:
            json.dump(tank_moving_panels, f)
        with open("./data/tank_reloading_panels.json", "w") as f:
            json.dump(tank_reloading_panels, f)
        with open("./data/tmtr_panels.json", "w") as f:
            json.dump(tmtr_panels, f)


def main():
    cnn_trainer = CNNTrainer()
    model = cnn_trainer.train_cnnn()
    cnn_trainer.post_processing()


if __name__ == "__main__":
    main()
