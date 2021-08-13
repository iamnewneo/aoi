import json

import cv2
import numpy as np
import torch
from torch import nn

from train_cnn import CNNModel

N_EPOCHS = 10
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNModel(nn.Module):
    def __init__(self, in_d=2, out_d=2, hidden_d=8, num_hidden=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=in_d, hidden_size=hidden_d, num_layers=num_hidden)
        self.fc = nn.Linear(hidden_d, out_d)

    def forward(self, x, h0):
        r, h = self.rnn(x, h0)
        y = self.fc(r)  # no activation on the output
        return y, h


class RNNTrainer:
    def __init__(self) -> None:
        self.cnn_model = CNNModel()
        self.cnn_model.load_state_dict(
            torch.load("./models/cnn_model.pth", map_location=torch.device(DEVICE))
        )
        self.cnn_model.eval()
        self.tankrc = []  # Tank Reloading Positions
        self.tankrc1 = []  # Tank Reloading Tank Moving Positions

        self.tank_moving_panels = None
        self.tank_reloading_panels = None
        self.tmtr_panels = None

        with open("./data/tank_moving_panels.json", "r") as f:
            self.tank_moving_panels = json.load(f)
        with open("./data/tank_reloading_panels.json", "r") as f:
            self.tank_reloading_panels = json.load(f)
        with open("./data/tmtr_panels.json", "r") as f:
            self.tmtr_panels = json.load(f)

        self.tank_moving_panels = {
            int(k): v for k, v in self.tank_moving_panels.items()
        }
        self.tank_reloading_panels = {
            int(k): v for k, v in self.tank_reloading_panels.items()
        }
        self.tmtr_panels = {int(k): v for k, v in self.tmtr_panels.items()}

    def preprocess_data(self):
        print("Started Preprocessing")

        cap = cv2.VideoCapture("./data/atari_2_vehicles.avi")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        keys = list(self.tank_reloading_panels.keys())
        keys_tmtr = list(self.tmtr_panels.keys())

        uframes = []

        def val_pos(array):
            value = np.sum(array)
            position = np.unravel_index(np.argmax(array), array.shape)
            scaled_position = (position[1] * 73, position[0] * 98)
            return value, scaled_position

        ret, frame = cap.read()

        while ret:
            uframes = []
            flag = False
            uframes.append(cv2.resize(frame, (100, 100)))
            img = uframes[0]
            data3 = []
            data3.append(
                np.array(np.transpose(np.asarray(img), (2, 0, 1)), dtype=np.float32)
            )
            y3 = np.array(data3)
            x_1 = torch.from_numpy(y3)
            output = self.cnn_model(x_1.reshape(1, 3, 100, 100))[0].detach().numpy()

            for key in keys:
                output_tankr = output[0][key]
                val_tankr, pos_tankr = val_pos(output_tankr)
                if val_tankr > self.tank_reloading_panels[key]:
                    # print("tank reloading present")
                    self.tankrc.append(pos_tankr)
                    flag = True
                    break
            if flag == False:
                for key in keys_tmtr:
                    output_tankr = output[0][key]
                    val_tankr, pos_tankr = val_pos(output_tankr)
                    if val_tankr > self.tmtr_panels[key]:
                        # print("tank reloading present")
                        self.tankrc1.append(pos_tankr)
                        break
            ret, frame = cap.read()

        # release the cap object
        cap.release()

        s1 = torch.tensor(self.tankrc, dtype=torch.float)[:, None, :]
        s2 = torch.tensor(self.tankrc1, dtype=torch.float)[:, None, :]

        s1 = s1.to(DEVICE)
        s2 = s1.to(DEVICE)

        x = torch.cat((s1, s2), dim=0)

        mu = x.mean(dim=0)
        sig = x.std(dim=0)
        sequences = [
            (s1 - mu) / sig,
            (s2 - mu) / sig,
        ]  # pythonic list to hold sequences of un-even length
        return sequences

    def train_lstm(self):
        sequences = self.preprocess_data()
        print("Started Training")
        in_d = sequences[0].shape[-1]
        out_d = in_d
        hidden_d = 8
        num_hidden = 1
        print(in_d, out_d, hidden_d, num_hidden)
        rnn = RNNModel(in_d, out_d, hidden_d, num_hidden)
        rnn.to(DEVICE)
        loss = []
        criterion = nn.MSELoss()
        opt = torch.optim.SGD(rnn.parameters(), lr=LR)
        for epoch in range(N_EPOCHS):
            for s_i, s in enumerate(sequences):
                pred, _ = rnn(
                    s[:-1, ...], torch.zeros(num_hidden, 1, hidden_d, dtype=torch.float)
                )  # predict next step, init hidden state to zero at the begining of the sequence
                err = criterion(pred, s[1:, ...])  # predict next step for each step
                opt.zero_grad()
                err.backward()
                opt.step()
                loss.append(err.item())

                print(
                    f"Epoch: {epoch+1}/{N_EPOCHS}. Sequence:{s_i}. Loss: {err.item()}"
                )
        torch.save(rnn.state_dict(), "./models/lstm_model.pth")


def main():
    rnn_trainer = RNNTrainer()
    rnn_trainer.train_lstm()


if __name__ == "__main__":
    main()
