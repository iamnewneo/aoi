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
                u_frames = pickle.load(f)
            data_exists = True

        if not data_exists:
            print("U_Frames not present, recreating.")
            u_frames = []
            vidcap = cv2.VideoCapture("./data/atari_2_vehicles.avi")
            success, image = vidcap.read()
            while success:
                u_frames.append(image)
                success, image = vidcap.read()

            for i, frame in enumerate(u_frames):
                u_frames[i] = cv2.resize(frame, (100, 100))

            # release the cap object
            vidcap.release()
            with open("./data/u_frames.pkl", "wb") as f:
                pickle.dump(u_frames, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.u_frames = u_frames

        data1 = []
        data2 = []
        for i in range(SAMPLES):
            data1.append(
                np.array(
                    np.transpose(np.asarray(u_frames[i % len(u_frames)]), (2, 0, 1)),
                    dtype=np.float32,
                )
            )
            data2.append(
                np.array(
                    np.transpose(
                        np.asarray(u_frames[(i + 1) % len(u_frames)]), (2, 0, 1)
                    ),
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

    def train_cnn(self):
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
            data3.append(
                np.array(np.transpose(np.asarray(img), (2, 0, 1)), dtype=np.float32)
            )
            y3 = np.array(data3)
            x_1 = torch.from_numpy(y3)
            x_1 = x_1.to(DEVICE)
            output = (
                self.model(x_1[0].reshape(1, 3, 100, 100))[0].detach().cpu().numpy()
            )
            return output

        for i in range(num_channels):

            output = image_to_numpy(self.u_frames[tmtr])
            tmtr_panel = output[0][i]
            val_tmtr, pos_tmtr = val_pos(tmtr_panel)

            output = image_to_numpy(self.u_frames[tm])
            tm_panel = output[0][i]
            val_tm, pos_tm = val_pos(tm_panel)

            output = image_to_numpy(self.u_frames[tr])
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


class CNNTest:
    def __init__(self) -> None:
        self.model = CNNModel()
        self.model.load_state_dict(
            torch.load("./models/cnn_model.pth", map_location=torch.device(DEVICE))
        )

    def test_cnn(self):
        cap = cv2.VideoCapture("./data/atari_2_vehicles.avi")

        fps = int(cap.get(1))

        frame_width = int(cap.get(3))

        frame_height = int(cap.get(4))
        tankrc = []
        tankm = []

        # tank_moving_panels = {
        #     0: 2939.852294921875,
        #     1: 3651.266845703125,
        #     2: 3918.51708984375,
        #     4: 2535.77880859375,
        #     6: 2757.001953125,
        #     8: 1897.1790771484375,
        #     12: 2739.127685546875,
        #     15: 2944.8779296875,
        # }

        # tank_reloading_panels = {
        #     3: 2042.2137451171875,
        #     5: 2765.12841796875,
        #     7: 3176.161376953125,
        #     10: 2638.602783203125,
        #     11: 3490.5673828125,
        #     14: 1660.1922607421875,
        # }

        # tmtr_panels = {9: 2719.021484375, 13: 3320.355712890625}

        with open("./data/tank_moving_panels.json", "r") as f:
            tank_moving_panels = json.load(f)
        with open("./data/tank_reloading_panels.json", "r") as f:
            tank_reloading_panels = json.load(f)
        with open("./data/tmtr_panels.json", "r") as f:
            tmtr_panels = json.load(f)

        tank_moving_panels = {int(k): v for k, v in tank_moving_panels.items()}
        tank_reloading_panels = {int(k): v for k, v in tank_reloading_panels.items()}
        tmtr_panels = {int(k): v for k, v in tmtr_panels.items()}

        keys = list(tank_reloading_panels.keys())
        keys_tmtr = list(tmtr_panels.keys())
        keys_tm = list(tank_moving_panels.keys())

        uframes = []
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

        out = cv2.VideoWriter(
            "Two Aliens.avi",
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            10,
            (frame_width, frame_height),
        )

        def val_pos(array):
            value = np.sum(array)
            position = np.unravel_index(np.argmax(array), array.shape)
            # print(f"position: {position}")
            return value, position

        ret, frame = cap.read()

        while ret:
            # Capture frames in the video
            # describe the type of font
            # to be used.
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Use putText() method for
            # inserting text on video
            # print(np.shape(frame))
            uframes = []
            uframes.append(cv2.resize(frame, (100, 100)))
            img = uframes[0]

            data3 = []
            data3.append(
                np.array(np.transpose(np.asarray(img), (2, 0, 1)), dtype=np.float32)
            )
            y3 = np.array(data3)
            x_1 = torch.from_numpy(y3)
            output = self.model(x_1.reshape(1, 3, 100, 100))[0].detach().numpy()

            for key in keys:
                output_tankr = output[0][key]
                val_tankr, pos_tankr = val_pos(output_tankr)
                if val_tankr > tank_reloading_panels[key]:
                    # print("tank reloading present")
                    tankrc.append(pos_tankr)
                    # l.append(i)
                    # Display the resulting frame
                    cv2.putText(
                        frame,
                        "Fire Here",
                        (73 * pos_tankr[1], 98 * pos_tankr[0]),
                        font,
                        2,
                        (0, 165, 255),
                        5,
                        cv2.LINE_4,
                    )
                    cv2.putText(
                        frame,
                        "Take Action: The Vehicle is going to fire",
                        (50, 50),
                        font,
                        2,
                        (0, 0, 255),
                        5,
                        cv2.LINE_4,
                    )

            for key in keys_tmtr:
                output_tankr = output[0][key]
                val_tankr, pos_tankr = val_pos(output_tankr)
                if val_tankr > tmtr_panels[key]:
                    # print("tank reloading present")
                    tankrc.append(pos_tankr)
                    # l.append(i)
                    # Display the resulting frame
                    cv2.putText(
                        frame,
                        "Fire Here",
                        (73 * pos_tankr[1], 98 * pos_tankr[0]),
                        font,
                        2,
                        (0, 165, 255),
                        5,
                        cv2.LINE_4,
                    )
                    cv2.putText(
                        frame,
                        "Take Action: The Vehicle is going to fire",
                        (50, 50),
                        font,
                        2,
                        (0, 0, 255),
                        5,
                        cv2.LINE_4,
                    )

            for key in keys_tm:
                output_tankr = output[0][key]
                val_tankr, pos_tankr = val_pos(output_tankr)
                if val_tankr > tank_moving_panels[key]:
                    # print("tank reloading present")
                    tankm.append(pos_tankr)
                    # l.append(i)
                    # Display the resulting frame

            out.write(frame)
            ret, frame = cap.read()


def main():
    cnn_trainer = CNNTrainer()
    model = cnn_trainer.train_cnn()
    cnn_trainer.post_processing()


def test_cnn():
    cnn_tester = CNNTest()
    cnn_tester.test_cnn()


if __name__ == "__main__":
    main()
    # test_cnn()
