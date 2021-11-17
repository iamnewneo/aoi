import os
import pickle
import time
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

BATCH_SIZE = 8
# N_EPOCHS = 5
N_EPOCHS = 1000
LR = 1e-5
SAMPLES = 20000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESIZE_WIDTH = 100
RESIZE_HEIGHT = 100
N_LABELS = 3  # 1. Moving, 2. Reloading, 3. Nothing


def reverse_x_y(x, y):
    x = (x * 100) // 18
    x = x * 6
    y = (y * 100) // 18
    y = y * 6
    return x, y


def h_score(fx, gy):
    fx = fx - fx.mean(0)
    gy = gy - gy.mean(0)
    Nsamples = fx.size(0)
    covf = torch.matmul(fx.t(), fx) / Nsamples
    covg = torch.matmul(gy.t(), gy) / Nsamples
    h = -2 * torch.mean((fx * gy).sum(1)) + (covf * covg).sum()
    return h


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
                u_frames[i] = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

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
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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


class NNDataset(Dataset):
    def __init__(self, csv_path, transform=None) -> None:
        super().__init__()
        df = pd.read_csv(csv_path)
        df["id"] = df["index"]
        df = df.set_index("index")
        df["label"] = df["enemy_reloading"].apply(lambda x: 1 if x else 2)
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        image_path = row["img_path"]
        image_label = row["label"]  # reloading true
        # image_encoded_label = row["encoded_label"]
        enemy_x = row["enemy_x"]
        enemy_y = row["enemy_y"]
        # Swap because screen x and array x is interchanged: axis vs row
        enemy_x, enemy_y = enemy_y, enemy_x
        image = cv2.imread(image_path)
        image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)
        sample = {
            "image": image,
            "enemy_x": enemy_x,
            "enemy_y": enemy_y,
            "label": image_label,
            "id": row["id"],
            # "encoded_label": image_encoded_label,
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


class NNLabel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, N_LABELS)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out


class NNLabelTrainer:
    def __init__(self) -> None:
        self.model = NNLabel()
        self.model.to(DEVICE)
        self.hscore_model = CNNModel()
        self.hscore_model.load_state_dict(
            torch.load("./models/cnn_model.pth", map_location=torch.device(DEVICE))
        )
        self.hscore_model.eval()
        self.dataset = NNDataset("./data/train.csv")
        self.dataloader = DataLoader(
            dataset=self.dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        self.data_len = len(self.dataset)
        self.optimizer = torch.optim.Adam(self.hscore_model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss()

    def preprocess_data(self, data):
        images = data["image"]
        # Adjusting Resize 600->100
        enemy_x = data["enemy_x"] // 6
        enemy_y = data["enemy_y"] // 6
        # Adjusting for image to channel
        enemy_x = (enemy_x * 18) / 100
        enemy_y = (enemy_y * 18) / 100

        enemy_x = enemy_x.int()
        enemy_y = enemy_y.int()
        label = data["label"]
        inp_images = []
        inp_labels = []
        with torch.no_grad():
            self.hscore_model.eval()
            channels = self.hscore_model(images)
            cnn_channels = channels[0]
            for idx, image_channels in enumerate(cnn_channels):
                dummy_labels = torch.zeros((18, 18))
                dummy_labels[enemy_x[idx]][enemy_y[idx]] = label[idx]
                temp_image_channels = []
                temp_image_labels = []
                for i in range(18):
                    for j in range(18):
                        temp_image_channels.append(image_channels[:, i, j])
                        temp_image_labels.append(dummy_labels[i][j].long())

                inp_images += temp_image_channels
                inp_labels += temp_image_labels

            # inp_images = torch.cat(cnn_channels.unbind())

            inp_images_tensor = torch.stack(inp_images)
            inp_labels_tensor = torch.stack(inp_labels)
        # counter = Counter(inp_labels_tensor.tolist())
        # print(f"Input Distribution: {counter}")
        return inp_images_tensor, inp_labels_tensor

    def fit(self, model, dataloader):
        model.train()
        train_running_loss = 0.0
        train_running_correct = 0
        total_training_examples = 0
        for i, data in tqdm(
            enumerate(dataloader),
            total=int(self.data_len / dataloader.batch_size),
            disable=True,
        ):
            images, targets = self.preprocess_data(data)
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            targets = targets.long()
            total_training_examples += targets.shape[0]
            self.optimizer.zero_grad()
            outputs = model(images)
            loss = self.criterion(outputs, targets)
            train_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            # counter = Counter(preds.tolist())
            # print(f"Output Distribution: {counter}")
            train_running_correct += (preds == targets).sum().item()
            loss.backward()
            self.optimizer.step()
        train_loss = train_running_loss / self.data_len
        train_accuracy = 100.0 * train_running_correct / total_training_examples

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")

        return train_loss, train_accuracy

    def train(self):
        train_loss, train_accuracy = [], []
        # val_loss, val_accuracy = [], []
        start = time.time()
        for epoch in range(N_EPOCHS):
            print(f"Epoch {epoch+1} of {N_EPOCHS}")
            train_epoch_loss, train_epoch_accuracy = self.fit(
                self.model, self.dataloader
            )
            # val_epoch_loss, val_epoch_accuracy = validate(model, testloader)
            train_loss.append(train_epoch_loss)
            train_accuracy.append(train_epoch_accuracy)
            # val_loss.append(val_epoch_loss)
            # val_accuracy.append(val_epoch_accuracy)
        end = time.time()
        print(f"{(end-start)/60:.3f} minutes")

        # # Plots
        # # accuracy plots
        # plt.figure(figsize=(10, 7))
        # plt.plot(train_accuracy, color="green", label="train accuracy")
        # # plt.plot(val_accuracy, color="blue", label="validataion accuracy")
        # plt.xlabel("Epochs")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.savefig("../outputs/accuracy.png")
        # # loss plots
        # plt.figure(figsize=(10, 7))
        # plt.plot(train_loss, color="orange", label="train loss")
        # # plt.plot(val_loss, color="red", label="validataion loss")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig("../outputs/loss.png")

        # save the model to disk
        print("Saving model...")
        torch.save(self.model.state_dict(), "./models/label_model.pth")


class TrainPipeline:
    def __init__(self) -> None:
        pass

    def train(self):
        cnn_trainer = CNNTrainer()
        nn_label_trainer = NNLabelTrainer()


class TestPipeline:
    def __init__(self) -> None:
        self.cnn_model = CNNModel()
        self.cnn_model.load_state_dict(
            torch.load("./models/cnn_model.pth", map_location=torch.device(DEVICE))
        )
        self.cnn_model.eval()
        self.label_model = NNLabel()
        self.label_model.load_state_dict(
            torch.load("./models/label_model.pth", map_location=torch.device(DEVICE))
        )
        self.label_model.eval()

    def visualize_test(self, preds, data):
        offset = 18 * 18
        n_unique_images = preds.shape[0] // offset
        ids = data["id"]
        for idx in range(n_unique_images):
            test_actual_image = cv2.imread(f"./data/test_images/{ids[idx]}.jpg")
            image = preds[idx * offset : (idx + 1) * offset]
            image = image.reshape(18, 18)
            for i in range(18):
                for j in range(18):
                    label = image[i][j]
                    if label > 0:
                        # print(f"Image: {idx} Label: {label}. Location: {i},{j}")
                        x, y = reverse_x_y(i, j)
                        if label == 1:
                            # BGR
                            color = (0, 0, 255)
                        else:
                            color = (255, 0, 0)
                        test_actual_image = cv2.circle(
                            test_actual_image,
                            (y, x),
                            radius=5,
                            color=color,
                            thickness=-1,
                        )
            cv2.imwrite(f"./data/output_images/{ids[idx]}.jpg", test_actual_image)

    def test_label(self):
        with torch.no_grad():
            nn_trainer = NNLabelTrainer()
            dataset = NNDataset("./data/test.csv")
            dataloader = DataLoader(
                dataset=dataset, batch_size=BATCH_SIZE, shuffle=True
            )
            for _, data in tqdm(enumerate(dataloader)):
                images, targets = nn_trainer.preprocess_data(data)
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = self.label_model(images)
                _, preds = torch.max(outputs.data, 1)
                # print(preds.shape)
                self.visualize_test(preds, data)


def main():
    nn_trainer = NNLabelTrainer()
    nn_trainer.train()

    test_pipeline = TestPipeline()
    test_pipeline.test_label()


if __name__ == "__main__":
    main()
