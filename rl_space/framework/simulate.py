from collections import defaultdict

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

from cnn_pipeline import CNNModel, NNLabel, NNLabelTrainer, reverse_x_y
from space_invader import Action, SpaceInvaderGame

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESIZE_WIDTH = 100
RESIZE_HEIGHT = 100

SHOOT_X_DELTA = 2

plt.ion()

# Label 2: Moving. Label 1: Relaoding
class MetricTracker:
    def __init__(self, game) -> None:
        self.game = game
        self.curr_step_metric = {}
        self.total_steps = 0
        self.step_metric_history = defaultdict(list)
        self.step_metric_running_history = defaultdict(list)
        self.n_steps_mean = None

        # Visualization
        self.acc_fig, self.acc_ax = plt.subplots(figsize=(5, 3))
        (self.acc_plot,) = self.acc_ax.plot([], [])
        self.acc_ax.set_xlabel("Time Step")
        # self.acc_ax.set_y_limit(0, 1)

        # self.msde_fig = plt.figure(figsize=(10, 6))

    def get_dist_error_and_accuraacy(self, pred_enemy_positions):
        # Dist Error
        sum_dict = {}
        count_disct = {}

        # Acc Error
        total_correct_count = 0
        for pred_enemy in pred_enemy_positions:
            tagged_enemy_idx = pred_enemy["tagged_enemy_idx"]
            if tagged_enemy_idx not in sum_dict:
                sum_dict[tagged_enemy_idx] = pred_enemy["dist"]
            else:
                sum_dict[tagged_enemy_idx] += pred_enemy["dist"]

            if tagged_enemy_idx not in count_disct:
                count_disct[tagged_enemy_idx] = 1
            else:
                count_disct[tagged_enemy_idx] += 1
            if pred_enemy["label"] == pred_enemy["tagged_enemy_label"]:
                total_correct_count += 1

        total_sum = sum(sum_dict.values())
        total_count = sum(count_disct.values())

        if len(self.game.enemies) > 0 and len(pred_enemy_positions) == 0:
            return 1000, 0
        mean_dist_error = total_sum / total_count
        accuracy = total_correct_count / len(pred_enemy_positions)
        return mean_dist_error, accuracy

    def calculate_metrics(self, pred_enemy_positions):
        # Assign one dot/pred enemy to actual enemy
        actual_enemies = self.game.enemies
        for pred_enemy in pred_enemy_positions:
            min_dist = 999999
            for idx, act_enemy in enumerate(actual_enemies):
                ab_dist = abs(pred_enemy["x"] - act_enemy.x)
                if ab_dist < min_dist:
                    min_dist = ab_dist
                    pred_enemy["tagged_enemy"] = act_enemy
                    pred_enemy["tagged_enemy_idx"] = idx
                    pred_enemy["dist"] = min_dist
                    pred_enemy["tagged_enemy_label"] = (
                        1 if act_enemy.reloading == True else 2
                    )

        mean_dist_error, accuracy = self.get_dist_error_and_accuraacy(
            pred_enemy_positions
        )

        self.curr_step_metric["mean_dist_error"] = mean_dist_error
        self.curr_step_metric["accuracy"] = accuracy

        self.step_metric_history["mean_dist_error"].append(mean_dist_error)
        self.step_metric_history["accuracy"].append(accuracy)

    def plot_acc(self):
        acc_xs = range(len(self.step_metric_running_history["accuracy"]))
        acc_ys = self.step_metric_running_history["accuracy"]
        plt.plot(acc_xs, acc_ys)
        plt.xlabel("Time Step")
        plt.ylabel("Enemy Status Accuracy")
        plt.draw()
        plt.pause(0.0001)
        self.acc_fig.clear()

        # acc_xs = range(len(self.step_metric_running_history["accuracy"]))
        # acc_ys = self.step_metric_running_history["accuracy"]
        # self.acc_plot.set_xdata(acc_xs)
        # self.acc_plot.set_ydata(acc_ys)
        # plt.pause(0.05)

        # acc_xs = range(len(self.step_metric_running_history["accuracy"]))
        # acc_ys = self.step_metric_running_history["accuracy"]
        # self.acc_ax.clear()
        # plt.plot(acc_xs, acc_ys)
        # self.acc_fig.canvas.draw()
        # plt.show()

    # def plot_metrics(self):
    #     acc_xs = range(len(self.step_metric_running_history["accuracy"]))
    #     acc_ys = self.step_metric_running_history["accuracy"]
    #     self.acc_plot.set_data(acc_xs, acc_ys)
    #     # return self.acc_plot,

    #     # # re-drawing the figure
    #     # fig.canvas.draw()

    #     # # to flush the GUI events
    #     # fig.canvas.flush_events()
    #     # time.sleep(0.1)
    #     # self.acc_ax.clear()
    #     # self.acc_ax.plot(acc_xs, acc_ys)

    def visualize_metrics(self):
        mse_history = self.step_metric_history["mean_dist_error"]
        acc_history = self.step_metric_history["accuracy"]

        if self.n_steps_mean is not None:
            mse_history = mse_history[-1 * self.n_steps_mean :]
            acc_history = acc_history[-1 * self.n_steps_mean :]
        mse = sum(mse_history) / len(mse_history)
        acc = sum(acc_history) / len(acc_history)

        self.step_metric_running_history["mean_dist_error"].append(mse)
        self.step_metric_running_history["accuracy"].append(acc)

        print(f"MSE: {mse:.2f}. Acc: {acc:.2f}")
        self.plot_acc()


class SimulateGame:
    def __init__(self) -> None:
        self.game = SpaceInvaderGame()
        self.nn_trainer = NNLabelTrainer()

        self.label_model = NNLabel()
        self.label_model.load_state_dict(
            torch.load(
                "./models/label_model_150.pth", map_location=torch.device(DEVICE)
            )
        )
        self.label_model.to(DEVICE)
        self.label_model.eval()

    def preprocess_input(self, frame):
        # frame = Image.fromarray(frame)
        frame = frame[:, :, ::-1]
        image = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        return image

    def get_enemy_positions(self, preds):
        enemies = []
        image = preds.reshape(18, 18)
        for i in range(18):
            for j in range(18):
                label = image[i][j]
                if label > 0:
                    x, y = reverse_x_y(i, j)
                    enemy_metadata = {}
                    enemy_metadata["x"] = x
                    enemy_metadata["y"] = y
                    enemy_metadata["label"] = label
                    enemies.append(enemy_metadata)
        return enemies

    def get_action(self, enemy_positions):
        player_x = self.game.player.x
        temp_dist = 999999
        action = Action.DO_NOTHING
        for enemy in enemy_positions:
            # Check for reloading: FIX THIS LATER
            # At the moment move on any label
            abs_dist = abs(enemy["x"] - player_x)

            # Directly Shoot if within range
            if abs_dist <= SHOOT_X_DELTA:
                return Action.UP_ARROW_KEY_PRESSED
            if abs_dist < temp_dist:
                temp_dist = abs_dist
                if (enemy["x"] - player_x) >= 0:
                    action = Action.RIGHT_ARROW_KEY_PRESSED
                else:
                    action = Action.LEFT_ARROW_KEY_PRESSED
        return action

    def get_best_action(self, frame_tensor, metric_tracker):
        image_channels = self.nn_trainer.preprocess_frame(frame_tensor)
        image_channels = image_channels.to(DEVICE)
        outputs = self.label_model(image_channels)
        _, preds = torch.max(outputs.data, 1)
        enemy_positions = self.get_enemy_positions(preds)
        for enemy_pos in enemy_positions:
            self.game.mark_enemy_position(enemy_pos)

        metric_tracker.calculate_metrics(enemy_positions)
        action = self.get_action(enemy_positions)
        return action

    def simulate(self):
        self.game.init_game()
        metric_tracker = MetricTracker(self.game)
        for _ in range(100000):
            screen = self.game.get_screen()
            frame_tensor = self.preprocess_input(screen)
            action = self.get_best_action(frame_tensor, metric_tracker)
            self.game.step(action)

            metric_tracker.visualize_metrics()


def main():
    simulator = SimulateGame()
    simulator.simulate()


if __name__ == "__main__":
    main()

