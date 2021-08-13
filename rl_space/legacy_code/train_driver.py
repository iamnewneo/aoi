from train_cnn import CNNTrainer
from train_lstm import RNNTrainer


def main():
    cnn_trainer = CNNTrainer()
    model = cnn_trainer.train_cnn()
    cnn_trainer.post_processing()

    rnn_trainer = RNNTrainer()
    rnn_trainer.train_lstm()


if __name__ == "__main__":
    main()
