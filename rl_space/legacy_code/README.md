## General Info
The folder in this repository includes code for replicating the Video Game code. The code is refactored in such a way that it is easier to understand and make any changes. Training on GPU is also supported for CNN and LSTM now.

#### The folder mainly consists of 4 files:
1. train_cnn.py - Used to train CNN model [H-Score] to identify the label of frame and location of enemy
2. train_lstm.py - Used to train LSTM model to predict next position of the enemy
3. train_driver.py - Single script which combines train_cnn.py and train_lstm.py. 
4. simulate_game.py - Once the CNN and LSTM models are trained, we can simulate the game using this script

#### Important Folders:
1. data - Contains data required for training and inference (e.g Video file and dictionary of the channel to label mapping)
2. models - Contains PyTorch model files, used in the simulation
3. res - Contains resources like images, sound, etc required to run the game

## Requirements
Python3, Pytorch, Pygame, Numpy

## Setup
To train the model (if model files are not present)
```
python train_driver.py
```
To simulate the game:
```
python simulate_game.py
```

## Note
None of the hyperparamters are changed from the original. I just removed unncessary code and refactored it to make it more readable. Current simulation is the best I could get. Hardcoding of channel to class label mapping is removed, they are now stored in a dictionary/json in data folder.