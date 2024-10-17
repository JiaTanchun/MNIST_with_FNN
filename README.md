# MNIST_with_FNN
# Fully Connected Neural Network with BatchNorm and Dropout on MNIST

This repository contains a PyTorch implementation of a fully connected neural network (FNN) with batch normalization and dropout, trained on the MNIST dataset.

## Requirements

To run this code, you need to have the following packages installed:

- `torch`
- `torchvision`
- `matplotlib`

You can install them using the following command:

```bash
pip install torch torchvision matplotlib
```

## Code Structure

- `FNN` class: Defines the architecture of the fully connected neural network with a hidden layer, batch normalization, and dropout.
- `train_loader` and `test_loader`: Load the MNIST dataset with transformations such as normalization.
- Training loop: Trains the neural network using the Adam optimizer and CrossEntropyLoss.
- Test evaluation: Evaluates the model performance on the test dataset.
- Plotting: Displays training and test loss curves for visualization.


## Hyperparameters

The following hyperparameters are used in the code, and you can adjust them to experiment with different configurations:

- `input_size`: The size of the input layer, set to `28 * 28` (the size of flattened MNIST images).
- `hidden_size`: The number of neurons in the hidden layer, set to `256`.
- `num_classes`: The number of output classes, set to `10` (for the digits 0-9).
- `num_epochs`: The number of epochs to train the model, set to `10`.
- `batch_size`: The number of samples per batch for training, set to `64`.
- `learning_rate`: The learning rate for the Adam optimizer, set to `0.001`.
- `dropout_prob`: The dropout probability for regularization, set to `0.5`.

You can modify these values directly in the code to see how they affect the model's performance.
