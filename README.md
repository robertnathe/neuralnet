# Introduction

This is a Python implementation of a neural network with two hidden layers. The network is trained using stochastic gradient descent with mini-batches and is designed to classify input data into one of three categories.

## Network Architecture

The network consists of the following layers:

* Input layer with 3 neurons
* Hidden layer 1 with 20 neurons
* Hidden layer 2 with 15 neurons
* Output layer with 3 neurons

The network uses the ReLU activation function for the hidden layers and the softmax activation function for the output layer.

## Training

The network is trained using stochastic gradient descent with mini-batches. The training data consists of 10 input samples, each with 3 features, and their corresponding target values.

## Usage

To use this code, simply run the `neuralnet.py` file. The network will be trained on the provided training data and then tested on a sample input.

## Functions

The code consists of the following functions:

* `relu(x)`: The ReLU activation function.
* `relu_derivative(x)`: The derivative of the ReLU activation function.
* `softmax(z)`: The softmax activation function.
* `initialize_weights(size, fan_in)`: Initializes weights using He initialization.
* `initialize_network(input_size, hidden1_size, hidden2_size, output_size, network)`: Initializes all weights and biases for the neural network.
* `train_network(inputs, targets, hidden1_size, hidden2_size, output_size, network, epochs, learning_rate, batch_size=5)`: Trains the neural network using stochastic gradient descent with mini-batches.
* `predict(network, input_data, hidden1_size, hidden2_size, output_size)`: Performs a forward pass through the network to get predictions for a single input.
* `prepare_data()`: Prepares a sample dataset for training and testing the network.

## Requirements

This code requires the following libraries:

* `math`
* `random`
* `numpy`

## Notes

This is a basic implementation of a neural network and is intended for educational purposes only. The network architecture and training parameters may need to be adjusted for specific use cases.

### License

GNU AFFERO GENERAL PUBLIC LICENSE
