# Feedforward Neural Network in C++

This project implements a feedforward neural network in C++ with two hidden layers. It includes functionalities for network initialization, training, and prediction using the backpropagation algorithm.

## Features

Multi-Layer Architecture: Implements a feedforward neural network with an input layer, two hidden layers, and an output layer.

ReLU Activation: Uses the Rectified Linear Unit (ReLU) activation function for hidden layers.

Softmax Output: Applies the Softmax function to the output layer for multi-class classification.

Backpropagation Training: Trains the network using the backpropagation algorithm with gradient descent.

Weight Initialization: Initializes weights using a normal distribution with a He initialization strategy.

Safe Vector Allocation: Includes a utility function to prevent excessively large vector allocations that could lead to errors.

The program will train the neural network on a predefined dataset and then output the predictions for a test input.

### Compilation

Compile the code using a C++ compiler that supports C++11 or later. For example, using g++:

g++ -std=c++11 neuralnet.cpp -o neuralnet

### Execution

Run the compiled executable: ./neuralnet

The program will train the neural network on a predefined dataset and then output the predictions for a test input.

### Code Structure

`Network` struct: Defines the structure to hold the weights and biases of the neural network layers.

`safe_vector` template function: A utility function to safely allocate vectors, preventing potential overflow issues.

`relu` and `relu_derivative` functions: Implement the ReLU activation function and its derivative.

`softmax` function: Implements the Softmax function for the output layer.

`initialize_weights` function: Initializes the weights of the network layers using a normal distribution.

`initialize_network` function: Initializes the entire network with the specified layer sizes.

`train_network` function: Implements the backpropagation algorithm to train the neural network.

`predict` function: Performs a forward pass through the network to generate predictions.

`prepare_data` function: Sets up the training data with sample inputs and targets.

`main` function: The main entry point of the program, which sets up the network, trains it, and tests it with a sample input.

### Network Architecture

The network consists of three layers:

1.  Input Layer: Size is determined by the input data.
2.  Hidden Layer 1: Size is configurable (e.g., 20 neurons).
3.  Hidden Layer 2: Size is configurable (e.g., 15 neurons).
4.  Output Layer: Size is determined by the number of classes in the classification problem.

### Weight Initialization

Weights are initialized using a normal distribution with a mean of 0 and a standard deviation calculated based on the He initialization strategy:

std = sqrt(2.0 / fan_in)

where `fan_in` is the number of inputs to the layer.

### Training

The network is trained using the backpropagation algorithm.  The key steps include:

1.  Forward Pass: Propagate the input through the network to compute the output.
2.  Error Calculation: Compute the error between the predicted output and the target.
3.  Backward Pass: Propagate the error backward through the network to compute the gradients.
4.  Weight Update: Update the weights and biases using gradient descent.

### Activation Functions

ReLU (Rectified Linear Unit): Applied to the hidden layers.
    
  f(x) = max(0, x)
    
  Softmax: Applied to the output layer to obtain probability distributions for multi-class classification.

### Example

The `main` function demonstrates how to:

Initialize the neural network.

  Prepare the training data.

  Train the network using the `train_network` function.

  Make predictions on a test input using the `predict` function.

### Future Enhancements

  Mini-Batch Gradient Descent: Implement mini-batch gradient descent to improve training efficiency.
  
  More Activation Functions: Add support for other activation functions like sigmoid and tanh.
  
  Regularization: Implement regularization techniques like L1 or L2 regularization to prevent overfitting.
  
  Optimization Algorithms: Implement more advanced optimization algorithms like Adam or RMSprop.
  
  Data Loading: Add functionality to load data from files.
  
  Evaluation Metrics: Include evaluation metrics such as accuracy, precision, recall, and F1-score.

### License

GNU AFFERO GENERAL PUBLIC LICENSE
