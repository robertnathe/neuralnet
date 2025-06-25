import math
import random
import numpy as np

# Define the Network structure using a class
class Network:
    """
    Represents the neural network structure, holding weights and biases for
    three layers: Input -> Hidden1, Hidden1 -> Hidden2, and Hidden2 -> Output.
    """
    def __init__(self):
        # Layer 1: Input -> Hidden1
        self.input_weights = []  # size = input_size * hidden1_size
        self.hidden1_biases = [] # size = hidden1_size

        # Layer 2: Hidden1 -> Hidden2
        self.hidden1_weights = [] # size = hidden1_size * hidden2_size
        self.hidden2_biases = []  # size = hidden2_size

        # Layer 3: Hidden2 -> Output
        self.hidden2_weights = [] # size = hidden2_size * output_size
        self.output_biases = []   # size = output_size

# Activation functions

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.
    Returns x if x > 0, otherwise returns 0.
    """
    return max(0.0, x)

def relu_derivative(x):
    """
    Derivative of the ReLU activation function.
    Returns 1 if x > 0, otherwise returns 0.
    Note: For x=0, the derivative is undefined. We use 0 as a common practice.
    """
    return 1.0 if x > 0 else 0.0

def softmax(z):
    """
    Softmax activation function.
    Converts a vector of numbers into a vector of probabilities that sum to 1.
    Uses a numerical stability trick by subtracting the maximum value from z.
    """
    # Convert list to numpy array for element-wise operations
    z_np = np.array(z)
    
    # Subtract the maximum value from z for numerical stability
    # This prevents exp() from overflowing for large z values
    max_z = np.max(z_np)
    exp_z = np.exp(z_np - max_z)
    
    sum_exp_z = np.sum(exp_z)
    
    # Handle case where sum_exp_z might be extremely small or zero
    if sum_exp_z == 0:
        return [0.0] * len(z) # Return zeros or handle as an error
    
    # Calculate softmax probabilities
    sm = exp_z / sum_exp_z
    return sm.tolist() # Convert back to list for consistency with C++ vector

def initialize_weights(size, fan_in):
    """
    Initializes weights using He initialization (suited for ReLU).
    Weights are drawn from a normal distribution with mean 0 and
    standard deviation sqrt(2 / fan_in).
    """
    std_dev = math.sqrt(2.0 / fan_in)
    # Using numpy's random normal for efficient generation
    return np.random.normal(0.0, std_dev, size).tolist()

def initialize_network(input_size, hidden1_size, hidden2_size, output_size, network):
    """
    Initializes all weights and biases for the neural network.
    Weights are initialized using He initialization, biases are initialized to 0.1.
    """
    # Initialize input -> hidden1 layer
    network.input_weights = initialize_weights(input_size * hidden1_size, input_size)
    network.hidden1_biases = [0.1] * hidden1_size

    # Initialize hidden1 -> hidden2 layer
    network.hidden1_weights = initialize_weights(hidden1_size * hidden2_size, hidden1_size)
    network.hidden2_biases = [0.1] * hidden2_size

    # Initialize hidden2 -> output layer
    network.hidden2_weights = initialize_weights(hidden2_size * output_size, hidden2_size)
    network.output_biases = [0.1] * output_size

def train_network(inputs, targets, hidden1_size, hidden2_size, output_size,
                  network, epochs, learning_rate, batch_size=5):
    """
    Trains the neural network using stochastic gradient descent with mini-batches.
    Performs forward pass, calculates error, backpropagates deltas, and updates
    weights and biases based on accumulated gradients from the batch.
    """
    input_size = len(inputs[0])
    num_samples = len(inputs)

    # Pre-allocate buffers for activations (forward pass)
    hidden1_outputs = [0.0] * hidden1_size
    hidden2_outputs = [0.0] * hidden2_size
    output_values = [0.0] * output_size # Pre-softmax outputs

    # Pre-allocate buffers for deltas (backpropagation)
    output_deltas = [0.0] * output_size
    hidden2_deltas = [0.0] * hidden2_size
    hidden1_deltas = [0.0] * hidden1_size

    # Initialize gradient accumulators for batch updates
    accum_input_weights = [0.0] * len(network.input_weights)
    accum_hidden1_biases = [0.0] * len(network.hidden1_biases)
    accum_hidden1_weights = [0.0] * len(network.hidden1_weights)
    accum_hidden2_biases = [0.0] * len(network.hidden2_biases)
    accum_hidden2_weights = [0.0] * len(network.hidden2_weights)
    accum_output_biases = [0.0] * len(network.output_biases)

    # Create indices for shuffling the dataset
    indices = list(range(num_samples))

    for epoch in range(epochs+1):
        total_error = 0.0
        random.shuffle(indices) # Shuffle data indices at the start of each epoch

        for batch_start in range(0, num_samples, batch_size):
            # Reset gradient accumulators for the new batch
            accum_input_weights = [0.0] * len(network.input_weights)
            accum_hidden1_biases = [0.0] * len(network.hidden1_biases)
            accum_hidden1_weights = [0.0] * len(network.hidden1_weights)
            accum_hidden2_biases = [0.0] * len(network.hidden2_biases)
            accum_hidden2_weights = [0.0] * len(network.hidden2_weights)
            accum_output_biases = [0.0] * len(network.output_biases)

            batch_end = min(batch_start + batch_size, num_samples)
            actual_batch_size = batch_end - batch_start

            for idx_in_batch in range(batch_start, batch_end):
                sample_idx = indices[idx_in_batch]
                current_input = inputs[sample_idx]
                current_target = targets[sample_idx]

                # Forward pass ---------------------------------------------------
                # Input -> Hidden1
                for h1 in range(hidden1_size):
                    sum_val = 0.0
                    for j in range(input_size):
                        # C++: network.input_weights[h1 * input_size + j]
                        sum_val += current_input[j] * network.input_weights[h1 * input_size + j]
                    hidden1_outputs[h1] = relu(sum_val + network.hidden1_biases[h1])
                
                # Hidden1 -> Hidden2
                for h2 in range(hidden2_size):
                    sum_val = 0.0
                    for h1 in range(hidden1_size):
                        # C++: network.hidden1_weights[h2 * hidden1_size + h1]
                        sum_val += hidden1_outputs[h1] * network.hidden1_weights[h2 * hidden1_size + h1]
                    hidden2_outputs[h2] = relu(sum_val + network.hidden2_biases[h2])
                
                # Hidden2 -> Output
                for o in range(output_size):
                    sum_val = 0.0
                    for h2 in range(hidden2_size):
                        # C++: network.hidden2_weights[o * hidden2_size + h2]
                        sum_val += hidden2_outputs[h2] * network.hidden2_weights[o * hidden2_size + h2]
                    output_values[o] = sum_val + network.output_biases[o]
                
                predictions = softmax(output_values)
                
                # Error calculation (Squared error for tracking, cross-entropy for deltas)
                # The output_deltas here are for cross-entropy loss derivative with softmax
                # For classification, error is (prediction - target), which is simplified
                # for softmax + cross-entropy to (prediction - one_hot_target).
                
                # Create one-hot encoded target vector
                target_vec = [0.0] * output_size
                target_vec[current_target] = 1.0

                for o in range(output_size):
                    error = predictions[o] - target_vec[o] # Difference for cross-entropy
                    output_deltas[o] = error # Delta for output layer (softmax + cross-entropy)
                    # For total_error tracking, we use squared error as in C++ example
                    total_error += (error * error) 
                
                # Backpropagation ------------------------------------------------
                # Calculate hidden2 deltas
                for h2 in range(hidden2_size):
                    error_sum = 0.0
                    for o in range(output_size):
                        # C++: network.hidden2_weights[o * hidden2_size + h2]
                        error_sum += output_deltas[o] * network.hidden2_weights[o * hidden2_size + h2]
                    hidden2_deltas[h2] = error_sum * relu_derivative(hidden2_outputs[h2])
                
                # Calculate hidden1 deltas
                for h1 in range(hidden1_size):
                    error_sum = 0.0
                    for h2 in range(hidden2_size):
                        # C++: network.hidden1_weights[h2 * hidden1_size + h1]
                        error_sum += hidden2_deltas[h2] * network.hidden1_weights[h2 * hidden1_size + h1]
                    hidden1_deltas[h1] = error_sum * relu_derivative(hidden1_outputs[h1])
                
                # Accumulate gradients for the current sample --------------------
                # Output layer gradients
                for o in range(output_size):
                    for h2 in range(hidden2_size):
                        # C++: accum_hidden2_weights[o * hidden2_size + h2]
                        accum_hidden2_weights[o * hidden2_size + h2] += \
                            output_deltas[o] * hidden2_outputs[h2]
                    accum_output_biases[o] += output_deltas[o]
                
                # Hidden2 layer gradients
                for h2 in range(hidden2_size):
                    for h1 in range(hidden1_size):
                        # C++: accum_hidden1_weights[h2 * hidden1_size + h1]
                        accum_hidden1_weights[h2 * hidden1_size + h1] += \
                            hidden2_deltas[h2] * hidden1_outputs[h1]
                    accum_hidden2_biases[h2] += hidden2_deltas[h2]
                
                # Hidden1 layer gradients
                for h1 in range(hidden1_size):
                    for j in range(input_size):
                        # C++: accum_input_weights[h1 * input_size + j]
                        accum_input_weights[h1 * input_size + j] += \
                            hidden1_deltas[h1] * current_input[j]
                    accum_hidden1_biases[h1] += hidden1_deltas[h1]

            # Apply batch updates after processing all samples in the batch
            lr_factor = learning_rate / actual_batch_size
            
            # Update hidden2->output weights and biases
            for o in range(output_size):
                for h2 in range(hidden2_size):
                    # C++: network.hidden2_weights[o * hidden2_size + h2]
                    network.hidden2_weights[o * hidden2_size + h2] -= \
                        lr_factor * accum_hidden2_weights[o * hidden2_size + h2]
                network.output_biases[o] -= lr_factor * accum_output_biases[o]
            
            # Update hidden1->hidden2 weights and biases
            for h2 in range(hidden2_size):
                for h1 in range(hidden1_size):
                    # C++: network.hidden1_weights[h2 * hidden1_size + h1]
                    network.hidden1_weights[h2 * hidden1_size + h1] -= \
                        lr_factor * accum_hidden1_weights[h2 * hidden1_size + h1]
                network.hidden2_biases[h2] -= lr_factor * accum_hidden2_biases[h2]
            
            # Update input->hidden1 weights and biases
            for h1 in range(hidden1_size):
                for j in range(input_size):
                    # C++: network.input_weights[h1 * input_size + j]
                    network.input_weights[h1 * input_size + j] -= \
                        lr_factor * accum_input_weights[h1 * input_size + j]
                network.hidden1_biases[h1] -= lr_factor * accum_hidden1_biases[h1]
        
        # Print average error for the epoch
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg Error: {total_error / num_samples:.6f}")


def predict(network, input_data, hidden1_size, hidden2_size, output_size):
    """
    Performs a forward pass through the network to get predictions for a single input.
    """
    input_size = len(input_data)

    hidden1_outputs = [0.0] * hidden1_size
    hidden2_outputs = [0.0] * hidden2_size
    output_values = [0.0] * output_size
    
    # Input -> Hidden1
    for h1 in range(hidden1_size):
        sum_val = 0.0
        for j in range(input_size):
            sum_val += input_data[j] * network.input_weights[h1 * input_size + j]
        hidden1_outputs[h1] = relu(sum_val + network.hidden1_biases[h1])
    
    # Hidden1 -> Hidden2
    for h2 in range(hidden2_size):
        sum_val = 0.0
        for h1 in range(hidden1_size):
            sum_val += hidden1_outputs[h1] * network.hidden1_weights[h2 * hidden1_size + h1]
        hidden2_outputs[h2] = relu(sum_val + network.hidden2_biases[h2])
    
    # Hidden2 -> Output
    for o in range(output_size):
        sum_val = 0.0
        for h2 in range(hidden2_size):
            sum_val += hidden2_outputs[h2] * network.hidden2_weights[o * hidden2_size + h2]
        output_values[o] = sum_val + network.output_biases[o]
    
    return softmax(output_values)

def prepare_data():
    """
    Prepares a sample dataset for training and testing the network.
    Returns inputs (list of lists) and targets (list of integers).
    """
    inputs = [
        [1.0/12, 2.0/12, 3.0/12], [1.0/12, 2.0/12, 4.0/12],
        [1.0/12, 2.0/12, 5.0/12], [1.0/12, 2.0/12, 6.0/12],
        [1.0/12, 2.0/12, 7.0/12], [1.0/12, 2.0/12, 8.0/12],
        [1.0/12, 2.0/12, 9.0/12], [1.0/12, 2.0/12, 10.0/12],
        [1.0/12, 2.0/12, 11.0/12], [1.0/12, 2.0/12, 1.0]
    ]
    targets = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    return inputs, targets

# Main execution block
if __name__ == "__main__":
    # Define network parameters
    input_size = 3
    hidden1_size = 20
    hidden2_size = 15
    output_size = 3
    epochs = 300
    learning_rate = 0.02
    batch_size = 5

    # Prepare training data
    inputs, targets = prepare_data()

    # Create and initialize the network
    network = Network()
    initialize_network(input_size, hidden1_size, hidden2_size, output_size, network)

    print("Starting network training...")
    # Train the network
    train_network(inputs, targets, hidden1_size, hidden2_size, output_size,
                  network, epochs, learning_rate, batch_size)

    # Test the network with a sample input
    test_input = [1.0/12, 2.0/12, 3.0/12]
    probs = predict(network, test_input, hidden1_size, hidden2_size, output_size)
    
    print("\nTest input predictions:")
    print([f"{p:.6f}" for p in probs]) # Format output to 6 decimal places for readability
