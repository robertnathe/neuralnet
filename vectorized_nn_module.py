import numpy as np
from dataclasses import dataclass

# --- ⚡ Activation Functions ---

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit activation function.
    Leverages NumPy's optimized element-wise maximum operation."""
    return np.maximum(0.0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU function.
    Returns 1 for x > 0 and 0 for x <= 0.
    The boolean comparison is efficiently converted to float (0s and 1s)."""
    # Explicitly use float64 for consistency with other NumPy operations,
    # and to maintain precision in gradient calculations.
    return (x > 0).astype(np.float64) 

def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax activation function with numerical stability for a batch.
    Input `z` can be of shape (O_size, batch_size).
    Subtracts the maximum logit per sample to prevent exp(z) overflow."""
    
    # Max is taken per sample (column) and `keepdims=True` ensures
    # the result `z_max` has shape (1, batch_size) for correct broadcasting.
    z_max = np.max(z, axis=0, keepdims=True)
    e_z = np.exp(z - z_max)
    
    # Sum is taken per sample (column) for normalization.
    # `keepdims=True` maintains (1, batch_size) shape for correct division broadcasting.
    sum_e_z = np.sum(e_z, axis=0, keepdims=True)
    return e_z / sum_e_z

# --- 🏗️ Network Architecture and Initialization ---

@dataclass(slots=True) # Added slots=True for minor memory optimization
class Network:
    """A simple data structure to hold the network's trainable parameters.
    Weights: W_i is a matrix (fan_out, fan_in)
    Biases: b_i is a column vector (fan_out, 1)"""
    W1: np.ndarray # (H1_size, I_size)
    b1: np.ndarray # (H1_size, 1)
    W2: np.ndarray # (H2_size, H1_size)
    b2: np.ndarray # (H2_size, 1)
    W3: np.ndarray # (O_size, H2_size)
    b3: np.ndarray # (O_size, 1)

def initialize_weights(fan_in: int, fan_out: int) -> np.ndarray:
    """
    Implements He initialization: W ~ N(0, sqrt(2/fan_in)).
    Returns a weight matrix of shape (fan_out, fan_in).
    `np.random.normal` is highly efficient for generating random numbers.
    It defaults to float64, ensuring type consistency.
    """
    std_dev = np.sqrt(2.0 / fan_in) # Ensure float division
    return np.random.normal(loc=0.0, scale=std_dev, size=(fan_out, fan_in))

def initialize_network(input_size: int, hidden1_size: int, hidden2_size: int, output_size: int) -> Network:
    """Initializes weights using He initialization and biases to 0.1."""
    
    # Layer 1 (Input -> Hidden 1)
    W1 = initialize_weights(input_size, hidden1_size)
    # Biases initialized to 0.1, shape (H_size, 1) for correct broadcasting.
    b1 = np.full((hidden1_size, 1), 0.1, dtype=np.float64) 
    
    # Layer 2 (Hidden 1 -> Hidden 2)
    W2 = initialize_weights(hidden1_size, hidden2_size)
    b2 = np.full((hidden2_size, 1), 0.1, dtype=np.float64)
    
    # Layer 3 (Hidden 2 -> Output)
    W3 = initialize_weights(hidden2_size, output_size)
    b3 = np.full((output_size, 1), 0.1, dtype=np.float64)
    
    return Network(W1, b1, W2, b2, W3, b3)

# --- 🧠 Training and Prediction ---

def forward_pass(network: Network, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Performs the forward pass for a batch of input vectors X.
    X must be a matrix of shape (input_size, batch_size).
    Returns the final prediction (probabilities) and a cache of intermediate
    activations and pre-activations (Z values) needed for backpropagation.
    """
    
    # Layer 1: Input -> Hidden 1
    # Matrix multiplication (`@`) is highly optimized by NumPy, leveraging BLAS.
    # Bias addition uses NumPy's broadcasting for efficiency.
    Z1 = network.W1 @ X + network.b1
    A1 = relu(Z1)
    
    # Layer 2: Hidden 1 -> Hidden 2
    Z2 = network.W2 @ A1 + network.b2
    A2 = relu(Z2)
    
    # Layer 3: Hidden 2 -> Output
    Z3 = network.W3 @ A2 + network.b3
    A3 = softmax(Z3) # Final prediction (probabilities)

    # Cache intermediate values for Backpropagation.
    # Storing A3 is useful for loss calculation and might be used in more complex backward passes.
    cache: dict[str, np.ndarray] = {
        'X': X, 'A1': A1, 'Z1': Z1, 
        'A2': A2, 'Z2': Z2, 'A3': A3
    }
    
    return A3, cache

def cross_entropy_loss(A3: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculates the cross-entropy loss for a batch.
    A3: (output_size, batch_size) - predicted probabilities
    Y: (output_size, batch_size) - one-hot encoded true labels
    """
    # Clip A3 to prevent log(0) and numerical instability.
    # `np.clip` is an efficient vectorized operation.
    A3_clipped = np.clip(A3, epsilon, 1.0 - epsilon)
    
    # Loss for each sample: -sum(Y_i * log(A3_i))
    # `np.sum` and `np.mean` are highly optimized for aggregate calculations.
    loss_per_sample = -np.sum(Y * np.log(A3_clipped), axis=0) # Sum over classes for each sample
    return np.mean(loss_per_sample) # Average loss over the batch

def backpropagation(network: Network, cache: dict[str, np.ndarray], Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the backpropagation step for a mini-batch.
    Returns the gradients for W1, b1, W2, b2, W3, b3.
    Y must be the one-hot encoded targets as a matrix of shape (output_size, batch_size).
    """
    X, A1, Z1, A2, Z2, A3 = cache['X'], cache['A1'], cache['Z1'], cache['A2'], cache['Z2'], cache['A3']
    
    batch_size = X.shape[1]
    
    # Layer 3 (Output Layer)
    # The delta for Softmax + Cross-Entropy is simplified to (Prediction - Target).
    dZ3 = A3 - Y # (O_size, batch_size)
    
    # Gradients are averaged over the batch size for consistent learning rates
    # regardless of batch size.
    dW3 = (dZ3 @ A2.T) / batch_size # (O_size, H2_size)
    # Summing gradients for biases across samples (axis=1) and keeping dimensions for updates.
    db3 = np.sum(dZ3, axis=1, keepdims=True) / batch_size # (O_size, 1)
    
    # Layer 2 (Hidden Layer 2)
    dA2 = network.W3.T @ dZ3 # (H2_size, batch_size)
    # Element-wise multiplication for ReLU derivative.
    dZ2 = dA2 * relu_derivative(Z2) # (H2_size, batch_size)
    dW2 = (dZ2 @ A1.T) / batch_size # (H2_size, H1_size)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / batch_size # (H2_size, 1)
    
    # Layer 1 (Hidden Layer 1)
    dA1 = network.W2.T @ dZ2 # (H1_size, batch_size)
    dZ1 = dA1 * relu_derivative(Z1) # (H1_size, batch_size)
    dW1 = (dZ1 @ X.T) / batch_size # (H1_size, I_size)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / batch_size # (H1_size, 1)

    return dW1, db1, dW2, db2, dW3, db3

def train_network(network: Network, X_train: np.ndarray, Y_train: np.ndarray, epochs: int, learning_rate: float, batch_size: int):
    """
    Trains the network using Mini-Batch Stochastic Gradient Descent (SGD).
    X_train: (input_size, num_samples)
    Y_train: (output_size, num_samples)
    """
    num_samples = X_train.shape[1]
    
    for epoch in range(epochs):
        # 1. Shuffle data indices at the start of each epoch for randomness.
        # `np.arange` and `np.random.shuffle` are efficient for this purpose.
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # 2. Iterate over mini-batches
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            
            # Select batch data using advanced indexing. This creates copies.
            X_batch = X_train[:, batch_indices]
            Y_batch = Y_train[:, batch_indices]
            
            # 3. Forward Pass for the mini-batch
            A3_batch, cache = forward_pass(network, X_batch)
            
            # 4. Backpropagation to get gradients
            dW1, db1, dW2, db2, dW3, db3 = backpropagation(network, cache, Y_batch)
            
            # 5. Weight Update (SGD step)
            # In-place subtraction operators (`-=`) modify the existing NumPy arrays,
            # which is efficient and avoids creating new array objects for each update.
            network.W1 -= learning_rate * dW1
            network.b1 -= learning_rate * db1
            network.W2 -= learning_rate * dW2
            network.b2 -= learning_rate * db2
            network.W3 -= learning_rate * dW3
            network.b3 -= learning_rate * db3
            
        # Optional: Print loss periodically for monitoring.
        # This calculates loss on the last batch of the epoch, which is a quick estimate
        # and avoids an extra full forward pass over the entire dataset.
        if (epoch + 1) % 100 == 0:
            current_loss = cross_entropy_loss(A3_batch, Y_batch)
            print(f"Epoch {epoch + 1}/{epochs} completed. Loss: {current_loss:.4f}")

def predict(network: Network, X: np.ndarray) -> np.ndarray:
    """
    Executes the forward pass for one or more inputs and returns the probability distribution.
    X must be a matrix of shape (input_size, num_samples_to_predict).
    """
    prediction_probs, _ = forward_pass(network, X)
    return prediction_probs

# --- ⚙️ Execution and Data Preparation ---

def prepare_data(num_samples: int, input_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic data for a simple classification problem.
    Returns X_data of shape (input_size, num_samples) and Y_data of shape (output_size, num_samples).
    """
    # Generate random input vectors (input_size, num_samples)
    # `np.random.rand` defaults to float64.
    X_data = np.random.rand(input_size, num_samples) * 10.0
    
    # Simple rule to generate a one-hot target class for each sample:
    # Sum of features for each sample, then modulo output_size.
    # `target_classes_idx` will have shape (1, num_samples).
    target_classes_idx = (np.sum(X_data, axis=0, keepdims=True) % output_size).astype(int)
    
    # Create one-hot encoded labels efficiently using advanced NumPy indexing.
    Y_data = np.zeros((output_size, num_samples), dtype=np.float64)
    # Sets Y_data[row_indices, col_indices] to 1.0.
    # `target_classes_idx.squeeze()` provides row indices by removing the single dimension,
    # potentially creating a view instead of a copy, which can be more memory efficient
    # than `flatten()` which always returns a copy.
    Y_data[target_classes_idx.squeeze(), np.arange(num_samples)] = 1.0
        
    return X_data, Y_data

if __name__ == "__main__":
    # --- Parameters ---
    INPUT_SIZE = 3
    HIDDEN1_SIZE = 20
    HIDDEN2_SIZE = 15
    OUTPUT_SIZE = 3
    NUM_SAMPLES = 1000
    
    # Tuned parameters for balanced demonstration and performance:
    # Reduced epochs and increased batch size to manage execution time efficiently.
    EPOCHS = 500  
    LEARNING_RATE = 0.01
    BATCH_SIZE = 32

    # Set seed for reproducibility of random operations across runs.
    np.random.seed(42)

    print("--- Neural Network Simulation Initialized ---")
    
    # 1. Prepare Data
    X_train, Y_train = prepare_data(NUM_SAMPLES, INPUT_SIZE, OUTPUT_SIZE)
    print(f"Dataset created: {NUM_SAMPLES} samples. X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    
    # 2. Initialize Network
    nn = initialize_network(INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE)
    print("Network initialized with He weights and 0.1 biases.")

    # 3. Train Network
    print(f"Starting training for {EPOCHS} epochs (LR={LEARNING_RATE}, Batch Size={BATCH_SIZE})...")
    # The Network instance is mutable, allowing direct in-place updates to its NumPy array fields.
    train_network(nn, X_train, Y_train, EPOCHS, LEARNING_RATE, BATCH_SIZE)
    print("Training complete.")

    # 4. Sample Prediction
    # Ensure `sample_input` is a column vector of shape (input_size, 1)
    # for consistent forward pass behavior with batch processing, even for a single sample.
    sample_input = np.array([[1.5], [2.5], [3.5]], dtype=np.float64)
    sample_sum = np.sum(sample_input)
    sample_target_class = int(sample_sum % OUTPUT_SIZE)
    
    print("\n--- Sample Prediction ---")
    # Transpose for cleaner printing of a single column vector.
    print(f"Input: \n{sample_input.T}") 
    print(f"Expected Class (based on sum % {OUTPUT_SIZE}): {sample_target_class} (from sum {sample_sum:.2f})")
    
    # Run prediction
    prediction_probs = predict(nn, sample_input)
    predicted_class = np.argmax(prediction_probs)
    
    # Transpose for cleaner printing of the output probabilities.
    print(f"Output Probabilities: \n{prediction_probs.T}") 
    print(f"Predicted Class: {predicted_class}")
    print(f"Prediction {'Successful' if predicted_class == sample_target_class else 'Failed'}")
