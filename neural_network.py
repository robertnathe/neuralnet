import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # Initialize weights and biases
        self.input_weights = self.he_initialization((input_size, hidden1_size))
        self.hidden1_biases = np.full(hidden1_size, 0.1)
        
        self.hidden1_weights = self.he_initialization((hidden1_size, hidden2_size))
        self.hidden2_biases = np.full(hidden2_size, 0.1)
        
        self.hidden2_weights = self.he_initialization((hidden2_size, output_size))
        self.output_biases = np.full(output_size, 0.1)
    
    @staticmethod
    def he_initialization(shape):
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, size=shape)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def forward(self, x):
        # Input to Hidden1
        self.hidden1_output = self.relu(np.dot(x, self.input_weights) + self.hidden1_biases)
        # Hidden1 to Hidden2
        self.hidden2_output = self.relu(np.dot(self.hidden1_output, self.hidden1_weights) + self.hidden2_biases)
        # Hidden2 to Output
        output = np.dot(self.hidden2_output, self.hidden2_weights) + self.output_biases
        return self.softmax(output)
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                # Forward pass
                x = X[i]
                target = np.zeros_like(self.output_biases)
                target[y[i]] = 1
                
                predictions = self.forward(x)
                
                # Calculate error
                error = target - predictions
                total_error += np.sum(error ** 2)
                
                # Backpropagation
                # Output layer delta
                output_delta = error
                
                # Hidden2 delta
                hidden2_error = np.dot(output_delta, self.hidden2_weights.T)
                hidden2_delta = hidden2_error * self.relu_derivative(self.hidden2_output)
                
                # Hidden1 delta
                hidden1_error = np.dot(hidden2_delta, self.hidden1_weights.T)
                hidden1_delta = hidden1_error * self.relu_derivative(self.hidden1_output)
                
                # Update weights and biases (fixed dimension issues)
                # Hidden2 -> Output
                self.hidden2_weights += learning_rate * np.outer(self.hidden2_output, output_delta)
                self.output_biases += learning_rate * output_delta
                
                # Hidden1 -> Hidden2
                self.hidden1_weights += learning_rate * np.outer(self.hidden1_output, hidden2_delta)
                self.hidden2_biases += learning_rate * hidden2_delta
                
                # Input -> Hidden1
                self.input_weights += learning_rate * np.outer(x, hidden1_delta)
                self.hidden1_biases += learning_rate * hidden1_delta
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Avg Error: {total_error / len(X)}")

def prepare_data():
    inputs = [
        [1/12, 2/12, 3/12], [1/12, 2/12, 4/12],
        [1/12, 2/12, 5/12], [1/12, 2/12, 6/12],
        [1/12, 2/12, 7/12], [1/12, 2/12, 8/12],
        [1/12, 2/12, 9/12], [1/12, 2/12, 10/12],
        [1/12, 2/12, 11/12], [1/12, 2/12, 1.0]
    ]
    outputs = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    return np.array(inputs), np.array(outputs)

if __name__ == "__main__":
    # Hyperparameters
    INPUT_SIZE = 3
    HIDDEN1_SIZE = 20
    HIDDEN2_SIZE = 15
    OUTPUT_SIZE = 3
    EPOCHS = 300
    LEARNING_RATE = 0.02
    
    # Prepare data
    X, y = prepare_data()
    
    # Initialize network
    nn = NeuralNetwork(INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE)
    
    # Train network
    nn.train(X, y, EPOCHS, LEARNING_RATE)
    
    # Test prediction
    test_input = np.array([1/12, 2/12, 3/12])
    probs = nn.forward(test_input)
    print("\nTest input predictions:")
    print(probs)