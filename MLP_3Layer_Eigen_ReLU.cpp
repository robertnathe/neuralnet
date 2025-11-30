// g++ -std=c++20 -O3 -Wall -Wextra -I/usr/include/eigen3 -I/usr/include testing_20.cpp -o testing_20 -lboost_system -lboost_filesystem -lboost_math_c99 -lcurl
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

// Include the Eigen library headers
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// --- 1. Activation Functions ---

/**
 * @brief Rectified Linear Unit (ReLU) activation function.
 * @param x Input value.
 * @return max(0, x)
 */
double relu(double x) {
    return max(0.0, x);
}

/**
 * @brief Derivative of the ReLU function.
 * @param x Input value.
 * @return 1 if x > 0, 0 otherwise.
 */
double relu_derivative(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}

/**
 * @brief Softmax function for the output layer.
 * @param z Vector of raw scores (logits).
 * @return Vector of probabilities.
 */
VectorXd softmax(const VectorXd& z) {
    // Subtract max for numerical stability (log-sum-exp trick for Softmax)
    double max_z = z.maxCoeff();
    VectorXd exp_z = (z.array() - max_z).exp();
    return exp_z / exp_z.sum();
}

// --- 2. Network Structure ---

struct Network {
    // Weights and biases are now stored as Eigen Matrices and Vectors
    // W1: Hidden1 Weights (hidden_size_1 x input_size)
    MatrixXd W1; 
    // B1: Hidden1 Biases (hidden_size_1 x 1)
    VectorXd B1; 
    
    // W2: Hidden2 Weights (hidden_size_2 x hidden_size_1)
    MatrixXd W2; 
    // B2: Hidden2 Biases (hidden_size_2 x 1)
    VectorXd B2;
    
    // W3: Output Weights (output_size x hidden_size_2)
    MatrixXd W3;
    // B3: Output Biases (output_size x 1)
    VectorXd B3;
};

// --- 3. Initialization ---

/**
 * @brief Initialize weights using He initialization (suitable for ReLU).
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix (fan_in).
 * @return Initialized Eigen Matrix.
 */
MatrixXd initialize_weights(int rows, int cols) {
    // He Initialization: variance = sqrt(2 / fan_in)
    double std_dev = sqrt(2.0 / cols);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0, std_dev);
    
    MatrixXd W(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            W(i, j) = d(gen);
        }
    }
    return W;
}

/**
 * @brief Initialize the entire network's weights and biases.
 * @param input_size Size of the input layer.
 * @param hidden_size_1 Size of the first hidden layer.
 * @param hidden_size_2 Size of the second hidden layer.
 * @param output_size Size of the output layer.
 * @return Initialized Network struct.
 */
Network initialize_network(int input_size, int hidden_size_1, int hidden_size_2, int output_size) {
    Network net;

    // Weights: using He initialization
    net.W1 = initialize_weights(hidden_size_1, input_size);
    net.W2 = initialize_weights(hidden_size_2, hidden_size_1);
    net.W3 = initialize_weights(output_size, hidden_size_2);

    // Biases: initialized to a small constant (0.1)
    net.B1 = VectorXd::Constant(hidden_size_1, 1, 0.1);
    net.B2 = VectorXd::Constant(hidden_size_2, 1, 0.1);
    net.B3 = VectorXd::Constant(output_size, 1, 0.1);

    return net;
}

// --- 4. Forward Propagation ---

/**
 * @brief Performs the forward pass through the network.
 * @param net The neural network structure.
 * @param input_data The input vector.
 * @param A1_out Reference to store the output of Hidden Layer 1 (ReLU activated).
 * @param A2_out Reference to store the output of Hidden Layer 2 (ReLU activated).
 * @param Z3_out Reference to store the logits (pre-Softmax) of the Output Layer.
 * @return The final probability vector (Softmax output).
 */
VectorXd forward_pass(const Network& net, const VectorXd& input_data, 
                      VectorXd& Z1_out, VectorXd& A1_out, 
                      VectorXd& Z2_out, VectorXd& A2_out, 
                      VectorXd& Z3_out) 
{
    // Layer 1: Input -> Hidden1 (ReLU)
    // Z1 = W1 * X + B1
    Z1_out = net.W1 * input_data + net.B1;
    // A1 = ReLU(Z1)
    A1_out = Z1_out.unaryExpr([](double z){ return relu(z); });

    // Layer 2: Hidden1 -> Hidden2 (ReLU)
    // Z2 = W2 * A1 + B2
    Z2_out = net.W2 * A1_out + net.B2;
    // A2 = ReLU(Z2)
    A2_out = Z2_out.unaryExpr([](double z){ return relu(z); });

    // Layer 3: Hidden2 -> Output (Softmax)
    // Z3 = W3 * A2 + B3
    Z3_out = net.W3 * A2_out + net.B3;
    // A3 = Softmax(Z3)
    return softmax(Z3_out);
}

// --- 5. Training ---

/**
 * @brief Trains the network using forward and backpropagation.
 * @param net The neural network structure (modified in place).
 * @param inputs_data Matrix of all input vectors (InputSize x NumSamples).
 * @param targets_data Matrix of all one-hot target vectors (OutputSize x NumSamples).
 * @param epochs Number of training iterations.
 * @param learning_rate Gradient Descent step size.
 */
void train_network(Network& net, const MatrixXd& inputs_data, const MatrixXd& targets_data, 
                   int epochs, double learning_rate) 
{
    int num_samples = inputs_data.cols();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
        
        // Accumulators for gradients (for eventual batch/mini-batch support)
        // We update the net after each sample (Stochastic Gradient Descent)
        // For simplicity, we just use a temporary delta for W and B
        
        for (int k = 0; k < num_samples; ++k) {
            VectorXd input_data = inputs_data.col(k);
            VectorXd target_data = targets_data.col(k);

            // --- Forward Pass (Store intermediate results for Backpropagation) ---
            VectorXd Z1, A1, Z2, A2, Z3;
            VectorXd prediction = forward_pass(net, input_data, Z1, A1, Z2, A2, Z3);

            // Cross-Entropy Loss (approximated for monitoring)
            total_loss -= (target_data.array() * (prediction.array() + 1e-9).log()).sum();

            // --- Backpropagation ---

            // 1. Output Layer Error (dLoss/dZ3)
            // Error = Prediction - Target (Softmax + Cross-Entropy derivative)
            VectorXd delta3 = prediction - target_data; 
            
            // 2. Hidden Layer 2 Error (dLoss/dZ2)
            // delta2 = (W3^T * delta3) .* ReLU'(Z2)
            VectorXd relu_prime_Z2 = Z2.unaryExpr([](double z){ return relu_derivative(z); });
            VectorXd delta2 = (net.W3.transpose() * delta3).cwiseProduct(relu_prime_Z2);

            // 3. Hidden Layer 1 Error (dLoss/dZ1)
            // delta1 = (W2^T * delta2) .* ReLU'(Z1)
            VectorXd relu_prime_Z1 = Z1.unaryExpr([](double z){ return relu_derivative(z); });
            VectorXd delta1 = (net.W2.transpose() * delta2).cwiseProduct(relu_prime_Z1);

            // --- Weight and Bias Update (Gradient Descent) ---

            // W3 update: dW3 = delta3 * A2^T
            net.W3 -= learning_rate * (delta3 * A2.transpose());
            net.B3 -= learning_rate * delta3; // dB3 = delta3

            // W2 update: dW2 = delta2 * A1^T
            net.W2 -= learning_rate * (delta2 * A1.transpose());
            net.B2 -= learning_rate * delta2; // dB2 = delta2

            // W1 update: dW1 = delta1 * X^T
            net.W1 -= learning_rate * (delta1 * input_data.transpose());
            net.B1 -= learning_rate * delta1; // dB1 = delta1
        }

        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << ", Average Loss: " << total_loss / num_samples << endl;
        }
    }
}

// --- 6. Prediction ---

/**
 * @brief Predicts the output class probabilities for a single input vector.
 * @param net The trained network structure.
 * @param input_data The input vector.
 * @return The final probability vector (Softmax output).
 */
VectorXd predict(const Network& net, const VectorXd& input_data) {
    VectorXd Z1, A1, Z2, A2, Z3;
    // Note: We ignore Z1, A1, Z2, A2, Z3 in prediction, just need the final output
    return forward_pass(net, input_data, Z1, A1, Z2, A2, Z3);
}

// --- 7. Data Preparation and Main Execution ---

/**
 * @brief Prepares the hard-coded training data.
 * @param input_size The expected input dimension.
 * @param output_size The expected output dimension.
 * @param inputs_out Reference to store the input data matrix.
 * @param targets_out Reference to store the one-hot target matrix.
 */
void prepare_data(int input_size, int output_size, MatrixXd& inputs_out, MatrixXd& targets_out) {
    // 3 samples for 3-class classification, 3 features
    vector<vector<double>> input_raw = {
        {1.0/12.0, 2.0/12.0, 3.0/12.0}, // Target 0
        {4.0/12.0, 5.0/12.0, 6.0/12.0}, // Target 1
        {7.0/12.0, 8.0/12.0, 9.0/12.0}  // Target 2
    };

    // The targets are the class indices (0, 1, 2)
    vector<int> target_raw = {0, 1, 2};
    int num_samples = input_raw.size();

    // Resize Eigen Matrices
    inputs_out.resize(input_size, num_samples);
    targets_out.resize(output_size, num_samples);
    targets_out.setZero(); // Initialize targets to zero

    // Populate matrices
    for (int k = 0; k < num_samples; ++k) {
        // Inputs (InputSize x NumSamples)
        for (int i = 0; i < input_size; ++i) {
            inputs_out(i, k) = input_raw[k][i];
        }
        
        // Targets (One-hot encoded, OutputSize x NumSamples)
        int class_index = target_raw[k];
        targets_out(class_index, k) = 1.0;
    }
}

int main() {
    // --- Network Parameters ---
    const int INPUT_SIZE = 3;
    const int HIDDEN_SIZE_1 = 4;
    const int HIDDEN_SIZE_2 = 4;
    const int OUTPUT_SIZE = 3;
    const int EPOCHS = 10000;
    const double LEARNING_RATE = 0.05;

    cout << "--- Three-Layer MLP Training (Eigen Optimized) ---" << endl;
    cout << "Architecture: " << INPUT_SIZE << " -> " << HIDDEN_SIZE_1 << " -> " << HIDDEN_SIZE_2 << " -> " << OUTPUT_SIZE << endl;
    cout << fixed << setprecision(6);
    
    // --- Data Preparation ---
    MatrixXd inputs, targets;
    prepare_data(INPUT_SIZE, OUTPUT_SIZE, inputs, targets);
    
    // --- Initialization ---
    Network net = initialize_network(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE);
    
    // --- Training ---
    cout << "Starting Training for " << EPOCHS << " epochs..." << endl;
    train_network(net, inputs, targets, EPOCHS, LEARNING_RATE);
    cout << "Training Complete." << endl;
    
    // --- Testing ---
    cout << "\n--- Testing Predictions ---" << endl;
    
    // Test the trained network with the first input
    VectorXd test_input = inputs.col(0); // Input for Class 0
    VectorXd prediction_0 = predict(net, test_input);
    
    cout << "Input: " << test_input.transpose() << endl;
    cout << "Target: [1, 0, 0]" << endl;
    cout << "Prediction: " << prediction_0.transpose() << endl;
    long predicted_class_0 = 0;
    prediction_0.maxCoeff(&predicted_class_0);
    cout << "Predicted Class: " << predicted_class_0 << endl;

    // Test the trained network with the third input
    test_input = inputs.col(2); // Input for Class 2
    VectorXd prediction_2 = predict(net, test_input);

    cout << "\nInput: " << test_input.transpose() << endl;
    cout << "Target: [0, 0, 1]" << endl;
    cout << "Prediction: " << prediction_2.transpose() << endl;
    long predicted_class_2 = 0;
    prediction_2.maxCoeff(&predicted_class_2);
    cout << "Predicted Class: " << predicted_class_2 << endl;
    
    return 0;
}
