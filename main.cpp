#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <algorithm> // Add for std::shuffle

struct Network {
    // Layer 1: Input -> Hidden1
    std::vector<double> input_weights;   // size = input_size * hidden1_size
    std::vector<double> hidden1_biases;  // size = hidden1_size
    
    // Layer 2: Hidden1 -> Hidden2
    std::vector<double> hidden1_weights; // size = hidden1_size * hidden2_size
    std::vector<double> hidden2_biases;  // size = hidden2_size
    
    // Layer 3: Hidden2 -> Output
    std::vector<double> hidden2_weights; // size = hidden2_size * output_size
    std::vector<double> output_biases;   // size = output_size
};

template <typename T>
std::vector<T> safe_vector(size_t size) {
    if (size > std::numeric_limits<typename std::vector<T>::size_type>::max() / sizeof(T)) {
        throw std::length_error("Vector size exceeds safe limits");
    }
    return std::vector<T>(size);
}

// Activation functions remain the same
double relu(double x) { return std::max(0.0, x); }
double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

std::vector<double> softmax(const std::vector<double>& z) {
    std::vector<double> sm(z.size());
    double max_z = *std::max_element(z.begin(), z.end());
    double sum = 0.0;
    for (double val : z) sum += std::exp(val - max_z);
    for (size_t i = 0; i < z.size(); ++i) sm[i] = std::exp(z[i] - max_z) / sum;
    return sm;
}

void initialize_weights(std::vector<double>& weights, size_t fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, std::sqrt(2.0 / fan_in));
    std::generate(weights.begin(), weights.end(), [&]() { return dis(gen); });
}

void initialize_network(size_t input_size, size_t hidden1_size, 
                       size_t hidden2_size, size_t output_size, Network& network) {
    // Initialize input -> hidden1
    network.input_weights = safe_vector<double>(input_size * hidden1_size);
    initialize_weights(network.input_weights, input_size);
    network.hidden1_biases = safe_vector<double>(hidden1_size);
    std::fill(network.hidden1_biases.begin(), network.hidden1_biases.end(), 0.1);

    // Initialize hidden1 -> hidden2
    network.hidden1_weights = safe_vector<double>(hidden1_size * hidden2_size);
    initialize_weights(network.hidden1_weights, hidden1_size);
    network.hidden2_biases = safe_vector<double>(hidden2_size);
    std::fill(network.hidden2_biases.begin(), network.hidden2_biases.end(), 0.1);

    // Initialize hidden2 -> output
    network.hidden2_weights = safe_vector<double>(hidden2_size * output_size);
    initialize_weights(network.hidden2_weights, hidden2_size);
    network.output_biases = safe_vector<double>(output_size);
    std::fill(network.output_biases.begin(), network.output_biases.end(), 0.1);
}

void train_network(const std::vector<std::vector<double>>& inputs,
                   const std::vector<int>& targets,
                   size_t hidden1_size,
                   size_t hidden2_size,
                   size_t output_size,
                   Network& network,
                   size_t epochs,
                   double learning_rate) {
    const size_t input_size = inputs[0].size();
    
    // Pre-allocate buffers
    std::vector<double> hidden1_outputs(hidden1_size);
    std::vector<double> hidden2_outputs(hidden2_size);
    std::vector<double> output_values(output_size);
    
    std::vector<double> output_deltas(output_size);
    std::vector<double> hidden2_deltas(hidden2_size);
    std::vector<double> hidden1_deltas(hidden1_size);

    for (size_t epoch = 0; epoch <= epochs; ++epoch) {
        double total_error = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            
            // Forward pass ---------------------------------------------------
            // Input -> Hidden1
            for (size_t h1 = 0; h1 < hidden1_size; ++h1) {
                double sum = 0.0;
                for (size_t j = 0; j < input_size; ++j) {
                    sum += input[j] * network.input_weights[h1 * input_size + j];
                }
                hidden1_outputs[h1] = relu(sum + network.hidden1_biases[h1]);
            }
            
            // Hidden1 -> Hidden2
            for (size_t h2 = 0; h2 < hidden2_size; ++h2) {
                double sum = 0.0;
                for (size_t h1 = 0; h1 < hidden1_size; ++h1) {
                    sum += hidden1_outputs[h1] * network.hidden1_weights[h2 * hidden1_size + h1];
                }
                hidden2_outputs[h2] = relu(sum + network.hidden2_biases[h2]);
            }
            
            // Hidden2 -> Output
            for (size_t o = 0; o < output_size; ++o) {
                double sum = 0.0;
                for (size_t h2 = 0; h2 < hidden2_size; ++h2) {
                    sum += hidden2_outputs[h2] * network.hidden2_weights[o * hidden2_size + h2];
                }
                output_values[o] = sum + network.output_biases[o];
            }
            
            auto predictions = softmax(output_values);
            
            // Error calculation ----------------------------------------------
            std::vector<double> target_vec(output_size, 0.0);
            target_vec[targets[i]] = 1.0;
            for (size_t o = 0; o < output_size; ++o) {
                double error = target_vec[o] - predictions[o];
                output_deltas[o] = error;
                total_error += error * error;
            }
            
            // Backpropagation ------------------------------------------------
            // Calculate hidden2 deltas
            for (size_t h2 = 0; h2 < hidden2_size; ++h2) {
                double error = 0.0;
                for (size_t o = 0; o < output_size; ++o) {
                    error += output_deltas[o] * network.hidden2_weights[o * hidden2_size + h2];
                }
                hidden2_deltas[h2] = error * relu_derivative(hidden2_outputs[h2]);
            }
            
            // Calculate hidden1 deltas
            for (size_t h1 = 0; h1 < hidden1_size; ++h1) {
                double error = 0.0;
                for (size_t h2 = 0; h2 < hidden2_size; ++h2) {
                    error += hidden2_deltas[h2] * network.hidden1_weights[h2 * hidden1_size + h1];
                }
                hidden1_deltas[h1] = error * relu_derivative(hidden1_outputs[h1]);
            }
            
            // Update weights and biases ---------------------------------------
            // Update hidden2->output weights
            for (size_t o = 0; o < output_size; ++o) {
                for (size_t h2 = 0; h2 < hidden2_size; ++h2) {
                    network.hidden2_weights[o * hidden2_size + h2] += 
                        learning_rate * output_deltas[o] * hidden2_outputs[h2];
                }
                network.output_biases[o] += learning_rate * output_deltas[o];
            }
            
            // Update hidden1->hidden2 weights
            for (size_t h2 = 0; h2 < hidden2_size; ++h2) {
                for (size_t h1 = 0; h1 < hidden1_size; ++h1) {
                    network.hidden1_weights[h2 * hidden1_size + h1] += 
                        learning_rate * hidden2_deltas[h2] * hidden1_outputs[h1];
                }
                network.hidden2_biases[h2] += learning_rate * hidden2_deltas[h2];
            }
            
            // Update input->hidden1 weights
            for (size_t h1 = 0; h1 < hidden1_size; ++h1) {
                for (size_t j = 0; j < input_size; ++j) {
                    network.input_weights[h1 * input_size + j] += 
                        learning_rate * hidden1_deltas[h1] * input[j];
                }
                network.hidden1_biases[h1] += learning_rate * hidden1_deltas[h1];
            }
        }
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Avg Error: " 
                      << total_error / inputs.size() << std::endl;
        }
    }
}

std::vector<double> predict(const Network& network, 
                            const std::vector<double>& input,
                            size_t hidden1_size,
                            size_t hidden2_size,
                            size_t output_size) {
    std::vector<double> hidden1_outputs(hidden1_size);
    std::vector<double> hidden2_outputs(hidden2_size);
    std::vector<double> output_values(output_size);
    
    // Input -> Hidden1
    for (size_t h1 = 0; h1 < hidden1_size; ++h1) {
        double sum = 0.0;
        for (size_t j = 0; j < input.size(); ++j) {
            sum += input[j] * network.input_weights[h1 * input.size() + j];
        }
        hidden1_outputs[h1] = relu(sum + network.hidden1_biases[h1]);
    }
    
    // Hidden1 -> Hidden2
    for (size_t h2 = 0; h2 < hidden2_size; ++h2) {
        double sum = 0.0;
        for (size_t h1 = 0; h1 < hidden1_size; ++h1) {
            sum += hidden1_outputs[h1] * network.hidden1_weights[h2 * hidden1_size + h1];
        }
        hidden2_outputs[h2] = relu(sum + network.hidden2_biases[h2]);
    }
    
    // Hidden2 -> Output
    for (size_t o = 0; o < output_size; ++o) {
        double sum = 0.0;
        for (size_t h2 = 0; h2 < hidden2_size; ++h2) {
            sum += hidden2_outputs[h2] * network.hidden2_weights[o * hidden2_size + h2];
        }
        output_values[o] = sum + network.output_biases[o];
    }
    
    return softmax(output_values);
}

// prepare_data remains the same
void prepare_data(std::vector<std::vector<double>>& inputs, std::vector<int>& outputs) {
    inputs = {
        {1.0/12, 2.0/12, 3.0/12}, {1.0/12, 2.0/12, 4.0/12},
        {1.0/12, 2.0/12, 5.0/12}, {1.0/12, 2.0/12, 6.0/12},
        {1.0/12, 2.0/12, 7.0/12}, {1.0/12, 2.0/12, 8.0/12},
        {1.0/12, 2.0/12, 9.0/12}, {1.0/12, 2.0/12, 10.0/12},
        {1.0/12, 2.0/12, 11.0/12}, {1.0/12, 2.0/12, 1.0}
    };
    outputs = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
}

int main() {
    const size_t input_size = 3;
    const size_t hidden1_size = 20;
    const size_t hidden2_size = 15;  // Second hidden layer size
    const size_t output_size = 3;
    const size_t epochs = 300;
    const double learning_rate = 0.02;

    std::vector<std::vector<double>> inputs;
    std::vector<int> targets;
    prepare_data(inputs, targets);

    Network network;
    initialize_network(input_size, hidden1_size, hidden2_size, output_size, network);
    train_network(inputs, targets, hidden1_size, hidden2_size, output_size, 
                 network, epochs, learning_rate);

    // Test the network
    std::vector<double> test_input = {1.0/12, 2.0/12, 3.0/12};
    auto probs = predict(network, test_input, hidden1_size, hidden2_size, output_size);
    std::cout << "\nTest input predictions:\n";
    for (double p : probs) std::cout << p << " ";
    std::cout << std::endl;

    return 0;
}
