# --- Makefile for MLP_3Layer_Eigen_ReLU.cpp ---

# Compiler and Flags
CXX = g++
# C++20 standard, O3 optimization, comprehensive warnings
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra

# Source and Target Files
SOURCE = MLP_3Layer_Eigen_ReLU.cpp
# Executable name: Neural Network ReLU Multi-Layer Perceptron
TARGET = nn_relu_mlp

# Include paths for Eigen and common system libraries
# These match the -I flags in your original command
INCLUDES = -I/usr/include/eigen3 -I/usr/include

# Libraries to Link
# Includes Boost (system, filesystem, math) and cURL
LIBS = -lboost_system -lboost_filesystem -lboost_math_c99 -lcurl

# Default target: builds the executable
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

# Phony targets: tasks that are not files
.PHONY: clean run

# Cleans up the compiled executable and any object files
clean:
	rm -f $(TARGET) *.o

# Runs the compiled program (optional convenience target)
run: $(TARGET)
	./$(TARGET)
