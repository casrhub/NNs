import numpy as np
import math 
class Neuron: 
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand() * 0.1


    def sigmoid (self, z): # Squash the weighted sum and trasnform it into a non linear 
        return 1 / (1 + np.exp(-z))

    def forwardPass (self, inputs): # 
        # Compute weighted sum: z = w·x + b
        z = np.sum(np.multiply(inputs, self.weights)) + self.bias
        #Activation
        y = self.sigmoid(z)

        return z, y 

    def computeGradients(self, inputs, output, expected): 
        error_signal = (output -expected) * output * (1 - output) # comes from chain rule 
        gradient_weights = error_signal * np.array(inputs) # We use the shared computation * Xi
        gradient_bias = error_signal
        return gradient_weights, gradient_bias

    def updateWeights(self, gradient_weights, gradient_bias, learning_rate):
        self.weights = self.weights - learning_rate * gradient_weights # How big of a step we take into the min
        self.bias = self.bias - learning_rate * gradient_bias
    
    def loss (self, output, expected): # MSE loss function (squared error)
        return (output - expected) ** 2





if __name__ == "__main__":
    neuron1 = Neuron(2) 
    inputs = [[0,0], [0,1], [1,0], [1,1]]
    expected = [0, 0, 0, 1] 
    
    # Test with one example
    test_input = inputs[3]  # [1, 1] - should output 1
    test_expected = expected[3]  # 1
    
    # Step 1: Forward pass
    z, output = neuron1.forwardPass(test_input)
    print("z:", z)
    print("output:", output)
    
    # Step 2: Loss
    error = neuron1.loss(output, test_expected)
    print("loss:", error)
    
    # Step 3: Gradients
    grad_w, grad_b = neuron1.computeGradients(test_input, output, test_expected)
    print("weight gradients:", grad_w)
    print("bias gradient:", grad_b)

    learning_rate = 0.5
    # Before update
    print("weights before:", neuron1.weights)
    print("bias before:", neuron1.bias)
    # Update
    neuron1.updateWeights(grad_w, grad_b, learning_rate)
    # After update
    print("weights after:", neuron1.weights)
    print("bias after:", neuron1.bias)

