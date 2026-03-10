import numpy as np
import math 
class Neuron: 
    def __init__(self, inputs):
        self.weights = np.random.rand(inputs)
        self.bias = np.random.rand() * 0.1

    def computeZ (self, input_arr): #Weighted sum 
            z = np.sum(np.multiply(input_arr, self.weights)) + self.bias
            return z

    def sigmoid (self, z): # Squash the weighted sum and trasnform it into a non linear 
        return 1 / (1 + np.exp(-z))







if __name__ == "__main__":
    neuron1 = Neuron(3) 
    n1_weigths = neuron1.weights
    print(neuron1.weights)
    n1_bias = neuron1.bias
    print(neuron1.bias) 
    n1_z = neuron1.computeZ([0.5, 0.8, 0.2])
    print(neuron1.sigmoid(n1_z))