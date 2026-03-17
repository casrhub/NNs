from neuron import Neuron

# Create neuron with 2 inputs (for AND gate)
neuron = Neuron(2)

# AND gate training data
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [0, 0, 0, 1]

# Training parameters
learning_rate = 0.5
epochs = 40000


for epoch in range(epochs):
    for i in range(len(inputs)):

        # forward pass
        z,y = neuron.forwardPass(inputs[i])

        #compute gradients 
        grad_w, grad_b = neuron.computeGradients(inputs[i], y, expected[i])
        

        # Update weights
        neuron.updateWeights(grad_w, grad_b, learning_rate)


correct = 0
for i in range(len(inputs)):
    z, y = neuron.forwardPass(inputs[i])
    rounded = 1 if y >= 0.5 else 0
    
    if rounded == expected[i]:
        correct += 1
    
    print(f"{inputs[i]} → {y:.4f} → {rounded} (expected: {expected[i]})")
accuracy = correct / len(inputs) * 100
print(f"\nAccuracy: {accuracy}% ({correct}/{len(inputs)} correct)")








