from neuron import Neuron
import matplotlib.pyplot as plt

# Create neuron with 2 inputs (for AND gate)
neuron = Neuron(2)

# AND gate training data
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected = [0, 0, 0, 1]

# Training parameters
learning_rate = 0.5
epochs = 40000

# Track loss over time
loss_history = []

for epoch in range(epochs):
    epoch_loss = 0  # Track loss for this epoch
    
    for i in range(len(inputs)):
        # forward pass
        z, y = neuron.forwardPass(inputs[i])
        
        # Compute loss for tracking
        epoch_loss += neuron.loss(y, expected[i])

        # compute gradients 
        grad_w, grad_b = neuron.computeGradients(inputs[i], y, expected[i])

        # Update weights
        neuron.updateWeights(grad_w, grad_b, learning_rate)
    
    # Store average loss for this epoch
    loss_history.append(epoch_loss / len(inputs))


correct = 0
for i in range(len(inputs)):
    z, y = neuron.forwardPass(inputs[i])
    rounded = 1 if y >= 0.5 else 0
    
    if rounded == expected[i]:
        correct += 1
    
    print(f"Input: {inputs[i]} Output: {y:.4f} Rounded: {rounded} (expected: {expected[i]})")
accuracy = correct / len(inputs) * 100
print(f"\nAccuracy: {accuracy}% ({correct}/{len(inputs)} correct)")

print(f"Final weights: {neuron.weights}")
print(f"Final bias: {neuron.bias}")

print(f"Loss at epoch 10000: {loss_history[10000]}")


# Plot loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Loss Over Training')
plt.grid(True, alpha=0.3)
plt.savefig('training_loss.png', dpi=150)
plt.show()








