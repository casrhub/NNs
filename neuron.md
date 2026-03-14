# Neurons

## Biological Neurons

### What is a neuron

![neurondiagram](images/neuron/neuron-diagram.png)

A neuron is a biological structure that receives electrochemical inputs from other neurons through dendrites and produces outputs through axons. These outputs are also electrochemical signals that are passed as inputs to other neurons. Signals are integrated in the cell body (soma), similar to the weighted sum + bias in an artificial neuron.

Electrochemical signals are the primary way neurons communicate, but they are not the only dimension; we also have neuromodulators, hormones, and glial cells.

[The biological inspiration](https://apxml.com/courses/introduction-to-neural-networks/chapter-1-neural-network-foundations/artificial-neurons)


## Artificial Neurons

### What is an artificial neuron

An artificial neuron is a mathematical function: a simplified version of what a biological neuron does. It takes input signals, processes them, and produces a **single** output signal.

If interested in the differences between ANNs and BNNs, read: [Difference between ANN and BNN](https://www.geeksforgeeks.org/machine-learning/difference-between-ann-and-bnn/)

Maybe we can get some ideas for translating/implementing elements of a BNN into an ANN. 


### Components

#### Inputs

Inputs are the raw data we feed into the neuron. They are represented as an array of numbers:

- **Image**: pixel values like [0.2, 0.5, 0.9, ...] where each number is a pixel's brightness (0 = black, 1 = white)
- **Weather prediction**: sensor readings like [temperature, humidity, pressure] = [22.5, 0.65, 1013]
- **Logic gates**: binary inputs like [0, 1] or [1, 1]

Each input gets its own weight, so the array sizes must match: 3 inputs : 3 weights.

#### Weights

Weights determine how much importance each input has. These values are crucial for learning, we modify them based on how much we are "losing" when we calculate the outputs.

#### Bias

A bias helps the activation function decide whether or not to fire the neuron. It shifts the z value before it goes into the activation function. 

You might wonder why do we need a bias? what does it mean?
Imagine this example: 

Inputs = [0,0]

Weights = [0.5, 0.5]

Bias = - 0.7

**Without bias:**

We compute z as $z = W_1 * X_1 + W_2 * X_2$ 

$z = (0.5 * 0) + (0.5 * 0) = 0$

If we pass 0 into our sigmoid function: $sigmoid(0) = 1 / (1 + e^0) = 1 / (1 + 1) = 0.5$

The output is exactly in the middle — the neuron is undecided. For an AND gate, we need [0,0] to output 0 (don't fire), but we're getting 0.5.

**With bias = -0.7:**

Now z includes the bias: $z = W_1 * X_1 + W_2 * X_2 + b$

$z = (0.5 * 0) + (0.5 * 0) + (-0.7) = -0.7$

$sigmoid(-0.7) \approx 0.33$

Now the output leans toward 0 (don't fire). The bias shifted the decision — it raised the bar for what counts as "enough evidence to fire."

[Read more about weights and biases](https://apxml.com/courses/introduction-to-neural-networks/chapter-1-neural-network-foundations/weights-and-biases)



### Forward Pass

The forward pass computes the neuron's output. After defining our weights and biases:

1. **Compute the weighted sum:**
![summation-eq](/images/neuron/summation.png)
$z = w \cdot x + b$

2. **Apply the activation function** to make the output non-linear:
![activation](/images/neuron/activation-funciton.png)

The weighted sum z is a linear operation. Even a complex neural network using only linear operations would behave like a simple linear regression model. By applying an activation function, we introduce non-linearity, enabling the neuron to learn complex patterns. 

### Why Activation Matters (Non-linearity)

Try drawing a straight line that separates an XOR x,y plot — you can't! This is why non-linear activation functions are essential.

[Learn more about activation functions](https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/)


### Artificial Neuron Flow Overview
![an-overview](/images/neuron/artificial-neuron-overview.png)

With that, we have a basic neuron. To summarize:

1. Take inputs $X_i$, multiply them by their corresponding weights $W_i$, and add a bias (which sets the threshold for how much evidence the neuron needs before firing)
2. Transform the output z into a non-linear version through an activation function


## Learning

Learning is the process of modifying our learnable parameters repeatedly until we reach a point where we are satisfied with the output of our loss function. The loss function measures how close our predictions are to the actual expected output.

We have two learnable parameters:
- **Weights**: how much importance each input has on the output
- **Bias**: how easily or reluctantly the neuron fires