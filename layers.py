import numpy as np

class Layer_Dense:
    def __init__(self, num_inputs, num_neurons):
        #self.weights = 0.1*np.random.randn(num_inputs, num_neurons)
        self.weights = np.random.randn(num_inputs, num_neurons) * np.sqrt(2. / num_inputs)

        self.biases = np.zeros((1,num_neurons))

        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)