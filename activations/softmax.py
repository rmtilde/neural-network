import numpy as np

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for i, (output, dvalue) in enumerate(zip(self.output, dvalues)):
            output = output.reshape(-1, 1)
            jacobian = np.diagflat(output) - np.dot(output, output.T)
            self.dinputs[i] = np.dot(jacobian, dvalue)