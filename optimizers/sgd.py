class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum

    def update_params(self, layer):

        layer.weight_momentums = self.momentum * layer.weight_momentums - self.lr * layer.dweights
        layer.bias_momentums = self.momentum * layer.bias_momentums - self.lr * layer.dbiases
        layer.weights += layer.weight_momentums
        layer.biases += layer.bias_momentums