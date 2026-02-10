from combined.softmax_crossentropy import Activation_Softmax_Loss_CategoricalCrossentropy
from layers import Layer_Dense
from losses.categorical_crossentropy import Loss_CategoricalCrossentropy


class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        output = X
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def backward(self, y):
        self.loss.backward(y)
        dvalues = self.loss.dinputs

        for layer in reversed(self.layers):
            layer.backward(dvalues)
            dvalues = layer.dinputs

    def train(self, X, y, epochs=1, print_every=100):
        self._auto_combine_loss_activation()
        
        for epoch in range(epochs):

            output = self.forward(X)
            loss = self.loss.forward(output, y)
            predictions = output.argmax(axis=1)
            accuracy = (predictions == y).mean()

            self.backward(y)

            for layer in self.layers:
                if hasattr(layer, "weights"):
                    self.optimizer.update_params(layer)

            if epoch % print_every == 0:
                print(epoch, loss, accuracy)
    
    def _auto_combine_loss_activation(self):
        last_layer = self.layers[-1]
        if isinstance(last_layer, Layer_Dense) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.loss = Activation_Softmax_Loss_CategoricalCrossentropy()