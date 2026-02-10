
import numpy as np
import nnfs

from activations import Activation_ReLu
from activations import Activation_Softmax
from optimizers import Optimizer_SGD
from data import create_data
from layers import Layer_Dense
from plot import plot_decision_boundary
from combined import Activation_Softmax_Loss_CategoricalCrossentropy
from model import Model

nnfs.init()
np.random.seed(0)

CLASSES = 3
EPOCHS = 10001
X,y = create_data(100, CLASSES)


model = Model()

model.add(Layer_Dense(2, 64))
model.add(Activation_ReLu())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLu())
model.add(Layer_Dense(64, 3))

model.set(
    loss=Activation_Softmax_Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_SGD(learning_rate=0.05)
)

model.train(X, y, epochs=10001, print_every=1000)

output = model.forward(X)
predictions = np.argmax(output, axis=1)

accuracy = np.mean(predictions == y)

plot_decision_boundary(X,y, model, CLASSES)





