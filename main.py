
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
from layers import Layer_Dropout

nnfs.init()
np.random.seed(0)

CLASSES = 3
EPOCHS = 4001

X,y = create_data(500, CLASSES)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

split = int(0.8 * len(X))

X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = Model()

model.add(Layer_Dense(2, 64, 1e-4))
model.add(Activation_ReLu())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(64, 64, 1e-4))
model.add(Activation_ReLu())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(64, 3, 1e-4))

model.set(
    loss=Activation_Softmax_Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_SGD(learning_rate=0.05, momentum=0.9)
)

model.train(X_train, y_train, epochs=10001, print_every=1000, validation_data=(X_val, y_val))

val_output = model.forward(X_val, False)
val_predictions = np.argmax(val_output, axis=1)
val_accuracy = np.mean(val_predictions == y_val)

print("Validation accuracy:", val_accuracy)

plot_decision_boundary(X_train,y_train, model, CLASSES)