import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from nnfs.datasets import spiral_data

# Initialize NNFS
import nnfs
nnfs.init()

# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU Activation
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Softmax Activation
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        self.dinputs /= len(dvalues)

# Common Loss Class
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Cross-Entropy Loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs /= samples

# Softmax Classifier - Combined Softmax Activation and Cross-Entropy Loss
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples

class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

def plot_decision_boundaries(X, y, model_states, epoch_list):
    h = .02  # step size in the mesh
    cmap_light = ListedColormap(['red', 'green', 'blue'])
    cmap_bold = ListedColormap(['darkred', 'darkgreen', 'darkblue'])

    # Create directory if it does not exist
    output_dir = 'plot_graphs'
    os.makedirs(output_dir, exist_ok=True)

    for epoch, state in zip(epoch_list, model_states):
        dense1_weights = state['dense1_weights']
        dense1_biases = state['dense1_biases']
        dense2_weights = state['dense2_weights']
        dense2_biases = state['dense2_biases']

        plt.figure(figsize=(10, 6))
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], dense1_weights) + dense1_biases), dense2_weights) + dense2_biases
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap=cmap_bold)
        plt.title(f'Decision Boundary at Epoch {epoch}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Save the plot to the 'plot_graphs' directory
        plt.savefig(f'{output_dir}/decision_boundary_epoch_{epoch}.png')
        plt.close()

if __name__ == '__main__':
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD()

    epochs_to_save = list(range(0, 100001, 1000))
    model_states = []

    for epoch in range(100001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if epoch in epochs_to_save:
            print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}')
            model_states.append({
                'dense1_weights': dense1.weights.copy(),
                'dense1_biases': dense1.biases.copy(),
                'dense2_weights': dense2.weights.copy(),
                'dense2_biases': dense2.biases.copy()
            })

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

    plot_decision_boundaries(X, y, model_states, epochs_to_save)
