"""
BaseNeuralNet.py — Simple feedforward neural network with vanilla SGD.
Two hidden layers with ReLU, softmax output. No advanced optimizers.
"""

import numpy as np
from ml_framework.layers.activations import relu, relu_derivative, softmax
from ml_framework.metrics import accuracy, f1_score
from ml_framework.visualization.training_plots import plot_metrics

from helper_functions import cross_entropy_loss


class BaseNeuralNetwork:
    """
    3-layer neural network (2 hidden + 1 output) with vanilla SGD.
    Activation: ReLU (hidden), Softmax (output).
    """

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size,
                 learning_rate=0.1):
        self.learning_rate = learning_rate
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = \
            self._initialize_weights(input_size, hidden1_size, hidden2_size, output_size)

    def _initialize_weights(self, input_size, h1, h2, output_size):
        W1 = np.random.randn(input_size, h1) * 0.1
        b1 = np.zeros((1, h1))
        W2 = np.random.randn(h1, h2) * 0.1
        b2 = np.zeros((1, h2))
        W3 = np.random.randn(h2, output_size) * 0.1
        b3 = np.zeros((1, output_size))
        return W1, b1, W2, b2, W3, b3

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = relu(Z2)
        Z3 = A2 @ self.W3 + self.b3
        A3 = softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    def backward(self, X, y, Z1, A1, Z2, A2, Z3, A3):
        m = X.shape[0]

        dZ3 = A3 - y
        dW3 = A2.T @ dZ3 / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dZ2 = dZ3 @ self.W3.T * relu_derivative(A2)
        dW2 = A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = dZ2 @ self.W2.T * relu_derivative(A1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Vanilla SGD update
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

    def predict(self, X):
        _, _, _, _, _, A3 = self.forward(X)
        return np.argmax(A3, axis=1)

    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        train_f1s, val_f1s = [], []

        for epoch in range(epochs):
            Z1, A1, Z2, A2, Z3, A3 = self.forward(X_train)
            self.backward(X_train, y_train, Z1, A1, Z2, A2, Z3, A3)

            # Train metrics
            train_loss = cross_entropy_loss(y_train, A3)
            train_acc = accuracy(y_train, A3)
            train_f1 = f1_score(y_train, A3)

            # Validation
            _, _, _, _, _, A3_val = self.forward(X_val)
            val_loss = cross_entropy_loss(y_val, A3_val)
            val_acc = accuracy(y_val, A3_val)
            val_f1 = f1_score(y_val, A3_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

        plot_metrics(train_losses, val_losses,
                     train_accuracies, val_accuracies,
                     train_f1s, val_f1s)
