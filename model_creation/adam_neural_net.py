"""
AdamNeuralNet.py — Neural networks with Adam optimizer.

Contains:
  - Adam: Optimizer class  
  - Layer: Single dense layer
  - AdamNeuralNetwork: 2-hidden-layer NN with Adam
  - FinalNetwork: 2-hidden-layer NN with Adam + early stopping + confusion matrix
  - NeuralNetwork: Full-featured NN with dropout, L2, multiple optimizers (M1 + O12)
"""

import numpy as np
from ml_framework.layers.activations import relu, relu_derivative, softmax
from ml_framework.metrics import accuracy, f1_score
from ml_framework.visualization.training_plots import plot_metrics, plot_confusion_matrix

from helper_functions import cross_entropy_loss


# ════════════════════════════════════════════════════════════════════
#  ADAM OPTIMIZER
# ════════════════════════════════════════════════════════════════════

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, layer):
        for param, grad_attr in [('weights', 'grad_weights'), ('bias', 'grad_bias')]:
            if hasattr(layer, param) and hasattr(layer, grad_attr):
                param_value = getattr(layer, param)
                grad = getattr(layer, grad_attr)
                key = (id(layer), param)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param_value)
                    self.v[key] = np.zeros_like(param_value)
                    self.t[key] = 0
                self.t[key] += 1
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
                m_hat = self.m[key] / (1 - self.beta1 ** self.t[key])
                v_hat = self.v[key] / (1 - self.beta2 ** self.t[key])
                setattr(layer, param,
                        param_value - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))


# ════════════════════════════════════════════════════════════════════
#  LAYER
# ════════════════════════════════════════════════════════════════════

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        self.grad_weights = None
        self.grad_bias = None


# ════════════════════════════════════════════════════════════════════
#  ADAM NEURAL NETWORK (basic — 2 hidden layers)
# ════════════════════════════════════════════════════════════════════

class AdamNeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size,
                 learning_rate=0.001):
        self.layer1 = Layer(input_size, hidden1_size)
        self.layer2 = Layer(hidden1_size, hidden2_size)
        self.layer3 = Layer(hidden2_size, output_size)
        self.optimizer = Adam(learning_rate=learning_rate)

    def forward(self, X):
        Z1 = X @ self.layer1.weights + self.layer1.bias
        A1 = relu(Z1)
        Z2 = A1 @ self.layer2.weights + self.layer2.bias
        A2 = relu(Z2)
        Z3 = A2 @ self.layer3.weights + self.layer3.bias
        A3 = softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    def backward(self, X, y, Z1, A1, Z2, A2, Z3, A3):
        m = X.shape[0]
        dZ3 = A3 - y
        self.layer3.grad_weights = (A2.T @ dZ3) / m
        self.layer3.grad_bias = np.sum(dZ3, axis=0, keepdims=True) / m
        dZ2 = (dZ3 @ self.layer3.weights.T) * relu_derivative(A2)
        self.layer2.grad_weights = (A1.T @ dZ2) / m
        self.layer2.grad_bias = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = (dZ2 @ self.layer2.weights.T) * relu_derivative(A1)
        self.layer1.grad_weights = (X.T @ dZ1) / m
        self.layer1.grad_bias = np.sum(dZ1, axis=0, keepdims=True) / m

    def step(self):
        self.optimizer.update(self.layer1)
        self.optimizer.update(self.layer2)
        self.optimizer.update(self.layer3)

    def predict(self, X):
        _, _, _, _, _, A3 = self.forward(X)
        return np.argmax(A3, axis=1)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=200,
              class_weights=None, class_names=None, plot_fn=None):

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []

        for epoch in range(epochs):
            Z1, A1, Z2, A2, Z3, A3 = self.forward(X_train)
            self.backward(X_train, y_train, Z1, A1, Z2, A2, Z3, A3)
            self.step()

            train_loss = cross_entropy_loss(y_train, A3, class_weights=class_weights)
            train_acc = accuracy(y_train, A3)
            train_f1 = f1_score(y_train, A3)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            train_f1s.append(train_f1)

            if X_val is not None and y_val is not None:
                _, _, _, _, _, A3_val = self.forward(X_val)
                val_loss = cross_entropy_loss(y_val, A3_val, class_weights=class_weights)
                val_acc = accuracy(y_val, A3_val)
                val_f1 = f1_score(y_val, A3_val)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                val_f1s.append(val_f1)

            if epoch % 10 == 0 or epoch == epochs - 1:
                msg = (f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
                if X_val is not None:
                    msg += (f" | Val Loss: {val_loss:.4f}, "
                            f"Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
                print(msg)

        if plot_fn is not None:
            plot_fn(train_losses, val_losses if X_val is not None else None,
                    train_accs, val_accs if X_val is not None else None,
                    train_f1s, val_f1s if X_val is not None else None)

        # Confusion Matrix
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            y_true = np.argmax(y_val, axis=1)
            n_classes = y_val.shape[1]
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            print("Confusion Matrix:\n", cm)
            plot_confusion_matrix(cm, class_names=class_names, normalize=False,
                                  title="Validation Confusion Matrix (Counts)")
            plot_confusion_matrix(cm, class_names=class_names, normalize=True,
                                  title="Validation Confusion Matrix (Normalized)")


# ════════════════════════════════════════════════════════════════════
#  FINAL NETWORK (Adam + Early Stopping + Confusion Matrix)
# ════════════════════════════════════════════════════════════════════

class FinalNetwork:
    """
    Production-ready 2-hidden-layer NN with Adam optimizer,
    early stopping, and automatic confusion matrix plotting.
    """

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size,
                 learning_rate=0.001):
        self.layer1 = Layer(input_size, hidden1_size)
        self.layer2 = Layer(hidden1_size, hidden2_size)
        self.layer3 = Layer(hidden2_size, output_size)
        self.optimizer = Adam(learning_rate=learning_rate)

    def forward(self, X):
        Z1 = X @ self.layer1.weights + self.layer1.bias
        A1 = relu(Z1)
        Z2 = A1 @ self.layer2.weights + self.layer2.bias
        A2 = relu(Z2)
        Z3 = A2 @ self.layer3.weights + self.layer3.bias
        A3 = softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    def backward(self, X, y, Z1, A1, Z2, A2, Z3, A3):
        m = X.shape[0]
        dZ3 = A3 - y
        self.layer3.grad_weights = (A2.T @ dZ3) / m
        self.layer3.grad_bias = np.sum(dZ3, axis=0, keepdims=True) / m
        dZ2 = (dZ3 @ self.layer3.weights.T) * relu_derivative(A2)
        self.layer2.grad_weights = (A1.T @ dZ2) / m
        self.layer2.grad_bias = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = (dZ2 @ self.layer2.weights.T) * relu_derivative(A1)
        self.layer1.grad_weights = (X.T @ dZ1) / m
        self.layer1.grad_bias = np.sum(dZ1, axis=0, keepdims=True) / m

    def step(self):
        self.optimizer.update(self.layer1)
        self.optimizer.update(self.layer2)
        self.optimizer.update(self.layer3)

    def predict(self, X):
        _, _, _, _, _, A3 = self.forward(X)
        return np.argmax(A3, axis=1)

    def train(self, X_train, y_train, X_val, y_val,
              epochs=300, patience=20, min_delta=1e-4,
              class_weights=None, class_names=None):

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []

        best_val_loss = float("inf")
        best_epoch = 0
        best_weights = None

        for epoch in range(epochs):
            Z1, A1, Z2, A2, Z3, A3 = self.forward(X_train)
            self.backward(X_train, y_train, Z1, A1, Z2, A2, Z3, A3)
            self.step()

            # Training metrics
            train_loss = cross_entropy_loss(y_train, A3, class_weights=class_weights)
            train_acc = accuracy(y_train, A3)
            train_f1 = f1_score(y_train, A3)

            # Validation
            _, _, _, _, _, A3_val = self.forward(X_val)
            val_loss = cross_entropy_loss(y_val, A3_val, class_weights=class_weights)
            val_acc = accuracy(y_val, A3_val)
            val_f1 = f1_score(y_val, A3_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
                      f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                best_weights = (
                    self.layer1.weights.copy(), self.layer1.bias.copy(),
                    self.layer2.weights.copy(), self.layer2.bias.copy(),
                    self.layer3.weights.copy(), self.layer3.bias.copy(),
                )
            elif epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best epoch was {best_epoch} with Val Loss {best_val_loss:.4f}.")
                break

        # Restore best weights
        if best_weights is not None:
            (self.layer1.weights, self.layer1.bias,
             self.layer2.weights, self.layer2.bias,
             self.layer3.weights, self.layer3.bias) = best_weights

        # Plots
        plot_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s)

        # Confusion matrix
        y_val_true = np.argmax(y_val, axis=1)
        y_val_pred = self.predict(X_val)
        cm_val = self._confusion_matrix(y_val_true, y_val_pred, num_classes=y_val.shape[1])

        print("Confusion Matrix (Validation):\n", cm_val)
        plot_confusion_matrix(cm_val, class_names=class_names, normalize=False,
                              title="Validation Confusion Matrix (Counts)")
        plot_confusion_matrix(cm_val, class_names=class_names, normalize=True,
                              title="Validation Confusion Matrix (Normalized)")

    @staticmethod
    def _confusion_matrix(y_true, y_pred, num_classes):
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm


# ════════════════════════════════════════════════════════════════════
#  FULL-FEATURED NEURAL NETWORK (M1 + O12)
#  Supports: dropout, L2, SGD / Momentum / Adam, gradient clipping,
#            mini-batch training, LR scheduling, early stopping
# ════════════════════════════════════════════════════════════════════

class NeuralNetwork:
    """
    Fully-connected neural network — NumPy only.
    Supports arbitrary hidden layers, ReLU activations, Softmax output.
    Regularization: L2 (weight decay) + Dropout (inverted).
    Optimizers (O12): 'sgd', 'momentum', 'adam'.
    """

    def __init__(self, layer_sizes, dropout_rate=0.0, l2_lambda=0.0, seed=42,
                 init_mode='small'):
        self.rng = np.random.default_rng(seed)
        self.n_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.training = True

        # Weight init
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            if init_mode == 'he':
                W = self.rng.standard_normal(size=(fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            else:
                W = self.rng.standard_normal(size=(fan_in, fan_out)) * 0.1
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

        self.z_cache = []
        self.a_cache = []
        self.dropout_masks = []

        self.optimizer = None
        self.opt_state = {}

    def init_optimizer(self, optimizer='adam', lr=0.001, momentum=0.9,
                       beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.optimizer = optimizer
        self.lr = lr
        self.opt_state = {'t': 0}

        if optimizer == 'momentum':
            self.opt_state['momentum'] = momentum
            self.opt_state['v_w'] = [np.zeros_like(w) for w in self.weights]
            self.opt_state['v_b'] = [np.zeros_like(b) for b in self.biases]

        elif optimizer == 'adam':
            self.opt_state['beta1'] = beta1
            self.opt_state['beta2'] = beta2
            self.opt_state['epsilon'] = epsilon
            self.opt_state['m_w'] = [np.zeros_like(w) for w in self.weights]
            self.opt_state['m_b'] = [np.zeros_like(b) for b in self.biases]
            self.opt_state['v_w'] = [np.zeros_like(w) for w in self.weights]
            self.opt_state['v_b'] = [np.zeros_like(b) for b in self.biases]

    def forward(self, X):
        self.z_cache = []
        self.a_cache = [X]
        self.dropout_masks = []

        a = X
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.z_cache.append(z)

            if i < self.n_layers - 1:
                a = relu(z)
                if self.training and self.dropout_rate > 0:
                    mask = (self.rng.random(a.shape) > self.dropout_rate).astype(a.dtype)
                    a = a * mask / (1.0 - self.dropout_rate)
                    self.dropout_masks.append(mask)
                else:
                    self.dropout_masks.append(None)
            else:
                a = softmax(z)

            self.a_cache.append(a)

        return a

    def backward(self, y_true):
        n = y_true.shape[0]
        grads_w = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        dz = self.a_cache[-1] - y_true

        for i in reversed(range(self.n_layers)):
            a_prev = self.a_cache[i]

            grads_w[i] = (a_prev.T @ dz) / n
            if self.l2_lambda > 0:
                grads_w[i] += 2 * self.l2_lambda * self.weights[i]

            grads_b[i] = np.sum(dz, axis=0, keepdims=True) / n

            if i > 0:
                dz = (dz @ self.weights[i].T) * relu_derivative(self.z_cache[i - 1])
                if self.dropout_masks[i - 1] is not None:
                    dz = dz * self.dropout_masks[i - 1] / (1.0 - self.dropout_rate)

        return grads_w, grads_b

    def update(self, grads_w, grads_b, max_grad_norm=5.0):
        """Apply optimizer step with gradient clipping."""
        all_grads = list(grads_w) + list(grads_b)
        global_norm = np.sqrt(sum(np.sum(g ** 2) for g in all_grads))
        if global_norm > max_grad_norm:
            scale = max_grad_norm / (global_norm + 1e-8)
            grads_w = [g * scale for g in grads_w]
            grads_b = [g * scale for g in grads_b]

        self.opt_state['t'] += 1

        if self.optimizer == 'sgd':
            for i in range(self.n_layers):
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]

        elif self.optimizer == 'momentum':
            mu = self.opt_state['momentum']
            for i in range(self.n_layers):
                self.opt_state['v_w'][i] = mu * self.opt_state['v_w'][i] - self.lr * grads_w[i]
                self.opt_state['v_b'][i] = mu * self.opt_state['v_b'][i] - self.lr * grads_b[i]
                self.weights[i] += self.opt_state['v_w'][i]
                self.biases[i] += self.opt_state['v_b'][i]

        elif self.optimizer == 'adam':
            t = self.opt_state['t']
            b1 = self.opt_state['beta1']
            b2 = self.opt_state['beta2']
            eps = self.opt_state['epsilon']
            for i in range(self.n_layers):
                self.opt_state['m_w'][i] = b1 * self.opt_state['m_w'][i] + (1 - b1) * grads_w[i]
                self.opt_state['v_w'][i] = b2 * self.opt_state['v_w'][i] + (1 - b2) * grads_w[i]**2
                m_hat = self.opt_state['m_w'][i] / (1 - b1**t)
                v_hat = self.opt_state['v_w'][i] / (1 - b2**t)
                self.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

                self.opt_state['m_b'][i] = b1 * self.opt_state['m_b'][i] + (1 - b1) * grads_b[i]
                self.opt_state['v_b'][i] = b2 * self.opt_state['v_b'][i] + (1 - b2) * grads_b[i]**2
                m_hat = self.opt_state['m_b'][i] / (1 - b1**t)
                v_hat = self.opt_state['v_b'][i] / (1 - b2**t)
                self.biases[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, X):
        self.training = False
        probs = self.forward(X)
        self.training = True
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        self.training = False
        probs = self.forward(X)
        self.training = True
        return probs
