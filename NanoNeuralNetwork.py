import numpy as np, tabulate

class NanoNeuralNetwork:
    def __init__(self, layers, activation_functions=None, hyperparameters=None):
        self.layers = layers
        self.activation_functions = activation_functions if activation_functions is not None else [NanoNeuralNetwork.relu for l in self.layers[1:]]
        self.activation_functions[-1] = NanoNeuralNetwork.sigmoid
        self.hyperparameters = hyperparameters if hyperparameters is not None else {'alpha': 0.1, 'lambda': 0}

        # Create weight+bias matrices per layer initialized to 0
        self.W = [np.zeros((n, m)) for n, m in zip(self.layers[1:], self.layers[:-1])]
        self.B = [np.zeros((1, n)) for n in self.layers[1:]]

    def forward_propagation(self, X):
        # Forward propagation step. Store the dot products, Z, and the activations, A, for each layer
        A = [X]
        Z = [None]

        for w, b, act_fn in zip(self.W, self.B, self.activation_functions):
            Z += [np.dot(A[-1], w.T) + b]
            A += [act_fn(Z[-1])]

        return (A, Z)

    def backward_propagation(self, Y, A, Z):
        # Backward propagation step. Store the weight, delta_W, and bias, delta_B, deltas for each layer
        delta_Z = A[-1] - Y
        delta_W = []
        delta_B = []

        for w, a, z, act_fn  in reversed(list(zip(self.W, A[:-1], Z[:-1], self.activation_functions))):
            delta_W = [np.dot(delta_Z.T, a)] + delta_W
            delta_B = [np.sum(delta_Z, axis=1, keepdims=True)] + delta_B
            delta_Z = np.dot(delta_Z, w) * act_fn(z, derivative=True) if z is not None else None

        return delta_W, delta_B

    def cost(self, Y, Yp):
        m = Y.shape[0]
        return np.squeeze((-1 / m) * (np.dot(Y.T, np.log(Yp)) + np.dot((1 - Y).T, np.log(1 - Yp))))

    def train(self, train_samples, train_labels, iterations=1000):
        m = train_labels.shape[0]
        alpha = self.hyperparameters['alpha']
        lambd = self.hyperparameters['lambda']
        J = []

        # Initialize weights using random values from a standard normal distribution
        self.W = [np.random.standard_normal(w.shape) * np.sqrt(2 / w.shape[1]) for w in self.W]

        for i in range(iterations):
            # Forward propagation to calculate predictions
            A, Z = self.forward_propagation(train_samples)

            # Compute the cost per iteration
            J += [self.cost(train_labels, A[-1])]

            # Backward propagation to calculate gradients
            delta_W, delta_B = self.backward_propagation(train_labels, A, Z)

            # Update the parameters
            self.W = [(1 - ((alpha * lambd) / m)) * w - (alpha / m) * dw for w, dw in zip(self.W, delta_W)]
            self.B = [b - (alpha / m) * db for b, db in zip(self.B, delta_B)]

        return J

    def test(self, test_samples, test_labels):
        A, Z = self.forward_propagation(test_samples)
        Yp = np.argmax([A[-1]], 2)
        accuracy = np.sum(Yp == test_labels) / test_labels.shape[1]
        return (accuracy, Yp)

    def __repr__(self):
        _repr_ = [[0, self.layers[0], 'N/A', 'N/A', 'N/A']] + [[i, self.layers[i], self.activation_functions[i-1].__name__, self.W[i-1], self.B[i-1]] for i in range(1, len(self.layers))]
        return tabulate.tabulate(_repr_, headers=["Layer", "Neurons", "Activation Function", "Weights", "Biases"])

    @staticmethod
    def relu(Z, derivative=False):
        return np.maximum(0, Z) if not derivative else (Z >= 0).astype(float) * np.ones(Z.shape)

    @staticmethod
    def sigmoid(Z, derivative=False):
        g = lambda z : 1. / (1. + np.exp(-z))
        return g(Z) if not derivative else g(Z) * (1 - g(Z))

    @staticmethod
    def tanh(Z, derivative=False):
        return np.tanh(Z) if not derivative else 1 - np.pow(np.tanh(Z), 2)
