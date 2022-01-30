import numpy as np, tabulate

class NanoNeuralNetwork:
    def __init__(self, layers, activation_functions=None, hyperparameters=None):
        """
        layers               - List of integers representing neurons per layer of the network
        activation_functions - List of activation functions to be used per layer
        hyperparameters      - Dictionary of hyperparameter settings. Supported hyperparameters
            - 'learning_rate'
            - 'weight_decay'
        """

        self.layers = layers
        self.activation_functions = activation_functions if activation_functions is not None else [NanoNeuralNetwork.relu for l in self.layers[1:]]
        self.activation_functions[-1] = NanoNeuralNetwork.sigmoid
        self.hyperparameters = hyperparameters if hyperparameters is not None else {'learning_rate': 0.1, 'weight_decay': 0}

        # Create weight+bias matrices per layer initialized to 0
        self.W = [np.zeros((n, m)) for n, m in zip(self.layers[1:], self.layers[:-1])]
        self.B = [np.zeros((1, n)) for n in self.layers[1:]]

    def forward_propagation(self, X):
        """
        Input:
        X - A matrix where each row is a sample and each column is a feature
        Output:
        A - A list of matrices containing the activations after each layer. The matrices in the list are composed of rows of samples and columns of activations per neuron.
        Z - A list of matrices containing the dot products after each layer.

        This routine will perform the forward propagation step and produces the dot products and activations after each layer. The formula implemented as follows:
        Z[l] = A[l-1] * W[l]_T + B[l]
        A[l] = g(Z[l])
        where 'l' is the layer number and l = 0 is the input layer containing the feature vectors from the dataset. 'g(z)' is the activation function.
        """

        # Forward propagation step. Store the dot products, Z, and the activations, A, for each layer
        A = [X]
        Z = [None]

        for w, b, act_fn in zip(self.W, self.B, self.activation_functions):
            Z += [np.dot(A[-1], w.T) + b]
            A += [act_fn(Z[-1])]

        return (A, Z)

    def backward_propagation(self, Y, A, Z):
        """
        Input:
        Y       - A matrix where each row is a sample containing the output labels for the dataset
        A       - A list of matrices containing the activations after each layer. The matrices in the list are composed of rows of samples and columns of activations per neuron.
        Z       - A list of matrices containing the dot products after each layer.
        Output:
        delta_W - A list of gradients for the weights for each layer
        delta_B - A list of gradients for the biases for each layer

        This routine performs the backward propagation step and produces the gradients for the weights and biases per layer. The optimization algorithm is gradient descent.
        The calculations performed as follows:
        dW[l] = dZ[l].T * A[l-1]
        dB[l] = sum(dZ[l])
        dZ[l-1] = (dZ[l] * W[l]) .* g_p(Z[l-1])
        where 'l' is the layer number and l = 0 is the input layer containing the feature vectors from the dataset. 'g_p(z)' is the derivative of the activation function.
        """

        # Backward propagation step. Store the weight, delta_W, and bias, delta_B, deltas for each layer
        delta_Z = A[-1] - Y
        delta_W = []
        delta_B = []

        for w, a, z, act_fn  in reversed(list(zip(self.W, A[:-1], Z[:-1], self.activation_functions))):
            delta_W = [np.dot(delta_Z.T, a)] + delta_W
            delta_B = [np.sum(delta_Z, axis=0, keepdims=True)] + delta_B
            delta_Z = np.dot(delta_Z, w) * act_fn(z, derivative=True) if z is not None else None

        return delta_W, delta_B

    def cost(self, Y, Yp):
        """
        Input:
        Y    - A matrix where each row is a sample containing the output labels for the dataset
        Yp   - A matrix containing the output layer result vector of the neural network model containing the estimates of Y
        Output:
        cost - Cross-entropy cost of the estimates against the actual result
        """

        m = Y.shape[0]
        return np.squeeze((-1 / m) * (np.dot(Y.T, np.log(Yp)) + np.dot((1 - Y).T, np.log(1 - Yp))))

    def train(self, train_samples, train_labels, iterations=1000):
        """
        Input:
        train_samples - Training dataset containing rows of samples where each row is a feature vector
        train_labels  - Actual correct mappings of the dataset for each feature vector
        iterations    - Number of iterations to perform the gradient descent algorithm
        Output:
        J             - An array of cross-entropy costs per iteration

        The train follows a straightforward sequence:
        0. Initialize the weights using random values from a standard normal distribution modified by a scaling factor of sqrt(2 / N[l]) where N[l] is the number of
        input activations for the given layer 'l'
        1. Perform forward propagation on the entire set of training samples
        2. Calculate the cost of the forward propagation results against the correct values
        3. Perform backward propagation on the entire set of training samples to calculate gradients used to update the weights/biases
        Steps 1-3 are performed in a loop for the given number of iterations.
        """
        m = train_labels.shape[0]
        alpha = self.hyperparameters['learning_rate']
        lambd = self.hyperparameters['weight_decay']
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
        """
        Input:
        test_samples - Test dataset containing rows of samples where each row is a feature vector
        test_labels  - Actual correct mappings of the dataset for each feature vector
        Output:
        accuracy     - Accuracy of the neural network model against the test dataset
        Yp           - Output layer result matrix produced by the neural network model

        The output layer vector values will be rounded to 0 or 1.
        """

        A, Z = self.forward_propagation(test_samples)
        Yp = np.round(A[-1])
        accuracy = np.sum(Yp == test_labels) / test_labels.shape[0]
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
