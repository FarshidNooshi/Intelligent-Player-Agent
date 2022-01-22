import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.layer_sizes = layer_sizes
        self.parameters = self.initialize_parameters_deep()

    @staticmethod
    def activation(x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        return 1.0 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        a = x
        deepness = len(self.parameters) // 2

        for le in range(1, deepness):
            a_prev = a
            a = self.linear_activation_forward(a_prev, self.parameters['W' + str(le)], self.parameters['b' + str(le)])

        al = self.linear_activation_forward(a, self.parameters['W' + str(deepness)],
                                            self.parameters['b' + str(deepness)])

        return al

    def linear_activation_forward(self, a_prev, w, b):
        z = (w @ a_prev) + b
        a = self.activation(z)
        return a

    def initialize_parameters_deep(self):
        parameters = {}
        deepness = len(self.layer_sizes)  # number of layers in the network

        for le in range(1, deepness):
            parameters['W' + str(le)] = np.random.normal(size=(self.layer_sizes[le], self.layer_sizes[le - 1]))
            parameters['b' + str(le)] = np.zeros((self.layer_sizes[le], 1))

        return parameters
