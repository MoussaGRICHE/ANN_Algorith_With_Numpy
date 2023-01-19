import numpy as np
from ANN_functions import *

class model_layer:
    update_count = 0
    delta = 0

    def __init__(self, input_shape, output_shape, Layer):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.Layer = Layer
        self.weights = weight_initialization(self.input_shape, self.output_shape)
        self.biases = np.random.random()
        self.sw = 0
        self.vw = 0
        self.sb = 0
        self.vb = 0


    def Forward_Propagation(self, x):
        self.input_values = x
        if self.Layer == "Input":
            self.output = Relu(np.dot(self.input_values, self.weights) + self.biases)

        elif self.Layer == "Hidden":
            self.output = Relu(np.dot(self.input_values, self.weights) + self.biases)

        elif self.Layer == "Output":
            self.output = Softmax(np.dot(self.input_values, self.weights) + self.biases)

        return self.output

    def Backward_Propagation(self, expected=0, next_layer_gamma=0, next_layer_weights=0):
        if self.Layer == "Input":
            self.error = np.dot(next_layer_gamma, next_layer_weights.T)
            self.gamma = self.error * Relu_der(self.output)

        elif self.Layer == "Hidden":
            self.error = np.dot(next_layer_gamma, next_layer_weights.T)
            self.gamma = self.error * Relu_der(self.output)

        elif self.Layer == "Output":
            self.error = Cross_Entropy(expected, self.output)
            self.gamma = self.error * Softmax_der(self.output)

        self.delta += np.dot(self.input_values.T, self.gamma)

        self.db = np.mean(self.gamma, axis=0)

    def Adam(self, beta1=0.9, beta2=0.99, epsilon=0.00000001, lr=0.01):
        self.update_count += 1
        ## Weights correction
        self.sw = self.sw * beta1 + self.delta * (1 - beta1)
        self.swc = self.sw / (1 - beta1 ** self.update_count)

        self.vw = self.vw * beta2 + self.delta ** 2 * (1 - beta2)
        self.vwc = self.vw / (1 - beta2 ** self.update_count)

        self.weights -= lr * self.swc / (np.sqrt(self.vwc) + epsilon)

        ## biases correction
        self.sb = beta1 * self.sb + (1 - beta1) * self.db
        self.sbc = self.sb / (1 - beta1 ** self.update_count)

        self.vb = beta2 * self.vb + (1 - beta2) * self.db
        self.vbc = self.vb / (1 - beta2 ** self.update_count)

        self.biases -= lr * self.sbc / (np.sqrt(self.vbc) + epsilon)

        self.delta = 0
        return self.weights , self.biases


