import numpy as np
from ANN_functions import *

class model_layer:
    update_count = 0
    delta = 0

    def __init__(self, input_shape, hidden_shape=list(), output_shape=0):

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.input_weights = 0
        self.hidden_weights = {}
        self.output_weights = 0
        self.input_biases = 0
        self.hidden_biases = {}
        self.output_biases = 0
        self.input_values = 0
        self.hidden_values = {}
        self.output_values = 0
        self.input_error = 0
        self.hidden_error = {}
        self.output_error = 0
        self.input_gamma = 0
        self.hidden_gamma = {}
        self.output_gamma = 0
        self.input_delta = 0
        self.hidden_delta = {}
        self.output_delta = 0
        self.input_sw = 0
        self.hidden_sw = {}
        self.output_sw = 0
        self.hidden_swc = {}
        self.output_swc = 0
        self.input_vw = 0
        self.hidden_vw = {}
        self.output_vw = 0
        self.hidden_vwc = {}
        self.output_vwc = 0
        self.input_sb = 0
        self.hidden_sb = {}
        self.output_sb = 0
        self.hidden_sbc = {}
        self.output_sbc = 0
        self.input_vb = 0
        self.hidden_vb = {}
        self.output_vb = 0
        self.hidden_vbc = {}
        self.output_vbc = 0
        self.update_count = 0
        self.hidden_values = {}

    def weights_biases_init(self):

        self.input_weights = weight_initialization(self.input_shape, self.hidden_shape[0])
        for i in range(len(self.hidden_shape)-1):
            self.hidden_weights[str(i)] = weight_initialization(self.hidden_shape[i], self.hidden_shape[i+1])
        self.output_weights = weight_initialization(self.hidden_shape[len(self.hidden_shape)-1], self.output_shape)

        self.input_biases = np.random.random()
        for i in range(len(self.hidden_shape)-1):
            self.hidden_biases[str(i)] = np.random.random()
        self.output_biases = np.random.random()

    def Forward_Propagation(self, x):
        self.input_values = x

        self.input_values = Relu(np.dot(self.input_values, self.input_weights) + self.input_biases)

        for i in range(len(self.hidden_shape)-1):
            if i == 0:
                self.hidden_values[str(i)] = Relu(np.dot(self.input_values, self.hidden_weights[str(i)]) + self.hidden_biases[str(i)])
            else:
                self.hidden_values[str(i)] = Relu(np.dot(self.hidden_values[str(i-1)], self.hidden_weights[str(i)]) + self.hidden_biases[str(i)])


        self.output_values = Softmax(np.dot(self.hidden_values[str(len(self.hidden_values)-1)],self.output_weights)+ self.output_biases)

        return self.output_values

    def Backward_Propagation(self, expected=0, next_layer_gamma=0, next_layer_weights=0):

        self.output_error = Cross_Entropy(expected, self.output_values)
        self.output_gamma = Softmax_der(self.output_values)

        for i in reversed(range(len(self.hidden_shape)-1)):
            if i == (len(self.hidden_shape)-2):
                self.hidden_error[str(i)] = np.dot(self.output_gamma, self.output_weights.T)
                self.hidden_gamma[str(i)] = self.hidden_error[str(i)] * Relu_der(self.hidden_values[str(i)])
            else:
                self.hidden_error[str(i)] = np.dot(self.hidden_gamma[str(i+1)], self.hidden_weights[str(i+1)].T)
                self.hidden_gamma[str(i)] = self.hidden_error[str(i)] * Relu_der(self.hidden_values[str(i)])


        self.input_error = np.dot(self.hidden_gamma['0'], self.hidden_weights['0'].T)
        self.input_gamme = self.input_error * Relu_der(self.input_values)

        self.input_delta = np.dot(self.input_values.T, self.input_gamma)
        for i in range(len(self.hidden_shape)-1):
            self.hidden_delta[str(i)] = np.dot(self.hidden_values[str(i)].T, self.hidden_gamma[str(i)])
        self.output_delta = np.dot(self.output_values.T, self.output_gamma)

        self.input_sb = np.mean(self.hidden_gamma['0'], axis=0)
        for i in range(len(self.hidden_shape)-1):
            self.hidden_sb[str(i)] = np.mean(self.hidden_gamma[str(i)], axis=0)
        self.output_sb = np.mean(self.output_gamma, axis=0)


    def Adam(self, beta1=0.9, beta2=0.99, epsilon=0.00000001, lr=0.01):
        self.update_count += 1

        ## Input layer weights correction
        self.input_sw = self.input_sw * beta1 + self.input_delta * (1 - beta1)
        self.input_swc = self.input_sw / (1 - beta1 ** self.update_count)
        self.input_vw = self.input_vw * beta2 + self.input_delta ** 2 * (1 - beta2)
        self.input_vwc = self.input_vw / (1 - beta2 ** self.update_count)
        self.input_weights -= lr * self.input_swc / (np.sqrt(self.input_vwc) + epsilon)

        ## Input layer biases correction
        self.input_sb = self.input_sb * beta1 + self.input_sb * (1 - beta1)
        self.input_sbc = self.input_sb / (1 - beta1 ** self.update_count)
        self.input_vb = self.input_vb * beta2 + self.input_sb ** 2 * (1 - beta2)
        self.input_vbc = self.input_vb / (1 - beta2 ** self.update_count)
        self.input_biases -= lr * self.input_sbc / (np.sqrt(self.input_vbc) + epsilon)


        ## Hidden layer weights correction
        for i in range(len(self.hidden_shape) - 1):
            self.hidden_sw[str(i)] = 0
            self.hidden_vw[str(i)] = 0
            self.hidden_sw[str(i)] = self.hidden_sw[str(i)] * beta1 + self.hidden_delta[str(i)] * (1 - beta1)
            self.hidden_swc[str(i)] = self.hidden_sw[str(i)] / (1 - beta1 ** self.update_count)
            self.hidden_vw[str(i)] = self.hidden_vw[str(i)] * beta2 + self.hidden_delta[str(i)] ** 2 * (1 - beta2)
            self.hidden_vwc[str(i)] = self.hidden_vw[str(i)] / (1 - beta2 ** self.update_count)
            self.hidden_weights[str(i)] -= lr * self.hidden_swc[str(i)] / (np.sqrt(self.hidden_vwc[str(i)]) + epsilon)

        ## Hidden layer biases correction
        for i in range(len(self.hidden_shape) - 1):
            self.hidden_sb[str(i)] = 0
            self.hidden_vb[str(i)] = 0
            self.hidden_sb[str(i)] = self.hidden_sb[str(i)] * beta1 + self.hidden_sb[str(i)] * (1 - beta1)
            self.hidden_sbc[str(i)] = self.hidden_sb[str(i)] / (1 - beta1 ** self.update_count)
            self.hidden_vb[str(i)] = self.hidden_vb[str(i)] * beta2 + self.hidden_sb[str(i)] ** 2 * (1 - beta2)
            self.hidden_vbc[str(i)] = self.hidden_vb[str(i)] / (1 - beta2 ** self.update_count)
            self.hidden_biases[str(i)] -= lr * self.hidden_sbc[str(i)] / (np.sqrt(self.hidden_vbc[str(i)]) + epsilon)

        ## Output layer weights correction
        self.output_sw = self.output_sw * beta1 + self.output_delta * (1 - beta1)
        self.output_swc = self.output_sw / (1 - beta1 ** self.update_count)
        self.output_vw = self.output_vw * beta2 + self.output_delta ** 2 * (1 - beta2)
        self.output_vwc = self.output_vw / (1 - beta2 ** self.update_count)
        self.output_weights -= lr * self.output_swc / (np.sqrt(self.output_vwc) + epsilon)

        ## Output layer biases correction
        self.output_sb = self.output_sb * beta1 + self.output_sb * (1 - beta1)
        self.output_sbc = self.output_sb / (1 - beta1 ** self.update_count)
        self.output_vb = self.output_vb * beta2 + self.output_sb ** 2 * (1 - beta2)
        self.output_vbc = self.output_vb / (1 - beta2 ** self.update_count)
        self.output_biases -= lr * self.output_sbc / (np.sqrt(self.output_vbc) + epsilon)
