from Forward_Backward import *
import pandas as pd
from keras.datasets import mnist
import numpy as np

########################################################################################################################
# Data preparation
########################################################################################################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

num_labels = len(np.unique(y_train))

y_train = pd.get_dummies(y_train, columns = num_labels)
y_test = pd.get_dummies(y_test, columns = num_labels)

y_train = np.array(y_train)
y_test = np.array(y_test)


########################################################################################################################
# The Model Architecture
########################################################################################################################
def Model_Architecture():
    """
    This function ask the user the number of hidden layers and the length of each hidden layer.
    :return: It prints the model architecture and returns a list with the length of each hidden layer
    """
    HL = input('How many hidden layer do you want in you model: ')
    HL = int(HL)

    NP = []
    for i in range(HL):
        NP.append(input("how many persoprton do you want in the hidden layer number" + " " + str(i + 1) + " :"))
    NP=list(map(int, NP))
    ML = [x_train.shape[1]] + NP + [y_train.shape[1]]

    print('#####################################################################')
    print('######################### Model Architecture #########################')
    print('#####################################################################','\n')

    for i in range(len(ML) - 1):
        if i == 0:
            print("l" + str(i + 1) + "= model_layer(" + str(ML[i]) + "," + str(ML[i + 1]) + "," " Input Layer)")
        elif i == max(range(len(ML) - 1)):
            print("l" + str(i + 1) + "= model_layer(" + str(ML[i]) + "," + str(ML[i + 1]) + "," " ""Output Layer"")")
        else:
            print("l" + str(i + 1) + "= model_layer(" + str(ML[i]) + "," + str(ML[i + 1]) + "," " Hidden Layer)")
    print('#####################################################################','\n')
    return NP


########################################################################################################################
# The model training
########################################################################################################################
def model_training(x, y):
    """
    This function trains the model by launching the weights_biases_init, Forward Propagation, Backward Propagation functions.
    :param x: The input data
    :param y: The target data
    :return: The accuracy of the fitting which is calculated from the confusion matrix of the predicted results.
    """
    batch_size = 128
    epochs = 10
    losss = []
    for ep in range(epochs):
        error_list = []
        K = len(y[1])  # Number of classes
        Con_Mat = np.zeros((K, K))
        seen_points = 0
        error = 0
        accuracy = 0
        M = {}
        for i in range(x.shape[0]):
            Model.weights_biases_init()

            Model.Forward_Propagation(x[i])

            Model.Backward_Propagation(y[i])

            error = np.mean(Model.output_error**2)
            error_list.append(error)

            if seen_points % batch_size == 0:
                Model.Adam()

            CM = Confusion_Matrix (y[i], Model.output_values)

            for o in range(len(Con_Mat)-1):
                for v in range(len(CM)-1):
                    Con_Mat[o][v] = Con_Mat[o][v] + CM[o][v]


        loss = sum(error_list) / x.shape[0] / batch_size

        losss.append(error)
        accuracy = float(np.trace(Con_Mat)) / (len(y))

        print("Epochs:", ep + 1, "/", epochs, "[=======================] - loss:", loss, "- acc :", accuracy)

    return losss, accuracy

########################################################################################################################
# The model execution
########################################################################################################################
if __name__ == "__main__":
    Architecture = Model_Architecture()
    parameter = model_parameter(x_train, y_train, Architecture)
    Model = model_layer(x_train.shape[1], Architecture, y_train.shape[1])
    Results = model_training(x_train, y_train)
