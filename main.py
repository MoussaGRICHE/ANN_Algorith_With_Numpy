from Forward_Backward import *
import pandas as pd
import itertools
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
# The model parameters
########################################################################################################################
HL = input('How many hidden layer do you want in you model: ')
HL = int(HL)

NP=[]
for i in range(HL):
   NP.append(input("how many persoprton do you want in the hidden layer number"+" "+str(i+1)+" :"))
NP=list(map(int, NP))

ML = [x_train.shape[1]]+NP+[y_train.shape[1]]

Model={}
for i in range(len(ML)-1):
   if i == 0:
     Model[str(i+1)] = model_layer(ML[i], ML[i+1], "Input")
     print ("l"+ str(i+1) + "= model_layer("+str(ML[i])+","+ str(ML[i+1])+"," " Input)")
   elif i == max(range(len(ML)-1)):
     Model[str(i+1)] = model_layer(ML[i], ML[i+1], "Output")
     print ("l"+ str(i+1) + "= model_layer("+str(ML[i])+","+ str(ML[i+1])+"," " ""Output"")")
   else:
     Model[str(i+1)] = model_layer(ML[i], ML[i+1], "Hidden")
     print ("l"+ str(i+1) + "= model_layer("+str(ML[i])+","+ str(ML[i+1])+"," " Hiden)")


########################################################################################################################
# The model training
########################################################################################################################


def model_training(x, y):
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
            for j in range(len(Model)):
                if (j == 0):
                    M[str(j+1)] = Model[str(j+1)].Forward_Propagation(x[i].reshape(1, -1))
                else:
                    M[str(j + 1)] = Model[str(j+1)].Forward_Propagation(M[str(j)])

            for k in reversed(range(len(Model))):
                if (k == len(Model)-1):
                    Model[str(k+1)].Backward_Propagation(y[i])
                else:
                    Model[str(k+1)].Backward_Propagation(next_layer_gamma=Model[str(k+2)].gamma, next_layer_weights=Model[str(k+2)].weights)


            error = np.mean(Model[str(len(Model))].error**2)
            error_list.append(error)

            if seen_points % batch_size == 0:
                for s in reversed(range(len(Model))):
                    Model[str(s + 1)].Adam()


            CM = np.array(Confusion_Matrix (y[i], list(itertools.chain.from_iterable(Model[str(len(Model))].output))))
            for o in range(len(y[i])):
                for p in range(len(y[i])):
                    Con_Mat[o][p] += CM[o][p]

            seen_points += 1
        los = sum(error_list) / x.shape[0] / batch_size

        losss.append(error)
        accuracy = float(np.trace(Con_Mat)) / (len(y))

        print("Epochs:", ep + 1, "/", epochs, "[=======================] - loss:", los, "- acc :", accuracy)

    return losss, accuracy

########################################################################################################################
# The model execution
########################################################################################################################
if __name__ == "__main__":
    parameter = model_parameter(x_train, y_train, NP)
    Results = model_training(x_train, y_train)
