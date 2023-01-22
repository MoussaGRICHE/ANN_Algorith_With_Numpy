import numpy as np

def Relu(x):
    return np.maximum(x, 0)

def Relu_der(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x

def Softmax(x, axis=-1):
    kw = dict(axis=axis, keepdims=True)
    xrel = x - x.max(**kw) # to avoid the problem of "overflow encountered in exp"
    return np.exp(xrel) / np.exp(xrel).sum(**kw)

def Softmax_der(x):  # Softmax derivative
    I = np.eye(x.shape[0])
    return Softmax(x) * (x - (x * Softmax(x).sum()))

def weight_initialization(input_shape, output_shape):
    W = np.random.rand(input_shape, output_shape) * np.sqrt(2 / (input_shape + output_shape))
    return W

def Cross_Entropy(y_true, y_pred):
    epsilon = 0.0000000000001
    log_losse = np.sum(y_true * np.log(y_pred + epsilon))
    return (-log_losse)/float(y_pred.shape[0])

def Confusion_Matrix(y_true, y_pred):
    # confusion matrix
    num_categories = len(y_true)
    y_true = np.ndarray.tolist(y_true)
    y_pred = np.ndarray.tolist(y_pred)
    # Create an empty confusion matrix
    confusion_matrix = np.zeros((num_categories, num_categories))

    # Fill the confusion matrix with counts
    for i in range(len(y_true)):
        if int(y_true[i]) == 1 and float(y_pred[i]) >= 0.6:
            confusion_matrix[i, i] += 1
    return confusion_matrix



def model_parameter(x, y, NP):
    t=[x.shape[1]] + NP + [y.shape[1]]
    print('#####################################################################')
    print('######################### Model Parameters  #########################')
    print('#####################################################################','\n')
    print('The total number of parameters required can be computed as follows:')
    for i in range(len(t)-1):
      par = t[i] * t[i+1] + t[i+1]
      if i==0:
        print("From input to Dense layer number"+" "+str(i+1)+":", par)
      elif i !=i+1 and i !=len(t)-2:
        print("From Dense layer number" +" " + str(i) +" "+ "to the Dense layer number"+" "+ str(i+1)+":", par)
      else:
        print("From Dense layer number"+" "+ str(i) +" "+ "to the output layer :", par, '\n')

    print('#####################################################################', '\n')