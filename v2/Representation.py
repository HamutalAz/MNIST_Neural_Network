import numpy as np


# reshape the data from 28*28 matrix to 784 vector & normalize the data by dividing by 255
def changeRepresentation(X):
    X = (X / 255).astype('float32')
    res = []
    for x in X:
        res.append(x.reshape(784))

    return np.array(res)


# change the labels representation from number to vector of len 10 with 1 in index corresponding to the number
def makeOneHot(Y):

    res = []
    for y in Y:
        newY = np.zeros(10)
        newY[y] = 1
        res.append(newY)

    return np.array(res)