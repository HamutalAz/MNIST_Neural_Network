import numpy as np
from tqdm import tqdm
import Representation as rep
import shape
from sklearn.model_selection import train_test_split


# our implementation for feed forward, fully connected neural network. The class should receive train set (x) with
# its label (y) as appears in the mnist data-set (raw, without any changes). Additional optional parameters: epochs
# number, layers (must be 4 while first must contain 784 neurons and last 10 neurons) and learning rate.
class NeuralNetwork:

    def __init__(self, X, y, epochs=54, layers=[784, 254, 64, 10], lr=0.1):

        self.input = rep.changeRepresentation(X)
        self.y = rep.makeOneHot(y)

        self.epoch = epochs
        self.layers = layers
        self.lr = lr

        self.weights0 = np.random.rand(layers[1], layers[0]+1) * (1 / np.sqrt(layers[0]))     # (254,785)
        self.weights1 = np.random.rand(layers[2], layers[1]+1) * (1 / np.sqrt(layers[1]))     # (64,255)
        self.weights2 = np.random.rand(layers[3], layers[2]+1) * (1 / np.sqrt(layers[2]))   # (10,65)

        self.activation_inputs = [np.zeros(layers[1]), np.zeros(layers[2]), np.zeros(layers[3])]
        self.layer_inputs = [np.zeros(layers[0]+1), np.zeros(layers[1]+1), np.zeros(layers[2]+1)]

        self.output = None
        self.bias = [1, 1, 1]


    # train the model. Each example run threw the model: feedForward and updating the weights
    # accordingly:backPropagation
    def fit(self):

        losses = []
        X_train, X_valid, y_train, y_valid = train_test_split(self.input, self.y, test_size=0.15, random_state=42)

        #for i in tqdm(range(self.epoch), desc='main train progress', position=0, leave=True):
        for i in range(self.epoch):
            j = 1
            #for x,y in (zip(X_train, y_train)):
            for x,y in tqdm((zip(X_train, y_train)),desc='Epoch {} of {}'.format(i+1, self.epoch), position=0, total=len(X_train)):
                pred = self.feedForward(x)
                toUpdate = self.backPropagtion(pred, y)
                self.updateWeights(toUpdate, j)
                j += 1
            loss = self.computeLoss(X_valid, y_valid)
            losses.append(loss)

        shape.plot(np.arange(1,self.epoch + 1), losses, "number of epoch", "loss")

    # pass one example in the model
    def feedForward(self, x):

        self.layer_inputs[0] = np.r_[self.bias[0], x]
        self.activation_inputs[0] = self.weights0 @ self.layer_inputs[0]

        self.layer_inputs[1] = np.r_[self.bias[1],sigmoid(self.activation_inputs[0])]
        self.activation_inputs[1] = self.weights1 @ self.layer_inputs[1]

        self.layer_inputs[2] = np.r_[self.bias[2], sigmoid(self.activation_inputs[1])]
        self.activation_inputs[2] = self.weights2 @ self.layer_inputs[2]

        self.output = sigmoid(self.activation_inputs[2])

        return self.output

    # calculate the change needed for the weights
    def backPropagtion(self, pred, y):

        der_weights2 = np.zeros(self.weights2.shape)   # (10,65)
        der_weights1 = np.zeros(self.weights1.shape)   # (64,255)
        der_weights0 = np.zeros(self.weights0.shape)   # (254,785)

        for i in range(self.layers[3]):

            delta2 = 2 * (pred[i] - y[i]) * sigmoid_der(self.activation_inputs[2][i])  # num
            der_weights2[i] = delta2 * self.layer_inputs[2]

            delta1 = delta2 * sigmoid_der(self.activation_inputs[1]) * self.weights2[i, 1:]
            der_weights1 += np.outer(delta1, self.layer_inputs[1])

            delta0 = (self.weights1[:, 1:].T @ delta1) * sigmoid_der(self.activation_inputs[0])
            der_weights0 += np.outer(delta0, self.layer_inputs[0])

        return [der_weights0, der_weights1, der_weights2]

    # update the weights according to backPropagation results
    def updateWeights(self, dw, t):

        self.weights2 -= self.lr * dw[2]
        self.weights1 -= self.lr * dw[1]
        self.weights0 -= self.lr * dw[0]

    # test the model on the input and return the percent of false prediction from total predictions on the data
    def score(self, X, Y):

        incorrect = 0
        X = rep.changeRepresentation(X)
        Y = rep.makeOneHot(Y)

        for i in range(X.shape[0]):
            self.feedForward(X[i])
            if np.argmax(self.output) - np.argmax(Y[i]) != 0:
                incorrect += 1

        return incorrect/X.shape[0]

    # compute l2 loss
    def computeLoss(self, X, Y):

        e = 0
        for i in range(X.shape[0]):
            self.feedForward(X[i])
            e += (np.argmax(self.output) - np.argmax(Y[i])) ** 2

        return e/X.shape[0]

    # save the model to a file which it's name is given by fileName
    def save(self, fileName):

        np.savez(fileName, self.weights0, self.weights1, self.weights2)

    # upload the model from a file which it's name is given by fileName
    def load(self, fileName):

        npzfile = np.load(fileName)
        self.weights0 = npzfile[npzfile.files[0]]
        self.weights1 = npzfile[npzfile.files[1]]
        self.weights2 = npzfile[npzfile.files[2]]


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp((-1) * x))


# derivative for sigmoid function
def sigmoid_der(x):
    q = sigmoid(x)
    return q * (1 - q)
