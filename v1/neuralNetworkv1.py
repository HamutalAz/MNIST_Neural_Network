import numpy as np
from tqdm import tqdm
import Representation as rep

# our implementation for feed forward, fully connected neural network. The class should receive train set (x) with
# its label (y) as appears in the mnist data-set (raw, without any changes). Additional optional parameters: epochs
# number, layers (must be 3 while first must contain 784 neurons and last 10 neurons).
class NeuralNetwork:

    def __init__(self, X, y, epochs=10, layers=[784, 64, 10]):
        self.input = rep.changeRepresentation(X)
        self.y = rep.makeOneHot(y)
        self.epoch = epochs

        self.weights0 = np.random.rand(layers[1], layers[0]) * (1 / np.sqrt(layers[0]))
        self.weights1 = np.random.rand(layers[2], layers[1]) * (1 / np.sqrt(layers[1]))

        self.activation_inputs = [np.zeros(layers[1]), layers[2]]
        self.layer_inputs = [np.zeros(layers[0]), np.zeros(layers[1])]
        self.output = layers[2]

    # train the model. Each example run threw the model: feedForward and updating the weights
    # accordingly:backPropagation
    def fit(self):
        j = 0
        for i in tqdm(range(self.epoch), desc='main train progress', leave=True):
            for x,y in zip(self.input, self.y):
                pred = self.feedForward(x)
                toUpdate = self.backPropagtion(pred, y)
                self.updateWeights(toUpdate)

    # pass one example in the model
    def feedForward(self, x):

        self.layer_inputs[0] = x
        self.activation_inputs[0] = self.weights0 @ self.layer_inputs[0]

        self.layer_inputs[1] = sigmoid(self.activation_inputs[0])
        self.activation_inputs[1] = self.weights1 @ self.layer_inputs[1]

        self.output = sigmoid(self.activation_inputs[1])

        return self.output

    # calculate the change needed for the weights
    def backPropagtion(self, pred, y):

        delta0, delta1 = [], []

        der_weights1 = np.zeros(self.weights1.shape)   # (10,64)
        der_weights0 = np.zeros(self.weights0.shape)   # (64,784)

        for i in range(10):

            delta1 = 2 * (y[i] - pred[i]) * sigmoid_der(self.activation_inputs[1][i])  #num
            der_weights1[i] = delta1 * self.layer_inputs[1]

            delta0 = delta1 * sigmoid_der(self.activation_inputs[0]) * self.weights1[i]
            der_weights0 += np.outer(delta0, self.layer_inputs[0])   #(64,784)

        return [der_weights0, der_weights1]

    # update the weights according to backPropagation results
    def updateWeights(self, toUpdate):

        self.weights1 += toUpdate[1]
        self.weights0 += toUpdate[0]

    # test the model on the input and return the percent of false prediction from total predictions on the data
    def score(self, X, Y):
        incorrect = 0
        e = 0
        X = rep.changeRepresentation(X)
        Y = rep.makeOneHot(Y)

        for i in range(X.shape[0]):
            self.feedForward(X[i])
            e += (np.argmax(self.output) - np.argmax(Y[i])) ** 2
            if np.argmax(self.output) - np.argmax(Y[i]) != 0:
                incorrect += 1

        return incorrect/X.shape[0]


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp((-1) * x))


# derivative for sigmoid function
def sigmoid_der(x):
    q = sigmoid(x)
    return q * (1 - q)

