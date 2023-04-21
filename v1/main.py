import neuralNetworkv1
import mnist

def getMnist():
    train_X = mnist.train_images()
    train_Y = mnist.train_labels()
    test_X = mnist.test_images()
    test_Y = mnist.test_labels()

    return train_X, train_Y, test_X, test_Y


def main():

    train_X, train_y, test_X, test_y = getMnist()
    nn = neuralNetworkv1.NeuralNetwork(train_X, train_y)
    nn.fit()
    score = nn.score(test_X, test_y)

    print("incorrect rate:", score)


if __name__ == "__main__":
    main()
