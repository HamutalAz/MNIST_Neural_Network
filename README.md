**MNIST Neural Network**

I created this project collaborating with [Michal Golan](https://github.com/MichalGolan) as part of an Intro to Machine Learning course we participated in while pursuing our BSc. in Computer Science.

We created a Neural Network from scratch using Stochastic gradient descent and train it on the MNIST dataset. We managed to get 98.06% accuracy with our model, after training it for 54 epochs, without using convolutional layers.

**MNIST Dataset**

The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. The database has a training set of 60,000 examples and a test set of 10,000 examples. In the mixed model of convolutional neural networks and capsule networks, the researchers establishes a state-of-the-art for the MNIST dataset with an accuracy of 99.84%.[[1](https://analyticsindiamag.com/image-classification-models-benchmark-state-of-the-art/%5C)]

**The Final Network Architecture**

- We created a feed-forward, fully connected neural network having layers: 784 -> 254 -> 64 -> 10.
- We added bias neurons to each level, excluding the output level.
- We used the Sigmoid function for the activation layers. 
- We used L2 loss to evaluate the model performance.

**Our Process**

First, we implemented the derivative calculation, the backpropagation, and the weights updates from scratch. We used Numpy for working with vectors, Matplotlib for graph plotting, and SkLearn for a train-test split.

**Preliminary steps**

First, we created a toy problem using make\_blobs. In every version of the model described below, we adjust the number of features and centers to the network structure in the specific run.

Before building the final model, our preliminary steps were:

1. We built a basic network structure that its input is 3, has a hidden layer of 4 neurons, and returns a single output.
1. We expand the network input to receive 784 features while the other layers remain unchanged.
1. We expand our hidden layer to have 64 neurons and normalized the weights.
1. We added another neuron to the output layer, which changed the output type from a number to a vector. This change affected the Backpropagation and enabled us to check if the network is capable classify two centers: 0 and 1.
1. At last, we expanded the output layer to have 10 neurons and check if the network can classify 10 different centers.

After these steps, we had a basic network structure, that capable of classifying 10 different groups with formation of 784->64->10.

This model was used to train MNIST dataset.

**Pre-Processing**

- We altered the data representation so every sample, represented by a 28\*28 Matrix, will be transformed into a vector sized 784.
- We normalized the vector by division to its max possible value (255).
- We changed the target representation to a one-hot vector.

We trained the described model for 10 epochs on the MNIST dataset and got 92.74% accuracy on the test set. This is the first version of our model (which can be found in the v1 directory).

**Our attempts to improve the model:**

1. We added bias neurons to each level, excluding the output level, and got 93.95% accuracy.
1. We added another hidden layer with 254 neurons + one bias neuron and got a new network structure: (784+1)->(254+1)->(64+1)->10.
1. The above structure didn’t get us any result while training 10 epochs with learning rate of 1. Therefore, we lowered our training rate to 0.1 and achieved 96.4% accuracy on the test set.
1. At this point, we realized the importance of the learning rate in the network structure we reached and made several experiments while aiming to achieve the best learning rate for the problem.

<img width="1440" alt="Screenshot 2023-04-21 at 12 42 07" src="https://user-images.githubusercontent.com/76840545/233612875-5848be18-acc0-4435-a8a6-df8d203da242.png">

` `Based on the graphs above, we set the learning rate to 0.1.

5. To find the optimal epochs number for generalization, we trained the model with 100 epochs on the training set, with the chosen learning rate.

<img width="1440" alt="Screenshot 2023-04-21 at 12 42 48" src="https://user-images.githubusercontent.com/76840545/233612899-ba4f1177-0fd0-447e-9fb4-1c8a837562e2.png">

At this run, we got 98.81% accuracy with 100 epochs. However, we noticed that after 50 epochs, the loss stabilized. Therefore, we decided to set the epochs number to 54.

Finally, we got to the described model and got 98.06% accuracy.

**How to run the model**

Our final model can be found in v2\myModel.npz

You can upload it and use it by running the method load(“myModel.npz”) that can be found in NeuralNetwork class.

**What can be improved?**

Convolutional layers can be added to the model and may improve the model's accuracy.

2\. Initializing the weights using Adam optimizer.

3\. Changing the learning rate throughout the training process.
