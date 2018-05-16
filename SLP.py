from numpy import exp, array, dot
import random

#Our neural network/ single layer perceptron has 3 inputs and 1 output
class NeuralNetwork():
    def __init__(self):
        #A seed is provided so that we can generate the same random numbers each time the program runs
        random.seed(1)

        #We assign 3 random numbers, in a 3x1 matrix, whose values lie between -1 and 1
        self.weights = 2 * random.random((3, 1)) - 1

    #This is our activation function
    #The sigmoid function takes a number 'x', and normalises it between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    #This is the derivative of our activation function, also called the Gradient
    #This shows how confident we're of the current weight values
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    #This is the training set, we try adjusting our weights through each iteration.
    #This is done by trial and error
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            #We pass the input data(training data) through our network
            output = self.predict(training_set_inputs)

            #Error is calculated as: Desired Output - Actual Output
            error = training_set_outputs - output
            
            #To adjust the weights, we do the following:
            #Multiply the error(obtained above) by the input and again by the gradient(derivative) of the Sigmoid Curve
            adjustment = dot(training_set_inputs.T, error * self.sigmoid_derivative(output))

            #We Adjust the weights
            self.weights += adjustment
    
    #This is where the neural network starts 'predicting'
    def predict(self, inputs):
        #We pass the input data/training data) through our network
        return self.sigmoid(dot(inputs, self.weights))


if __name__ == "__main__":
    #Initialize a single layer perceptron
    neural_network = NeuralNetwork()
    
    #We start with Randomized weights
    print("Random starting synaptic weights: ")
    print(neural_network.weights)

    #We take four 3x1 matrices as our imput(training) data
    #And 1 output value, which is like the answer to the problem we're trying to solve/model
    training_set_inputs = array([[1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    #Train the network with the training data
    #Here we're doing the training 10,000 times
    #With each run, we make adjustments to our weights
    #Try different values, and see how this helps our output
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    #We print our weights after the training has been finished
    #Compare the weights from numerous runs, to see variations
    print("New synaptic weights after training: ")
    print(neural_network.weights)

    #Here we can ask the network to finally predict values for a given input
    print("Predicting labels for:  [0, 1, 0] -> ?: ")
    print(neural_network.predict(array([0, 1, 0])))