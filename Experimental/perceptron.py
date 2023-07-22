# This code implements a perceptron
import numpy as np

class Perceptron:
    def __init__(self, number_outputs, learning_rate=0.01, epochs = 10000) -> None:
        self.number_outputs = number_outputs
        self.learning_r = learning_rate
        self.epochs = epochs
        self.activation_func = self.unit_step_activation
        self.weights = []
        self.bias = None

    # This step function returns 1 where x > 0 and return 0 otherwise:
    def unit_step_activation(self, x):
        return np.where(x >= 1, 1, 0)

    # do the prediction
    def predict_function(self, X, weight_j = 0):
        
        print("About to do the dot product between input", X," and ", self.weights[weight_j])
        linear_output = np.dot(X, self.weights[weight_j])
        print("And the result from the dot product above is ", linear_output)
        y_predicted = self.activation_func(linear_output)
        print("However, our activation function maps this value to ", y_predicted, "aka: y sub j for this j")

        print("**************************")
        print("")
        print("")
        return y_predicted

    # define fit method with x and y
    def train(self, X, train_vector):
        # X is a 2d array size m*n where m=num of rows(samples), n=cols(features)
        samples, features = X.shape  # initializing number of samples and number of features

        # initialize the weights
        print("Initializing the weights and will print them as they get initialized")
        for i in range(self.number_outputs):
            self.weights.append(np.zeros(features))
            # clean as matrix for printing
            y_print = np.array([1 if i > 0 else 0 for i in self.weights[i]])
            print(y_print, " ----> weight", i+1)

        y_ = np.array([1 if i > 0 else 0 for i in train_vector]) # make sure nothing passes that is not 1's and 0's as training values

        # # start training
        for epoch in range(self.epochs):
            print("training at epoch ........", epoch)
            print("About to dot the sample with all the weights and modify the weights: ")
            print("*********************************************")
            for index_, current_sample in enumerate(X):
                print("For the first sample: there are", self.number_outputs, "outputs")
                for index_w, weight in enumerate(self.weights):
                    predicted_y = self.predict_function(current_sample, index_)
                    update = float(self.learning_r * (y_[index_] - predicted_y))
                    print("For weight", weight, ", the update to be made is", update, " as t - y =", y_[index_] - predicted_y)
                    # making changes to the weights:
                    print(" ")
                    print("     The changes are made on each of the weights as follows:")
                    for index, value in enumerate(current_sample):
                        if(value == 0):
                            print("The x at this point is 0 and thus didn't contribute to any weight change: Pass")
                        else:
                            prev = weight[index]
                            weight[index] += (value * update)
                            print("This weight at" , index, index_w, "has changed from", prev, "to", weight[index])

    
# Create example:
my_perceptron = Perceptron(4, 0.1, 3)
#my_perceptron.weights = [[1,1,1]]
#a = my_perceptron.predict_function([1,1,1])
X = np.array([[1,1,1]])
my_perceptron.train(X,[1, 0, 1, 1])