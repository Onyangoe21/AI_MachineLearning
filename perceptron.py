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
        return np.where(x >= 0, 1, 0)

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
    def train(self, X, y):
        # X is a 2d array size m*n where m=num of rows(samples), n=cols(features)
        samples, features = X.shape  # initializing number of samples and number of features

        # initialize the weights
        print("Initializing the weights and will print them as they get initialized")
        for i in range(self.number_outputs):
            self.weights.append(np.zeros(features))
            # clean as matrix for printing
            y_print = np.array([1 if i > 0 else 0 for i in self.weights[i]])
            print(y_print, " ----> weight", i+1)
        self.bias = 0



        y_ = np.array([1 if i > 0 else 0 for i in y]) # make sure nothing passes that is not 1's and 0's as training values

        # # start training
        # for epoch in range(self.epochs):
        #     for index_, current_sample in enumerate(X):
        #         linear_output = np.dot(current_sample, self.weights) + self.bias
        #         predicted_y = self.activation_func

        #         update = self.learning_r * (y_[index_] - predicted_y)
        #         self.weights += update * current_sample

        #         self.bias += update
    
    # define fit method with x and y
    def perfomance_test(self, X, y):
        # X is a 2d array size m*n where m=num of rows(samples), n=cols(features)
        samples, features = X.shape  # initializing number of samples and number of features

        # initialize the weights
        self.weights = np.zeros(features)
        self.bias = 0


        y_ = np.array([1 if i > 0 else 0 for i in y])

        # start training
        for epoch in range(self.epochs):
            for index_, current_sample in enumerate(X):
                linear_output = np.dot(current_sample, self.weights) + self.bias
                predicted_y = self.activation_func

                update = self.learning_r * (y_[index_] - predicted_y)
                self.weights += update * current_sample

                self.bias += update

    

# Create example:
my_perceptron = Perceptron(4, 0.1, 100)
my_perceptron.weights = [[1,1,1]]
a = my_perceptron.predict_function([1,1,1])
X = np.array([[1,1,1]])
my_perceptron.train(X,[1])
print("Done running", a)