import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.



class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True, deterministic=-1):
        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.alpha = momentum
        self.shuffle = shuffle
        self.num_epoch = deterministic
        self.train_val_split = 0.25
        self.num_windows_after_bssf = 20

    def fit(self, X, y, initial_weights1=None, initial_weights2=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        X, X_val, y, y_val = train_test_split(X, y, test_size=self.train_val_split)
        instances, features = y.shape
        self.num_output_node = features
        self.num_hidden_node = self.hidden_layer_widths[0]
        self.row, self.col = X.shape
        if initial_weights1 is None:
            self.weights1 = self.initialize_weights(self.col + 1, self.num_hidden_node)
        else:
            self.weights1 = initial_weights1

        if initial_weights2 is None:
            self.weights2 = self.initialize_weights(self.num_hidden_node + 1, self.num_output_node)
        else:
            self.weights2 = initial_weights2

        hidden_nodes = np.zeros((self.row, self.num_hidden_node))
        self.output_nodes = np.zeros((self.row, self.num_output_node))
        aug = np.ones((self.row, 1))
        X = np.concatenate((X,aug),axis=1)
        self.hidden_nodes = np.concatenate((hidden_nodes, aug), axis=1)
        self.training_MSE_history = []
        self.VS_MSE_history = []
        self.classification_accuracy_history = []
        self.prev_dw1 = np.zeros((self.col + 1, self.num_hidden_node))
        self.prev_dw2 = np.zeros((self.num_hidden_node + 1, self.num_output_node))

        if self.num_epoch == -1:
            self.epoch_counter = 0
            bssf = -1
            windows = self.num_windows_after_bssf
            while True:
                if self.shuffle is True:
                    X, y = self._shuffle_data(X, y)
                self.iterate(X, y)
                self.epoch_counter += 1
                accuracy = 1 - self.getMSE(X_val, y_val)
                self.VS_MSE_history.append(1 - accuracy)
                X_noBias = np.delete(X, np.s_[-1:], axis=1)
                self.training_MSE_history.append(self.getMSE(X_noBias, y))
                self.classification_accuracy_history.append(self.score(X_val, y_val))
                if accuracy > bssf:
                    bssf = accuracy
                    print(bssf)             # print out bssf
                    windows = self.num_windows_after_bssf
                    self.bssf_weights1 = np.copy(self.weights1)
                    self.bssf_weights2 = np.copy(self.weights2)
                else:
                    windows = windows - 1
                if windows <= 0:
                    self.weights1 = np.copy(self.bssf_weights1)
                    self.weights2 = np.copy(self.bssf_weights2)
                    break
        else:
            for i in range(self.num_epoch):
                if self.shuffle is True:
                    X, y = self._shuffle_data(X, y)
                self.iterate(X, y)

        # outfile = np.concatenate((self.weights2.transpose().reshape(-1,1), self.weights1.transpose().reshape(-1,1)),axis=0)
        # np.savetxt("evaluation.csv", outfile, delimiter =',')
        print("max epoch for best VS")
        index = self.VS_MSE_history.index(min(self.VS_MSE_history))
        print(index)
        print("MSE VS" )
        print(1- bssf)
        X_noBias = np.delete(X, np.s_[-1:], axis=1)
        bssf_training = self.getMSE(X_noBias, y)
        print("MSE training")
        print(bssf_training)
        return self

    def iterate(self, X, y):
        # this will do one epoch with online method
        for i in range(self.row):
            # forward pass
            net = np.dot(X[i,:],self.weights1)
            for j in range(self.num_hidden_node):
                self.hidden_nodes[i,j] = self.sigmoid(net[j])
            net = np.dot(self.hidden_nodes[i,:],self.weights2)
            for j in range(self.num_output_node):
                self.output_nodes[i,j] = self.sigmoid(net[j])
            # backward pass
                # output nodes
            delta2 = np.zeros((1,self.num_output_node))
            dw2 = np.zeros((self.num_hidden_node + 1, self.num_output_node))
            for j in range(self.num_output_node):
                delta2[0,j] = (y[i,j] - self.output_nodes[i,j])*self.output_nodes[i,j]*(1 - self.output_nodes[i,j])
                for k in range(self.num_hidden_node + 1):
                    dw2[k,j] = self.lr * self.hidden_nodes[i, k]*delta2[0,j]
                # hidden nodes
            delta1 = np.zeros((1, self.num_hidden_node))
            dw1 = np.zeros((self.col + 1, self.num_hidden_node))
            for j in range(self.num_hidden_node):
                delta1[0,j] = np.sum(delta2*self.weights2[j,:].transpose())*self.hidden_nodes[i,j]*(1-self.hidden_nodes[i,j])
                for k in range(self.col + 1):
                    dw1[k,j] = self.lr * X[i,k]*delta1[0,j]
            dw2 += self.alpha*self.prev_dw2
            dw1 += self.alpha*self.prev_dw1
            self.weights2 = self.weights2 + dw2
            self.weights1 = self.weights1 + dw1
            # print(i)
            # print(X[i,:])
            # print(self.weights1)
            # print(self.weights2)
            self.prev_dw1 = np.copy(dw1)
            self.prev_dw2 = np.copy(dw2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1*x))

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        row, col = X.shape
        hidden_nodes = np.zeros((row, self.num_hidden_node))
        aug = np.ones((row, 1))
        X = np.concatenate((X, aug), axis=1)
        hidden_nodes = np.concatenate((hidden_nodes, aug), axis=1)
        output_nodes = np.zeros((row, self.num_output_node))
        for i in range(row):
            net = np.dot(X[i,:],self.weights1)
            for j in range(self.num_hidden_node):
                hidden_nodes[i,j] = self.sigmoid(net[j])
            net = np.dot(hidden_nodes[i,:],self.weights2)
            for j in range(self.num_output_node):
                output_nodes[i,j] = self.sigmoid(net[j])

        return output_nodes, output_nodes.shape

    def initialize_weights(self, num_input_node, num_output_node):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        weight = (np.random.rand(num_input_node, num_output_node) - 0.5)*0.1
        # weight = np.zeros((num_input_node , num_output_node ))
        return weight

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        output, shape = self.predict(X)
        row, col = output.shape

        correct = 0
        for i in range(row):
            index = np.argmax(output[i,:])
            target = np.argmax(y[i,:])
            if index == target:
                correct += 1
        return correct / row

    def getMSE(self, X, y):
        output, shape = self.predict(X)
        row, col = output.shape
        error = output - y
        SSE = np.sum(np.square(error))
        MSE = SSE / row
        return MSE

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        rand = np.random.permutation(self.row)
        X = X[rand]
        y = y[rand]
        return X, y

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights1, self.weights2

