from utils import *

class NeuralNetwork():
    def __init__(self, solver="lbfgs" , layers_sizes=(5,2), activation="logistic", learning_rate=0.001):
        """ 
        @author: Achraf
        """
        self.model = MLPClassifier(solver=solver, hidden_layer_sizes=layers_sizes, random_state=1, activation=activation, learning_rate_init=learning_rate)

    def __str__(self):
        """ 
        @author: Achraf
        """
        try :
            return "this is the weights of the model" ,self.model.coefs_
        except :
            return "please train your model first"
    
    def fitAndScore(self, X_train, X_test, y_train, y_test):
        """ 
        @author: Achraf
        """
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)


data1 = CleanData("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
column_names = ["variance", "skewness", "curtosis", "entropy", "classification"]
X_train, X_test, y_train, y_test = data1.splitData(1/3, 42, column_names)


modelNeural=NeuralNetwork(solver="lbfgs" , layers_sizes=(5,2), activation="logistic", learning_rate=0.001)

# solver=["lbfgs", "adam", "sgd"] even though lbgs is used in classification
# layers_sizes= choose different number of layers and different number of neurons per layer (last layer has to be 2)
# activation=["logistic", "tanh"]
# learning_rate=[range(0.001,0.1,10)]

modelNeural.fitAndScore(X_train, X_test, y_train, y_test)