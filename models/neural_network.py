from sklearn.neural_network import MLPClassifier


class NeuralNetwork:
    def __init__(self, solver="lbfgs", layers_sizes=(5, 2), activation="logistic", learning_rate=0.001):
        """ 
        @author: Achraf
        """
        self.model = MLPClassifier(solver=solver, hidden_layer_sizes=layers_sizes, random_state=1,
                                   activation=activation, learning_rate_init=learning_rate)

    def __str__(self):
        """ 
        @author: Achraf
        """
        try:
            return f"The weights of the model {self.model.coefs_}"
        except:
            return "Please train your model first"

    def fitAndScore(self, X_train, X_test, y_train, y_test):
        """ 
        @author: Achraf
        """
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)

    def assertEqual(self, param, param1):
        return param == param1
