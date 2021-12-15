from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


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

    def fitAndScore(self, X_train, X_test, y_train):
        """ 
        @author: Achraf
        """
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)
    
    def assertEqual(self, param, param1):
        return param == param1



def bestParameters(X_train, X_test, y_train, y_test, layers_sizes,activations,learning_rate):
        """ 
        @author: Achraf
        """
        neuralModels=[]
        neuralMetric=[]
        for layerSize in layers_sizes:
            for activation in activations:
                for lr in learning_rate:
                    neuralModel=NeuralNetwork(solver="lbfgs", layers_sizes=layerSize,
                                   activation=activation, learning_rate=lr)
                    neuralModels.append((layerSize, activation, lr))
                    y_pred=neuralModel.fitAndScore(X_train, X_test, y_train)
                    neuralMetric.append(accuracy_score(y_test,y_pred))
        max_value = max(neuralMetric)
        max_index = neuralMetric.index(max_value)
        
        return f"The neural network with the best accuracy has the parameters: layer size: {neuralModels[max_index][0]} activation function: {neuralModels[max_index][1]}, learning rate {neuralModels[max_index][2]}, with an accuracy score of : {max_value}"