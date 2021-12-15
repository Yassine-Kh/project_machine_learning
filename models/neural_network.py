from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class NeuralNetwork:
    def __init__(self, solver="lbfgs", layers_sizes=(5, 2), activation="logistic", learning_rate=0.001):
        """ 
        @author: Achraf
        
        params: solver          : The solver for weight optimization. always lbfgs in for classification
        params: layers_sizes    : A tuple containing the layers sizes
        params: activation      : Activation function for the hidden layer
        params: learning_rate   : The initial learning rate used
        
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
        params: X_train :pandas dataframe
        params: X_test  :pandas dataframe
        params: y_train : numpy array
        
        
        output: prediction done on X_test
        """
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)
    
    def assertEqual(self, param, param1):
        return param == param1



def bestParameters(X_train, X_test, y_train, y_test, layers_sizes,activations,learning_rate):
        """ 
        @author: Achraf
        params: layers_sizes    : Array of a tuple containing the layers sizes
        params: activation      : Array of activation function for the hidden layer
        params: learning_rate   : Array of the initial learning rate used
        
        
        output: The parameters of the model with the best accuracy
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
        
        return f"The neural network with the best accuracy has the parameters: layer size: {neuralModels[max_index][0]} " \
               f"activation function: {neuralModels[max_index][1]}, learning rate {neuralModels[max_index][2]}, " \
               f"with an accuracy score of : {max_value}"