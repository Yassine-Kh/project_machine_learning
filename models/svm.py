from sklearn import svm


class SVM:
    def __init__(self, kernel='rbf',  gamma='scale', tol=0.001):
        """ 
        @author: Achraf
        """
        self.model = svm.SVC(kernel='rbf',  gamma='scale', tol=0.001)


    def __str__(self):
        """ 
        @author: Achraf
        """
        try :
            return f"This is the weights of the model {self.model.dual_coef_}"
        except :
            return "Please train your model first"
    
    def fitAndPredict(self, X_train, X_test, y_train):
        """ 
        @author: Achraf
        """
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)

    def assertEqual(self, param, param1):
        """
        @author: Achraf
        """
        return param == param1
