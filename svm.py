from utils import *


class SVM():
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
            return "this is the weights of the model" ,self.model.dual_coef_
        except :
            return "please train your model first"
    
    def fitAndPredict(self, X_train, X_test, y_train, y_test):
        """ 
        @author: Achraf
        """
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)


data1 = CleanData("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
column_names = ["variance", "skewness", "curtosis", "entropy", "classification"]
X_train, X_test, y_train, y_test = data1.splitData(1/3, 42, column_names)


modelSVM=SVM( kernel='rbf',  gamma='scale', tol=0.001)

# kernel=["linear", "poly", "rbf", "sigmoid"] 
# gamma=["scale", "auto"]
# tol=[range(0.001,0.1,10)]

print(modelSVM.fitAndPredict(X_train, X_test, y_train, y_test))
print(modelSVM.__str__())

