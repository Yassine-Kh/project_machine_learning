from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SVM:
    def __init__(self, kernel='rbf',  gamma='scale', tol=0.001):
        """ 
        @author: Achraf
        params: kernel    : Specifies the kernel type to be used in the algorithm.
        params: gamma     : scale or auto
        params: tol       : Tolerance for stopping criterion
        
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

# kernel=["linear", "poly", "rbf", "sigmoid"] 
# gamma=["scale", "auto"]
# tol=[range(0.001,0.1,10)]



def bestParameters(X_train, X_test, y_train, y_test, kernels,gammas,tols):
        """ 
        @author: Achraf
        params: kernels    : list of string containing the used kernels
        params: gammas     : list of string containing the used gammas
        params: tols       : list of tolerance for stopping criterion
        
        output: The parameters of the model with the best accuracy
        """
        svmModels=[]
        svmMetric=[]
        for kernel in kernels:
            for gamma in gammas:
                for tol in tols:
                    svmModel=SVM(kernel=kernel,  gamma=gamma, tol=tol)
                    svmModels.append((kernel, gamma, tol))
                    y_pred=svmModel.fitAndPredict(X_train, X_test, y_train)
                    svmMetric.append(accuracy_score(y_test,y_pred))
        max_value = max(svmMetric)
        max_index = svmMetric.index(max_value)
        bestpred=SVM(kernel=svmModels[max_index][0],  gamma=svmModels[max_index][1], tol=svmModels[max_index][2]).fitAndPredict(X_train, X_test, y_train)
        print("Classification report :\n", classification_report(y_test, bestpred))
        return f"The SVM model with the best accuracy has the parameters: kernel: {svmModels[max_index][0]}, gamma: {svmModels[max_index][1]}, tol {svmModels[max_index][2]}, with an accuracy score of : {max_value}"