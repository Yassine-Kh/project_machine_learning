import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotData(X_test, y_test, y_pred, model):
    """
    @author: Ala Eddine
    """
    x1 = X_test['variance']
    x2 = X_test['entropy']
    X1, X2 = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
    y_pred = y_pred[0:441]
    y_test1 = pd.Series.to_numpy(y_test)[0:441]
    plt.subplot(2,2,1)
    plt.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
    plt.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
    plt.contourf(X1, X2, y_test1.reshape(21, 21), cmap='bwr', alpha=0.5)
    plt.scatter(x1, x2, c=y_test, cmap='bwr')
    plt.legend(['y = 0', 'y = 1'], prop={'size': 10})
    plt.xlabel("$x_1$ : variance", fontsize=10)
    plt.ylabel("$x_2$ : entropy", fontsize=10)
    plt.title('Binary classification : True model', size=10)
    plt.subplot(2, 2, 2)
    plt.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
    plt.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
    plt.contourf(X1, X2, y_pred.reshape(21, 21), cmap='bwr', alpha=0.5)
    plt.scatter(x1, x2, c=y_test, cmap='bwr')
    plt.legend(['y = 0', 'y = 1'], prop={'size': 10})
    plt.xlabel("$x_1$ : variance", fontsize=10)
    plt.ylabel("$x_2$ : entropy", fontsize=10)
    plt.title('Binary classification : ' + model, size=10)
    plt.tight_layout()