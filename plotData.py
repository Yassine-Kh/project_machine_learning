import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def plotData(X_test, y_test, y_pred, model):
    """
    @author: Ala Eddine
    """
    N = X_test.shape[0]
    i = 0
    while i < N and not ((np.sqrt(N - i) - math.floor(np.sqrt(N - i))) == 0):
        i += 1
    number = N - i

    for column in X_test.columns[1:]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 15))
        x1 = X_test[X_test.columns[0]]
        x2 = X_test[column]
        X1, X2 = np.meshgrid(np.linspace(0, 1, int(np.sqrt(number))), np.linspace(0, 1, int(np.sqrt(number))))
        y_pred = y_pred[0:number]
        y_test1 = pd.Series.to_numpy(y_test)[0:number]
        ax1.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
        ax1.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
        ax1.contourf(X1, X2, y_test1.reshape(int(np.sqrt(number)), int(np.sqrt(number))), cmap='bwr', alpha=0.5)
        ax1.scatter(x1, x2, c=y_test, cmap='bwr')
        ax1.legend(['y = 0', 'y = 1'], prop={'size': 40})
        ax1.set_xlabel("$x_1$ : " + X_test.columns[0], fontsize=40)
        ax1.set_ylabel("$x_2$ : " + column, fontsize=40)
        ax1.set_title('Binary classification : True model', size=40)
        ax2.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
        ax2.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
        ax2.contourf(X1, X2, y_pred.reshape(int(np.sqrt(number)), int(np.sqrt(number))), cmap='bwr', alpha=0.5)
        ax2.scatter(x1, x2, c=y_test, cmap='bwr')
        ax2.legend(['y = 0', 'y = 1'], prop={'size': 40})
        ax2.set_xlabel("$x_1$ : " + X_test.columns[0], fontsize=40)
        ax2.set_ylabel("$x_2$ : " + column, fontsize=40)
        ax2.set_title('Binary classification : ' + model, size=40)
        plt.tight_layout()