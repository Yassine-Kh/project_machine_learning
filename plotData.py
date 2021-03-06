import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def plotData(X_test, y_test, y_pred, feature1, feature2, model):
    """
     Plots binary classification of the true model and the predicted model with 2 chosen
    features among the list of features

    Parameters
    ----------
    X_test : data features for test.
    y_test : ground truth classes.
    y_pred : predicted classes.
    feature1 : first feature chosen among the list of all features.
    feature2 : second feature chosen among the list of all features.
    model : classifier model(DecisionTreeClassifier, RandomForestClassifier and AdaBoostClassifier).

    Returns
    -------
    """

    N = X_test.shape[0]
    i = 0
    while i < N and not ((np.sqrt(N - i) - math.floor(np.sqrt(N - i))) == 0):
        i += 1
    number = N - i

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 15))
    x1 = X_test[feature1]
    x2 = X_test[feature2]
    X1, X2 = np.meshgrid(np.linspace(0, 1, int(np.sqrt(number))), np.linspace(0, 1, int(np.sqrt(number))))
    y_pred = y_pred[0:number]
    y_test1 = pd.Series.to_numpy(y_test)[0:number]
    ax1.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
    ax1.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
    ax1.contourf(X1, X2, y_test1.reshape(int(np.sqrt(number)), int(np.sqrt(number))), cmap='bwr', alpha=0.5)
    ax1.scatter(x1, x2, c=y_test, cmap='bwr')
    ax1.legend(['y = 0', 'y = 1'], prop={'size': 40})
    ax1.set_xlabel("$x_1$ : " + feature1, fontsize=40)
    ax1.set_ylabel("$x_2$ : " + feature2, fontsize=40)
    ax1.set_title('Binary classification : True model', size=40)
    ax2.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
    ax2.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
    ax2.contourf(X1, X2, y_pred.reshape(int(np.sqrt(number)), int(np.sqrt(number))), cmap='bwr', alpha=0.5)
    ax2.scatter(x1, x2, c=y_test, cmap='bwr')
    ax2.legend(['y = 0', 'y = 1'], prop={'size': 40})
    ax2.set_xlabel("$x_1$ : " + feature1, fontsize=40)
    ax2.set_ylabel("$x_2$ : " + feature2, fontsize=40)
    ax2.set_title('Binary classification : ' + model, size=40)
    plt.tight_layout()