import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotData_banknote_authentication(X_test, y_test, y_pred, model):
    """
    @author: Ala Eddine
    """
    fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(20,15))
    # ________________________________________________________________________________________
    x1 = X_test['variance']
    x2 = X_test['skewness']
    X1, X2 = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
    y_pred = y_pred[0:441]
    y_test1 = pd.Series.to_numpy(y_test)[0:441]
    ax1.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
    ax1.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
    ax1.contourf(X1, X2, y_test1.reshape(21, 21), cmap='bwr', alpha=0.5)
    ax1.scatter(x1, x2, c=y_test, cmap='bwr')
    ax1.legend(['y = 0', 'y = 1'], prop={'size': 10})
    ax1.set_xlabel("$x_1$ : variance", fontsize=10)
    ax1.set_ylabel("$x_2$ : skewness", fontsize=10)
    ax1.set_title('Binary classification : True model', size=10)
    ax2.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
    ax2.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
    ax2.contourf(X1, X2, y_pred.reshape(21, 21), cmap='bwr', alpha=0.5)
    ax2.scatter(x1, x2, c=y_test, cmap='bwr')
    ax2.legend(['y = 0', 'y = 1'], prop={'size': 10})
    ax2.set_xlabel("$x_1$ : variance", fontsize=10)
    ax2.set_ylabel("$x_2$ : skewness", fontsize=10)
    ax2.set_title('Binary classification : ' + model, size=10)
    plt.tight_layout()
    #________________________________________________________________________________________
    x1 = X_test['variance']
    x2 = X_test['curtosis']
    ax3.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
    ax3.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
    ax3.contourf(X1, X2, y_test1.reshape(21, 21), cmap='bwr', alpha=0.5)
    ax3.scatter(x1, x2, c=y_test, cmap='bwr')
    ax3.legend(['y = 0', 'y = 1'], prop={'size': 10})
    ax3.set_xlabel("$x_1$ : variance", fontsize=10)
    ax3.set_ylabel("$x_2$ : curtosis", fontsize=10)
    ax3.set_title('Binary classification : True model', size=10)
    ax4.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
    ax4.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
    ax4.contourf(X1, X2, y_pred.reshape(21, 21), cmap='bwr', alpha=0.5)
    ax4.scatter(x1, x2, c=y_test, cmap='bwr')
    ax4.legend(['y = 0', 'y = 1'], prop={'size': 10})
    ax4.set_xlabel("$x_1$ : variance", fontsize=10)
    ax4.set_ylabel("$x_2$ : curtosis", fontsize=10)
    ax4.set_title('Binary classification : ' + model, size=10)
    plt.tight_layout()
    # ________________________________________________________________________________________
    x1 = X_test['variance']
    x2 = X_test['entropy']
    ax5.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
    ax5.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
    ax5.contourf(X1, X2, y_test1.reshape(21, 21), cmap='bwr', alpha=0.5)
    ax5.scatter(x1, x2, c=y_test, cmap='bwr')
    ax5.legend(['y = 0', 'y = 1'], prop={'size': 10})
    ax5.set_xlabel("$x_1$ : variance", fontsize=10)
    ax5.set_ylabel("$x_2$ : entropy", fontsize=10)
    ax5.set_title('Binary classification : True model', size=10)
    ax6.scatter(x1[y_test == 0], x2[y_test == 0], c='b')
    ax6.scatter(x1[y_test == 1], x2[y_test == 1], c='r')
    ax6.contourf(X1, X2, y_pred.reshape(21, 21), cmap='bwr', alpha=0.5)
    ax6.scatter(x1, x2, c=y_test, cmap='bwr')
    ax6.legend(['y = 0', 'y = 1'], prop={'size': 10})
    ax6.set_xlabel("$x_1$ : variance", fontsize=10)
    ax6.set_ylabel("$x_2$ : entropy", fontsize=10)
    ax6.set_title('Binary classification : ' + model, size=10)
    plt.tight_layout()
    # ________________________________________________________________________________________
