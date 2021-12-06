#@author : Ala Eddine NOUALI

import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


class DecisionTree:

    def __init__(self, n_depths=10):
        self.cvp = ShuffleSplit(1000, train_size=2 / 3)
        self.n_depths = n_depths
        self.depths = np.linspace(1, 10, self.n_depths)

    def classification(self,  X_train, y_train):
        tab_log_loss_tree = np.zeros(self.n_depths)
        tab_log_loss_tree_box = []
        for i in range(self.n_depths):
            reg_tree = DecisionTreeClassifier(max_depth=self.depths[i])
            log_loss = np.sqrt(-cross_val_score(reg_tree, X_train, y_train, scoring='neg_log_loss', cv=self.cvp))
            tab_log_loss_tree_box.append(log_loss)
            tab_log_loss_tree[i] = np.median(log_loss)
        return tab_log_loss_tree, tab_log_loss_tree_box

    def plot(self, X_train, y_train):
        plt.plot(self.depths, self.classification(X_train, y_train)[0])
        plt.boxplot(self.classification(X_train, y_train)[1])
        plt.xlabel('Max depth of the tree', size=20)
        plt.ylabel('CROSS-Validation', size=20)
        plt.show()