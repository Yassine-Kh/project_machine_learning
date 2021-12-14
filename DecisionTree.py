import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from graphviz import Source


class DecisionTree:
    """
    @author: Ala Eddine
    """
    def __init__(self, n_depths=10):
        self.cvp = ShuffleSplit(1000, train_size=2 / 3)
        self.n_depths = n_depths
        self.depths = np.linspace(1, 10, self.n_depths)

    def classification(self,  X_train, y_train):
        tab_log_loss_tree = np.zeros(self.n_depths)
        tab_log_loss_tree_box = []
        index = 0
        for i in range(self.n_depths):
            reg_tree = DecisionTreeClassifier(max_depth=self.depths[i])
            log_loss = np.sqrt(-cross_val_score(reg_tree, X_train, y_train, scoring='neg_log_loss', cv=self.cvp))
            tab_log_loss_tree_box.append(log_loss)
            tab_log_loss_tree[i] = np.median(log_loss)
            if tab_log_loss_tree[i] < tab_log_loss_tree[index]:
                index = i
        optimal_depth = index+1
        return tab_log_loss_tree, tab_log_loss_tree_box, optimal_depth

    def export_pdf(self, column_names, classif_model):
        # Export the tree to "plot_tree.pdf"
        plot_tree = export_graphviz(classif_model, out_file=None, feature_names=column_names, filled=True)
        graph = Source(plot_tree)
        graph.render(str(classif_model))
        # Plot the tree
        return graph

    def adjust_classification(self, X_train, X_test, y_train,optimal_depth, column_names):
        # Adjust classification tree with optimal depth
        classif_tree = DecisionTreeClassifier(max_depth=optimal_depth)
        classif_tree.fit(X_train, y_train)
        y_tree = classif_tree.predict(X_test)
        graph = self.export_pdf(column_names, classif_tree)
        classif_forest = RandomForestClassifier(max_depth=optimal_depth)
        classif_forest.fit(X_train, y_train)
        y_forest = classif_forest.predict(X_test)
        classif_ada = AdaBoostClassifier()
        classif_ada.fit(X_train, y_train)
        y_ada = classif_ada.predict(X_test)
        return y_tree, y_forest, y_ada, graph

    def calculate_metrics(self, y_test, y_pred, model):
        accuracy = accuracy_score(y_test, y_pred)
        print(model," Accuracy : ",accuracy)
        print("Classification report :\n", classification_report(y_test, y_pred))



    def plot(self, X_train, X_test, y_train, optimal_depth, column_names):
        plt.plot(self.depths, self.classification(X_train, y_train)[0])
        plt.boxplot(self.classification(X_train, y_train)[1])
        plt.xlabel('Max depth of the tree', size=20)
        plt.ylabel('CROSS-Validation', size=20)
        plt.show()