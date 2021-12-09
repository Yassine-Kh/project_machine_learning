"""
Created on Wed Dec  1 10:09:33 2021

"""
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from DecisionTree import DecisionTree
from plotData import plotData


class CleanData():

    def __init__(self, file_path):
        """
        @author: Yassine & Achraf
        """
        self.df = pd.read_csv(file_path)

    def __str__(self):
        """
        @author: Yassine
        """
        return "This dataframe looks like this: \n\n{}".format(self.df.head())

    def describeData(self):
        """
        @author: Ala Eddine
        """
        # show summary statistics
        print(self.df.describe())
        # plot histograms
        self.df.hist()
        plt.show()

    def getColumns(self):
        """
        @author: Yassine
        """
        return self.df.columns

    def setColumns(self, column_names):
        self.df.columns = column_names

    def checkData(self):
        """
        @author: Yassine
        Check if the data contains any null values
        """
        counter=0
        for col_name in self.getColumns:
            if self.df[col_name].isnull().values.any():
                counter+=1
        return counter

    def cleanData(self, column_names):
        """
        @author: Yassine
        """
        if any(element.isdigit() for element in self.getColumns()):
            self.setColumns(column_names)
        for col_name in self.getColumns():
                if self.df[col_name].isnull().values.any():
                    self.df[col_name] = self.df[col_name].fillna(method='bfill')
        df_to_treat = (self.df.drop("id", axis=1)) if ("id" in list(self.getColumns())) else self.df
        df_to_treat= (self.df.drop("class", axis=1)) if ("class" in list(self.getColumns())) else self.df
        return df_to_treat

    def normalizeData(self, dataframe,
                      method="minmax"):  # We don't want to change directly the attribute df because we need it after
        min_max_scaler = MinMaxScaler()
        for col_name in dataframe.columns:
            if is_numeric_dtype(self.df[col_name]):
                dataframe[col_name] = min_max_scaler.fit_transform(
                    dataframe[col_name].values.astype(float).reshape(-1, 1))
        return dataframe

    def arrangedData(self, test_size, random_state, *column_names):
        self.new_df = self.normalizeData(self.cleanData(column_names[0]))
        X, y = self.new_df, self.df["class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test




data1 = CleanData("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
data1.describeData()

column_names = ["variance", "skewness", "curtosis", "entropy", "class"]
X_train, X_test, y_train, y_test = data1.arrangedData(1 / 3, 42, column_names)

classif_tree = DecisionTree()
tab_log_loss_tree, tab_log_loss_tree_box, optimal_depth = classif_tree.classification(X_train, y_train)
print("optimal depth = ", optimal_depth)

classif_tree.plot(X_train, X_test, y_train, optimal_depth, column_names)
y_tree, y_forest, y_ada = classif_tree.adjust_classification(X_train, X_test, y_train, optimal_depth, column_names)

plotData(X_test, y_test, y_tree,"Decision Tree")
plotData(X_test, y_test, y_forest,"Random Forest")
plotData(X_test, y_test, y_ada,"AdaBoost")

classif_tree.calculate_metrics(y_test, y_tree, "DecisionTree")
classif_tree.calculate_metrics(y_test, y_forest, "RandomForest")
classif_tree.calculate_metrics(y_test, y_ada, "AdaBoost")
plt.show()
