"""
Created on Wed Dec  1 10:09:33 2021

"""
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA


class CleanData:

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
        counter = 0
        for col_name in self.getColumns():
            if self.df[col_name].isnull().values.any():
                counter += 1
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
        df_no_id = (self.df.drop("id", axis=1)) if ("id" in list(self.getColumns())) else self.df
        df_to_treat = df_no_id.drop("classification", axis=1) if (
                "classification" in list(self.getColumns())) else self.df
        return df_to_treat

    @staticmethod
    def defineType(dataframe):
        """
        @author: Achraf
        """
        numeric_columns = []
        string_columns = []
        for column in dataframe.columns:
            if dataframe[column].dtype in ['int32', 'int64', 'float32', 'float64', 'int', 'float']:
                numeric_columns.append(column)
            else:
                string_columns.append(column)
        return numeric_columns, string_columns

    @staticmethod
    def encodestring(dataframe, string_columns, numeric_columns):
        """
        @author: Achraf
        """
        encoder = OrdinalEncoder()
        if string_columns != []:
            stringData = dataframe[string_columns]
            encodedData = encoder.fit_transform(stringData)
            columns = string_columns
            data_tr_table = pd.DataFrame(encodedData, columns=columns)
        else:
            return dataframe
        data_tr_table = pd.concat([data_tr_table, dataframe[numeric_columns]], axis=1)
        return data_tr_table

    @staticmethod
    def reduceDimension(dataframe):
        """
        @author: Achraf
        """
        pca = PCA(n_components=0.95)
        pca.fit(dataframe)
        reduced = pca.transform(dataframe)
        reduced = pd.DataFrame(reduced[:, :len(reduced[0])])
        reduced[dataframe.columns[-1]] = dataframe.iloc[:, -1]
        return reduced

    def scaleData(self, dataframe):
        """ 
        @author: Achraf
        """
        min_max_scaler = MinMaxScaler()
        for col_name in dataframe.columns:
            if is_numeric_dtype(dataframe[col_name]):
                dataframe[col_name] = min_max_scaler.fit_transform(
                    dataframe[col_name].values.astype(float).reshape(-1, 1))
        return dataframe

    def splitData(self, test_size=1 / 3, random_state=42, *column_names):
        """ 
        @author: Achraf
        """
        tempdf=self.cleanData(column_names[0])
        numericColumns, stringColumns = self.defineType(tempdf)
        self.new_df = CleanData.reduceDimension(self.scaleData(CleanData.encodestring(tempdf, stringColumns, numericColumns)))
        X, y = self.new_df, self.df["classification"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test


data1 = CleanData("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
column_names = ["variance", "skewness", "curtosis", "entropy", "classification"]
essain = data1.scaleData(data1.df)
X_train, X_test, y_train, y_test = data1.splitData(1 / 3, 42, column_names)
print(len(X_test) == len(essain)/3)
