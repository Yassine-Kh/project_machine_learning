"""
Created on Wed Dec  1 10:09:33 2021

"""
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA


class CleanData:

    def __init__(self, file_path):
        """
        @author: Yassine & Achraf
        
        
        Args: 
        :param file_path        : path for the dataset.
            
        Returns: 
        NONE
        
        import the csv into a pandas df
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
        Args: 
        :param dataframe        : pandas dataframe.
            
        Returns: 
        numeric_columns, string_columns
        
        Returns the names of numeric and string columns
        
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
        Args: 
        :param dataframe        : pandas dataframe.
        :param string_columns   : string array
        :param numeric_columns  : string array

        Returns: 
        data_tr_table
        
        Returns a dataframe after converting the string values into ints
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
        Args: 
        :param dataframe        : pandas dataframe.

        Returns: 
        reduced
        
        Returns a dataframe after applying PCA with a cut off variance of 0.95
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
        Args: 
        :param dataframe        : pandas dataframe.

        Returns: 
        dataframe
        
        Returns a dataframe after scaling the data with a MinMax scaler
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
        Args: 
        :param test_size        : fraction.
        :param random_state     : int.
        :param columns_names    : string .
        
        

        Returns: 
        X_train, X_test, y_train, y_test
        
        Returns the data split into training and testing after the applying the different cleaning functions
        """
        tempdf=self.cleanData(column_names[0])
        numericColumns, stringColumns = self.defineType(tempdf)
        self.new_df = CleanData.reduceDimension(self.scaleData(CleanData.encodestring(tempdf, stringColumns, numericColumns)))
        X, y = self.new_df, self.df["classification"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
