"""
Created on Wed Dec  1 10:09:33 2021

"""
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class CleanData():
    
    def __init__(self, file_url):
        """ 
        @author: Yassine
        """
        self.df = pd.read_csv(file_url)
        
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
        counter=0
        for col_name in self.getColumns:
            if  self.df[col_name].isnull().values.any():
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
                    self.df[col_name] = self.df[col_name].fillna(method= 'bfill')
        df_to_treat = (self.df.drop("id", axis=1)) if ("id" in list(self.getColumns())) else self.df
        df_to_treat= (self.df.drop("classification", axis=1)) if ("classification" in list(self.getColumns())) else self.df
        return df_to_treat

        
data1 = CleanData("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
print(data1)
column_names = ["variance", "skewness", "curtosis", "entropy", "classification"]
