import pytest
import pandas as pd
from math import *
from utils import CleanData


@pytest.fixture
def clean_data_class_banknote():
    file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    data_to_clean = CleanData(file_path)
    real_df = pd.read_csv(file_path)
    data = [data_to_clean, real_df]
    return data


@pytest.fixture()
def clean_data_class_kidney():
    file_path = "..\kidney_disease.csv"
    data_to_clean = CleanData(file_path)
    real_df = pd.read_csv(file_path)
    return [data_to_clean, real_df]


@pytest.mark.parametrize("data", ["clean_data_class_banknote", "clean_data_class_kidney"])
def test_get_columns(data, request):
    data = request.getfixturevalue(data)
    data_to_clean, real_df = data[0], data[1]
    assert (data_to_clean.getColumns()[k] == real_df.columns[k] for k in range(len(real_df.columns)))


def test_set_columns(clean_data_class_banknote):
    data_to_clean, real_df = clean_data_class_banknote[0], clean_data_class_banknote[1]
    column_names = ["variance", "skewness", "curtosis", "entropy", "classification"]
    data_to_clean.setColumns(column_names)
    assert (data_to_clean.getColumns()[k] == column_names[k] for k in range(len(column_names)))


def test_check_data_banknote(clean_data_class_banknote):
    data_to_clean, real_df = clean_data_class_banknote[0], clean_data_class_banknote[1]
    assert data_to_clean.checkData() == 0


def test_check_data_kidney(clean_data_class_kidney):
    data_to_clean, real_df = clean_data_class_kidney[0], clean_data_class_kidney[1]
    assert data_to_clean.checkData() == 24


@pytest.mark.parametrize("data", ["clean_data_class_banknote", "clean_data_class_kidney"])
def test_clean_data(data, request):
    data = request.getfixturevalue(data)
    data_to_clean, real_df = data[0], data[1]
    column_names = ["variance", "skewness", "curtosis", "entropy", "classification"]
    clean_dataframe = data_to_clean.cleanData(column_names)
    assert data_to_clean.checkData() == 0
    assert "id" not in list(clean_dataframe.columns) and "classification" not in list(clean_dataframe.columns)


@pytest.mark.parametrize("data", ["clean_data_class_banknote", "clean_data_class_kidney"])
def test_scale_data(data, request):
    data = request.getfixturevalue(data)
    data_to_clean, real_df = data[0], data[1]
    clean_data = data_to_clean.scaleData(data_to_clean.df)
    assert (data_to_clean[column][index] <= 1 or data_to_clean[column][index] >= 0 for index in clean_data.index
            for column in clean_data.columns)


@pytest.mark.parametrize("data", ["clean_data_class_banknote", "clean_data_class_kidney"])
def test_split_data(data, request):
    data = request.getfixturevalue(data)
    data_to_clean, real_df = data[0], data[1]
    column_names = ["variance", "skewness", "curtosis", "entropy", "classification"]
    splited_data = data_to_clean.splitData(1 / 3, 42, column_names)
    assert (len(splited_data[0]) == floor(len(data_to_clean.df) * (2 / 3))
            and len(splited_data[1]) == ceil(len(data_to_clean.df) / 3)
            and len(splited_data[2]) == floor(len(data_to_clean.df) * (2 / 3))
            and len(splited_data[3]) == ceil(len(data_to_clean.df) / 3))
