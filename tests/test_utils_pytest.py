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

