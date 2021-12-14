import pytest
import pandas as pd
from math import *

from DecisionTree import DecisionTree
from utils import CleanData


@pytest.fixture
def clean_data_class_banknote():
    file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    data_to_clean = CleanData(file_path)
    column_names = ["variance", "skewness", "curtosis", "entropy", "classification"]
    clean_data = data_to_clean.splitData(1 / 3, 42, column_names)
    real_df = pd.read_csv(file_path)
    data = [clean_data, real_df]
    return data


@pytest.fixture()
def clean_data_class_kidney():
    file_path = "..\kidney_disease.csv"
    data_to_clean = CleanData(file_path)
    column_names = ["variance", "skewness", "curtosis", "entropy", "classification"]
    clean_data = data_to_clean.splitData(1 / 3, 42, column_names)
    real_df = pd.read_csv(file_path)
    data = [clean_data, real_df]
    return data


@pytest.mark.parametrize("data", ["clean_data_class_banknote", "clean_data_class_kidney"])
def test_classification(data, request):
    data = request.getfixturevalue(data)
    clean_data, real_df = data[0], data[1]
    decision_tree_model = DecisionTree()
    tab_log_loss_tree, tab_log_loss_tree_box, optimal_depth = decision_tree_model.classification(clean_data[0],
                                                                                                 clean_data[2])
    assert optimal_depth == 3
    


def test_export_pdf():
    assert False


def test_adjust_classification():
    assert False


def test_calculate_metrics():
    assert False


def test_plot():
    assert False
