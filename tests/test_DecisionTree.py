import numpy
import pytest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from models.DecisionTree import DecisionTree
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
    assert optimal_depth >= 2
    

@pytest.mark.parametrize("data", ["clean_data_class_banknote", "clean_data_class_kidney"])
def test_export_pdf(data, request):
    data = request.getfixturevalue(data)
    clean_data, real_df = data[0], data[1]
    X_train, X_test, y_train, y_test = clean_data
    classification_tree = DecisionTreeClassifier(max_depth=3)
    classification_tree.fit(X_train,y_train)
    decision_tree_model = DecisionTree()
    column_names = list(X_train.columns)
    graph = decision_tree_model.export_png(column_names, classification_tree)
    assert graph is not None


@pytest.mark.parametrize("data", ["clean_data_class_banknote", "clean_data_class_kidney"])
def test_adjust_classification(data, request):
    data = request.getfixturevalue(data)
    clean_data, real_df = data[0], data[1]
    X_train, X_test, y_train, y_test = clean_data
    decision_tree_model = DecisionTree()
    y_tree, y_forest, y_ada = decision_tree_model.adjust_classification(X_train, X_test, y_train, optimal_depth=3)
    assert isinstance(y_tree, numpy.ndarray) and isinstance(y_forest, numpy.ndarray) and isinstance(y_ada, numpy.ndarray)


def test_calculate_metrics():
    pass


def test_plot():
    pass
