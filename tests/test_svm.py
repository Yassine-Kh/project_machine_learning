import pytest
import pandas as pd

from models.svm import SVM
from utils import CleanData

"""
@author: Yassine Khalsi
"""


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
def test_class_neural_network(data, request):
    model_svm = SVM(kernel='rbf', gamma='scale', tol=0.001)
    assert model_svm.assertEqual(str(model_svm), 'Please train your model first')


def test_assertEqual():
    param1 = "test"
    param2 = "test"
    param3 = "test3"
    assert param1 == param2 and param3 != param1


@pytest.mark.parametrize("data", ["clean_data_class_banknote", "clean_data_class_kidney"])
def test_fitAndPredict(data, request):
    data = request.getfixturevalue(data)
    data_to_clean, real_df = data[0], data[1]
    column_names = ["variance", "skewness", "curtosis", "entropy", "classification"]
    X_train, X_test, y_train, y_test = data_to_clean.splitData(1 / 3, 42, column_names)
    model_svm = SVM(kernel='rbf', gamma='scale', tol=0.001)
    assert model_svm.assertEqual(str(model_svm), 'Please train your model first')
    model_svm.fitAndPredict(X_train, X_test, y_train)
    assert not model_svm.assertEqual(str(model_svm), 'Please train your model first')
