import pytest

from .utils import CleanData
@pytest.fixture
def clean_data_class_banknote():
    return CleanData("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")


@pytest.fixture
def clean_data_class_kidney():
    file_path = "../../kidney_disease.csv"
    return CleanData(file_path)


def test(clean_data_class_kidney):
    print("test")
    print(clean_data_class_kidney)
    assert 1 == 1
