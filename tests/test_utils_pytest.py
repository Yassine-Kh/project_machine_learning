import pytest
import pandas as pd

from utils import CleanData


@pytest.fixture
def clean_data_class_banknote():
    file_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    clean_data_obj = CleanData(file_path)
    real_df = pd.read_csv(file_path)
    return [clean_data_obj, real_df](request.param)



@pytest.fixture
def clean_data_class_kidney():
    file_path = "..\kidney_disease.csv"
    clean_data_obj = CleanData(file_path)
    real_df = pd.read_csv(file_path)
    return [clean_data_obj, real_df]


@pytest.mark.parametrize("clean_data_obj", "real_df", [
    (clean_data_class_kidney[0], clean_data_class_kidney[1]),
    (clean_data_class_banknote[0], clean_data_class_banknote[1]),
    indirect=True])

def test_get_columns(clean_data_obj, real_df):
    assert (clean_data_obj.getColumns()[k] == real_df.columns()[k] for k in range(len(real_df.columns)))


def test_set_columns():
    assert False


def test_check_data():
    assert False


def test_clean_data():
    assert False


def test_scale_data():
    assert False


def test_split_data():
    assert False
