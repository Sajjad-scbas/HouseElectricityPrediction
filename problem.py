import os
from glob import glob
import pandas as pd
from sklearn.model_selection import ShuffleSplit

import rampwf as rw

problem_title = "House Electricity Prediction"
_target_column_name = 'Appliances'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.Estimator()


score_types = [
    rw.score_types.RMSE(name = 'rmse', precision=4)
]


def get_file_list_from_dir(*, path, filename):
    data_files = sorted(glob(os.path.join(path, "data/public", filename)))
    return data_files


def _get_data(path, f_name):
    #data_files = get_file_list_from_dir(path=path, filename=f_name)

    dataset = pd.read_csv("Data/" +f_name , index_col=0)

    #X = dataset.drop([_target_column_name] + [_ignore_column_names], axis=1)
    #y = dataset[_target_column_name].values

    X = dataset.loc[:, dataset.columns != _target_column_name]
    y = dataset.loc[:, dataset.columns == _target_column_name].to_numpy().flatten()
    print(y.shape)

    return X, y


def get_train_data(path="."):
    f_name = "train.csv"
    return _get_data(path, f_name)


def get_test_data(path="."):
    f_name = "train.csv"
    return _get_data(path, f_name)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)