import numpy as np
import pandas as pd
from prefect import task, flow

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


from dataprocessing import data_processing
from utils import DROP_COLUMNS, TRAIN_DATASET_PATH, TARGET_COLUMN
from model import train_LR, train_Ridge, train_Lasso, train_DT, train_RF


# Load Data
# @task
def load_data(data_path, drop_columns, target_column):

    data = data_processing(data_path, drop_columns)
    X = data
    y = data.pop(target_column)
    return (X, y)


# @task
def split_data(data):

    X, y = data
    X = X.to_dict(orient="record")
    (train_x, test_x, train_y, test_y) = train_test_split(
        X, y, test_size=0.3, random_state=1993
    )
    return (train_x, test_x, train_y, test_y)


def encode_data(x_train, x_test):

    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    return (x_train, x_test)


# @flow(name="Training and Model Evaluation")
def main():
    # Load the processed dataset and split into train and test sets
    data = load_data(TRAIN_DATASET_PATH, DROP_COLUMNS, TARGET_COLUMN)
    (train_x, test_x, train_y, test_y) = split_data(data)
    (train_x, test_x) = encode_data(train_x, test_x)
    data = (train_x, train_y, test_x, test_y)

    # Train the model and get the evaluation results on the training set
    print("Training Linear Regression..")
    train_LR(data)
    print("Successfully Trained Linear Regression")
    print()

    print("Training Lasso Regression..")
    lasso_best_params = train_Lasso(data)
    print("Successfully Trained Lasso Regression")
    print()

    print("Training Ridge Regression..")
    ridge_best_params = train_Ridge(data)
    print("Successfully Trained Ridge Regression")
    print()

    print("Training Decision Tree..")
    dt_best_params = train_DT(data)
    print("Successfully Trained Decision Tree")
    print()

    print("Training Random Forest..")
    rf_best_params = train_RF(data)
    print("Successfully Trained Random Forest")
    print()

    print("Best Linear Lasso Regression Parameters")
    print(lasso_best_params)
    print()
    print("=========================================================================")

    print("Best Linear Ridge Regression Parameters")
    print(ridge_best_params)
    print()
    print("=========================================================================")

    print("Best Decision Tree Parameters")
    print(dt_best_params)
    print()
    print("=========================================================================")

    print("Best Random Forest Parameters")
    print(rf_best_params)
    print()
    print("=========================================================================")


if __name__ == "__main__":
    main()
