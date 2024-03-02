import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from dataprocessing import data_processing
from sklearn.feature_extraction import DictVectorizer
from utils import TRAIN_DATASET_PATH, DROP_COLUMNS, TARGET_COLUMN, BEST_PARAMS


def load_data(data_path, drop_columns, target_column):

    data = data_processing(data_path, drop_columns)
    X = data
    y = data.pop(target_column)
    return (X, y)


def encode_data(X, y):

    X = X.to_dict(orient="record")
    y = np.log1p(y)

    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(X)
    return X, y, vectorizer


def train_best(data, params):

    X, y = data
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    return model


def save_model(model, vectorizer):

    with open("model.pkl", "wb") as f_in:
        pickle.dump([model, vectorizer], f_in)
    print("Model saved successfully")


def main(data_path, drop_columns, target_column, params):

    X, y = load_data(data_path, drop_columns, target_column)
    X, y, vectorizer = encode_data(X, y)
    model = train_best((X, y), params)
    save_model(model, vectorizer)


if __name__ == "__main__":
    main(TRAIN_DATASET_PATH, DROP_COLUMNS, TARGET_COLUMN, BEST_PARAMS)
