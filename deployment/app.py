import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from dataprocessing import data_processing
from utils import MODEL_PATH, DROP_COLUMNS


def load_model(model_path):

    with open(model_path, "rb") as f_out:
        model, vectorizer = pickle.load(f_out)
    return model, vectorizer


def load_and_process_data(data_path, drop_columns):

    data = data_processing(data_path, drop_columns)
    house_id = data.pop("Id")
    return data, house_id


def save_prediction(house_id, prediction):

    prediction = np.round(np.expm1(prediction))
    dicts = {"Id": house_id, "SalePrice": prediction}
    output = pd.DataFrame(dicts)
    output.to_csv("prediction.csv", index=False)
    print("Predictions have been successfully saved to as csv file")


app = Flask("house price")


@app.route("/predict", methods=["POST"])
# # @flow
def predict():

    data_path = request.get_json()
    model, vectorizer = load_model(MODEL_PATH)

    data, house_id = load_and_process_data(data_path, DROP_COLUMNS)
    data = data.to_dict(orient="records")

    data = vectorizer.transform(data)
    prediction = model.predict(data)
    save_prediction(house_id, prediction)
    return jsonify({"Status": "Successfully Saved Prediction To Directory"})
