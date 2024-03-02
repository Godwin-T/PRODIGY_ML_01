import pandas as pd


def load_data(data_path: str):

    data = pd.read_csv(data_path)
    return data


def remove_redundant_features(data):

    uneeded_columns = []

    for column in data.columns:
        percentage = data[column].isna().sum() * 100 / len(data)
        if percentage > 30:
            uneeded_columns.append([column, percentage])

    drop = []
    for i, j in uneeded_columns:
        drop.append(i)

    data.drop(columns=drop, axis=1, inplace=True)
    return data


def fill_numerical_values_with_mean(dataset=pd.DataFrame(), column=str):

    mean = dataset[column].mean()
    dataset[column] = dataset[column].fillna(mean)
    return dataset[column]


def fill_categorical_values_with_mode(dataset, column):
    mode = dataset[column].mode()[0]
    dataset[column] = dataset[column].fillna(mode)
    return dataset[column]


def handle_numerical_missing_values(data, num_columns):

    for column in num_columns:
        data[column] = fill_numerical_values_with_mean(data, column)
    return data


def handle_categorical_missing_values(data, cat_columns):

    for column in cat_columns:

        data[column] = fill_categorical_values_with_mode(data, column)

    return data


def feature_engineering(data):

    data["Totalarea"] = data["LotArea"] + data["LotFrontage"]
    data["TotalBsmtFin"] = data["BsmtFinSF1"] + data["BsmtFinSF2"]
    data["TotalSF"] = data["TotalBsmtSF"] + data["2ndFlrSF"]
    data["TotalBath"] = data["FullBath"] + data["HalfBath"]
    data["TotalPorch"] = (
        data["ScreenPorch"] + data["EnclosedPorch"] + data["OpenPorchSF"]
    )
    return data


def remove_columns(data, columns):

    data = data.drop(columns, axis=1)
    return data


def bool_to_int(data):

    for col in data.columns:
        if data[col].dtype == "bool":
            data[col] = data[col].astype("int32")

    return data


def data_processing(data_path, drop_columns):

    data = load_data(data_path)
    data = remove_redundant_features(data)
    numerical_cols = data.dtypes[data.dtypes != "object"].index.tolist()
    categorical_cols = data.dtypes[data.dtypes == "object"].index.tolist()

    data = handle_categorical_missing_values(data, categorical_cols)
    data = handle_numerical_missing_values(data, numerical_cols)

    data = feature_engineering(data)
    data = remove_columns(data, drop_columns)
    data = bool_to_int(data)
    return data
