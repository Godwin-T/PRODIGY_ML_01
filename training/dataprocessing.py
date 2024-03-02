import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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


def fill_numerical_values_with_linear_model(dataset, column):

    data = dataset[[column, "SalePrice"]].copy()
    data[column] = data[column].fillna(-1)
    train = data[data[column] != -1]
    missed_data = pd.DataFrame(data[data[column] == -1]["SalePrice"])

    x_train, x_test, y_train, y_test = train_test_split(
        train.drop(columns=column, axis=1),
        train[column],
        train_size=0.01,
        random_state=42,
    )

    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    predction = list(lin_reg.predict(missed_data))

    def update(value):
        if value == -1:
            ret = int(predction[0])
            predction.pop(0)
            return ret
        return value

    dataset[column] = dataset[column].fillna(-1)
    dataset[column] = dataset[column].apply(update)

    return dataset[column]


def fill_numerical_values_with_mean(dataset=pd.DataFrame(), column=str):

    mean = dataset[column].mean()
    dataset[column] = dataset[column].fillna(mean)
    return dataset[column]


def fill_categorical_values_with_RF_model(dataset, column):

    data = dataset[[column, "SalePrice"]].copy()
    data[column] = data[column].fillna("missed_data")
    train = data[data[column] != "missed_data"]
    missed_data = pd.DataFrame(data[data[column] == "missed_data"]["SalePrice"])

    x_train, x_test, y_train, y_test = train_test_split(
        train.drop(columns=column, axis=1),
        train[column],
        train_size=0.01,
        random_state=42,
    )

    RF = RandomForestClassifier(ccp_alpha=0.015)
    RF.fit(x_train, y_train)
    predction = list(RF.predict(missed_data))

    def update(value):
        if value == "missied_data":
            ret = predction[0]
            predction.pop(0)
            return ret
        return value

    dataset[column] = dataset[column].fillna("missied_data")
    dataset[column] = dataset[column].apply(update)

    return dataset[column]


def fill_categorical_values_with_mode(dataset, column):
    mode = dataset[column].mode()[0]
    dataset[column] = dataset[column].fillna(mode)
    return dataset[column]


def handle_numerical_missing_values(data, num_columns, data_group):

    for column in num_columns:
        percentage = data[column].isna().sum() * 100 / len(data)

        if percentage <= 3:  # with mean
            data[column] = fill_numerical_values_with_mean(data, column)

        else:  # with model
            if data_group == "train":
                data[column] = fill_numerical_values_with_linear_model(data, column)
            else:
                data[column] = fill_numerical_values_with_mean(data, column)
    return data


def handle_categorical_missing_values(data, cat_columns):

    for column in cat_columns:
        percentage = data[column].isna().sum() * 100 / len(data)

        if percentage <= 3:
            data[column] = fill_categorical_values_with_mode(data, column)

        else:  # with model
            data[column] = fill_categorical_values_with_RF_model(data, column)
    return data


def remove_outliers(data, data_group):

    if data_group == "train":
        data = data.drop(data[data["LotFrontage"] > 185].index)
        data = data.drop(data[data["LotArea"] > 100000].index)
        data = data.drop(data[data["BsmtFinSF1"] > 4000].index)
        data = data.drop(data[data["TotalBsmtSF"] > 5000].index)
        data = data.drop(data[data["GrLivArea"] > 4000].index)

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


def data_processing(data_path, drop_columns, data_group="train"):

    data = load_data(data_path)
    data = remove_redundant_features(data)
    data = remove_outliers(data, data_group)
    numerical_cols = data.dtypes[data.dtypes != "object"].index.tolist()
    categorical_cols = data.dtypes[data.dtypes == "object"].index.tolist()

    data = handle_categorical_missing_values(data, categorical_cols)
    data = handle_numerical_missing_values(data, numerical_cols, data_group)

    data = feature_engineering(data)
    data = remove_columns(data, drop_columns)
    data = bool_to_int(data)
    return data
