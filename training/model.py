import mlflow
import numpy as np


from hyperopt.pyll import scope
from hyperopt import hp, STATUS_OK, fmin, Trials, tpe

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("House SalePrice Prediction")


def evaluate_model(y_train, y_test, train_prediction, test_prediction):

    train_prediction = np.expm1(train_prediction)
    test_prediction = np.expm1(test_prediction)

    y_train, y_test = np.expm1(y_train), np.expm1(y_test)

    train_mae = mean_absolute_error(y_true=y_train, y_pred=train_prediction)
    train_mse = mean_squared_error(y_true=y_train, y_pred=train_prediction)
    train_rmse = np.sqrt(mean_squared_error(y_true=y_train, y_pred=train_prediction))

    test_mae = mean_absolute_error(y_true=y_test, y_pred=test_prediction)
    test_mse = mean_squared_error(y_true=y_test, y_pred=test_prediction)
    test_rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=test_prediction))

    metrics = {
        "Train_MAE": train_mae,
        "Train_MSE": train_mse,
        "Train_RMSE": train_rmse,
        "Test_MAE": test_mae,
        "Test_MSE": test_mse,
        "Test_RMSE": test_rmse,
    }
    return metrics


def train_LR(data):

    (x_train, y_train, x_test, y_test) = data
    y_train, y_test = np.log1p(y_train), np.log1p(y_test)
    with mlflow.start_run():

        model = LinearRegression()
        model.fit(x_train, y_train)

        mlflow.set_tag("Model", "LinearRegression")
        mlflow.set_tag("Scaler", None)

        train_prediction = model.predict(x_train)
        test_prediction = model.predict(x_test)

        metrics = evaluate_model(y_train, y_test, train_prediction, test_prediction)
        mlflow.log_metrics(metrics)


def train_Ridge(data):

    (x_train, y_train, x_test, y_test) = data
    y_train, y_test = np.log1p(y_train), np.log1p(y_test)

    best_rmse = float("inf")
    alpha_values = np.arange(0, 20, 0.4)
    for val in alpha_values:
        with mlflow.start_run():
            mlflow.set_tag("Model", "Ridge")
            mlflow.set_tag("Scaler", None)
            mlflow.log_param("alpha", val)

            model = Ridge(alpha=val)
            model.fit(x_train, y_train)

            train_prediction = model.predict(x_train)
            test_prediction = model.predict(x_test)

            metrics = evaluate_model(y_train, y_test, train_prediction, test_prediction)
            mlflow.log_metrics(metrics)

        if metrics["Test_RMSE"] < best_rmse:
            alpha = val
    return {"alpha": alpha}


def train_Lasso(data):

    (x_train, y_train, x_test, y_test) = data
    y_train, y_test = np.log1p(y_train), np.log1p(y_test)

    best_rmse = float("inf")
    alpha_values = np.arange(0, 20, 0.4)
    for val in alpha_values:
        with mlflow.start_run():
            mlflow.set_tag("Model", "Lasso")
            mlflow.set_tag("Scaler", None)
            mlflow.log_param("alpha", val)

            model = Lasso(alpha=val)
            model.fit(x_train, y_train)

            train_prediction = model.predict(x_train)
            test_prediction = model.predict(x_test)

            metrics = evaluate_model(y_train, y_test, train_prediction, test_prediction)
            mlflow.log_metrics(metrics)
        if metrics["Test_RMSE"] < best_rmse:
            alpha = val
    return {"alpha": alpha}


def train_DT(data):

    (x_train, y_train, x_test, y_test) = data
    y_train, y_test = np.log1p(y_train), np.log1p(y_test)

    def objective(params):

        with mlflow.start_run():
            mlflow.set_tag("Model", "DecisionTree")
            mlflow.log_params(params)

            model = DecisionTreeRegressor(**params)
            model.fit(x_train, y_train)

            train_prediction = model.predict(x_train)
            test_prediction = model.predict(x_test)

            metrics = evaluate_model(y_train, y_test, train_prediction, test_prediction)
            mlflow.log_metrics(metrics)

        return {"loss": metrics["Test_RMSE"], "status": STATUS_OK}

    space = {
        "max_depth": hp.randint("max_depth", 1, 15),
        "min_samples_split": hp.randint("min_samples_split", 2, 15),
        "min_samples_leaf": hp.randint("min_samples_leaf", 1, 15),
    }

    best_result = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=Trials()
    )
    return best_result


# Random Forest Model
def train_RF(data):

    (x_train, y_train, x_test, y_test) = data
    y_train, y_test = np.log1p(y_train), np.log1p(y_test)

    def objective(params):

        with mlflow.start_run():
            mlflow.set_tag("Model", "RandomForest")
            mlflow.log_params(params)

            model = RandomForestRegressor(**params)
            model.fit(x_train, y_train)

            train_prediction = model.predict(x_train)
            test_prediction = model.predict(x_test)

            metrics = evaluate_model(y_train, y_test, train_prediction, test_prediction)
            mlflow.log_metrics(metrics)

        return {"loss": metrics["Test_RMSE"], "status": STATUS_OK}

    space = {
        "n_estimators": hp.choice("n_estimators", [2, 5, 10, 20, 30, 50, 100]),
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 5)),
        "min_samples_split": hp.randint("min_samples_split", 2, 15),
        "min_samples_leaf": hp.randint("min_samples_leaf", 1, 15),
    }

    best_result = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=Trials()
    )
    return best_result
