import warnings
from pathlib import Path
from collections import namedtuple

import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

from tools.load import load_parameters
from tools.save import save_parameters

warnings.filterwarnings("ignore")


def train_xgboost_regressor(params: namedtuple) -> XGBRegressor:
    params.logger.info("Starting XGBooster Training")

    params.X_train[params.schema.catvar_features()] = params.X_train[params.schema.catvar_features()].astype(int)
    params.X_val[params.schema.catvar_features()] = params.X_val[params.schema.catvar_features()].astype(int)

    params_path = Path(params.config.paths.data) / "xgboost_regressor.json"

    if params_path.exists():
        best_params = load_parameters(filepath=params_path, logger=params.logger)
    else:
        params.logger.info("Starting study with Optuna")

        func = lambda trial: objective(trial, params)
        study = optuna.create_study(direction="minimize", study_name="XGBooster")
        study.optimize(func, n_trials=10)
        best_params = study.best_params

    best_model = XGBRegressor(**best_params)
    best_model.fit(params.X_train, params.y_train)

    y_val_pred = best_model.predict(params.X_val)
    val_score = root_mean_squared_error(y_val_pred, params.y_val)

    params.logger.info(f"Best model parameters {best_params}")
    params.logger.info(f"validation score: {val_score:.5f}")

    save_parameters(path=params_path, parameters=best_params, logger=params.logger)
    return best_model


def objective(trial, params: namedtuple) -> float:
    param = {
        "objective": "reg:squarederror",
        "eval_metric": "rmsle",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "random_state": params.config.random_state,
        "use_label_encoder": False,
        "enable_categorical": True,
    }

    model = XGBRegressor(**param)
    model.fit(params.X_train, params.y_train, verbose=False)

    y_val_pred = model.predict(params.X_val)
    val_score = root_mean_squared_error(y_val_pred, params.y_val)
    scores = cross_val_score(model, params.X_train, params.y_train, cv=5, scoring="neg_mean_squared_error")

    return (np.mean(-1 * scores) + val_score) / 2
