import warnings
from pathlib import Path
from collections import namedtuple

import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_log_error

from tools.load import load_parameters
from tools.save import save_parameters

warnings.filterwarnings("ignore")


def train_catboost_regressor(params: namedtuple) -> CatBoostRegressor:
    params.logger.info("Starting CatBoostRegressor Training")

    params.X_train[params.schema.catvar_features()] = params.X_train[params.schema.catvar_features()].astype(int)
    params.X_val[params.schema.catvar_features()] = params.X_val[params.schema.catvar_features()].astype(int)

    params_path = Path(params.config.paths.data) / "catboost_regressor.json"

    if params_path.exists():
        best_params = load_parameters(filepath=params_path, logger=params.logger)
    else:
        params.logger.info("Starting study with Optuna")

        func = lambda trial: objective(trial, params)
        study = optuna.create_study(direction="minimize", study_name="CatBoostRegressor")
        study.optimize(func, n_trials=15)
        best_params = study.best_params

    best_model = CatBoostRegressor(**best_params, verbose=False)
    best_model.fit(params.X_train, params.y_train, early_stopping_rounds=10)

    params.logger.info(f"Best model parameters {best_params}")
    params.logger.info(f"Best model score: {best_params}")

    save_parameters(path=params_path, parameters=best_params, logger=params.logger)
    return best_model


def objective(trial, params: namedtuple) -> float:
    param = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        "depth": trial.suggest_int("depth", 3, 10),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 0.01, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.1, 10.0),
        "cat_features": params.schema.catvar_features(),  # Specify categorical features if any
        "random_seed": params.config.random_state,
        "eval_metric": "MSLE",
        "verbose": False,
    }

    model = CatBoostRegressor(**param)
    model.fit(
        params.X_train,
        params.y_train,
        early_stopping_rounds=10,
        eval_set=[(params.X_val, params.y_val)],
    )

    y_pred = model.predict(params.X_val)
    score = root_mean_squared_log_error(y_pred, params.y_val)

    return score
