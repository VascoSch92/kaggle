import warnings
from pathlib import Path
from collections import namedtuple

import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import cross_val_score

from tools.load import load_parameters
from tools.save import save_parameters

warnings.filterwarnings("ignore")


def train_light_lgbm_regressor(params: namedtuple) -> LGBMRegressor:
    params.logger.info("Starting LGBMRegressor Training")

    params_path = Path(params.config.paths.data) / "light_lgbm_regressor.json"

    if params_path.exists():
        best_params = load_parameters(filepath=params_path, logger=params.logger)
    else:
        params.logger.info("Start study with Optuna")

        func = lambda trial: objective(trial, params)
        study = optuna.create_study(direction="minimize", study_name="LGBMRegressor")
        study.optimize(func, n_trials=20)
        best_params = study.best_params

    best_model = LGBMRegressor(**best_params)
    best_model.fit(
        params.X_train,
        params.y_train,
        eval_set=[(params.X_val, params.y_val)],
        eval_names=["validation"],
        eval_metric="neg_mean_squared_log_error",
    )

    params.logger.info(f"Best parameters found: {best_params}")
    params.logger.info(f"Best model score: {best_model.best_score_}")

    save_parameters(path=params_path, parameters=best_params, logger=params.logger)
    return best_model


def objective(trial, params: namedtuple) -> float:
    param_grid = {
        "metric": "neg_mean_squared_log_error",
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        "random_state": params.config.random_state,
        "verbose": -1,
        "num_threads": -1,  # Use a single thread,
    }

    model = LGBMRegressor(**param_grid)
    model.fit(
        params.X_train,
        params.y_train,
        eval_set=[(params.X_val, params.y_val)],
        eval_names=["validation"],
        eval_metric="neg_mean_squared_log_error",
    )

    y_val_pred = model.predict(params.X_val)
    val_score = root_mean_squared_log_error(y_val_pred, params.y_val)
    scores = cross_val_score(model, params.X_train, params.y_train, cv=5, scoring="neg_root_mean_squared_log_error")

    return np.sqrt((np.mean(-1 * scores) + val_score) / 2)
