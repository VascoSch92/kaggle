from pathlib import Path
from collections import namedtuple

import optuna
from sklearn.metrics import root_mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor

from tools.load import load_parameters
from tools.save import save_parameters


def train_random_forest_regressor(params: namedtuple) -> RandomForestRegressor:
    params.logger.info("Start training of RandomForestRegressor")

    params_path = Path(params.config.paths.data) / "random_forest_regressor.json"

    if params_path.exists():
        best_params = load_parameters(filepath=params_path, logger=params.logger)
    else:
        params.logger.info("Start study with Optuna")

        func = lambda trial: objective(trial, params)
        study = optuna.create_study(direction="minimize", study_name="RandomForestRegressor")
        study.optimize(func, n_trials=10)
        best_params = study.best_params

    best_model = RandomForestRegressor(best_params)
    best_model.fit(params.X_train, params.y_train)

    params.logger.info(f"Best parameters found: {best_params}")

    save_parameters(path=params_path, parameters=best_params, logger=params.logger)
    return best_model


def objective(trial, params: namedtuple) -> float:
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 100),
        "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error"]),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "random_state": params.config.random_state,
        "verbose": 0,
    }

    model = RandomForestRegressor(**param_grid)
    model.fit(params.X_train, params.y_train)

    y_val_pred = model.predict(params.X_val)
    val_score = root_mean_squared_log_error(y_val_pred, params.y_val)

    return val_score
