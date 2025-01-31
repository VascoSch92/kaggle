import warnings
from collections import namedtuple

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


def train_hist_gradient_boosting(params: namedtuple) -> HistGradientBoostingClassifier:
    params.logger.info("Starting HistGradientBoostingClassifier Training")

    params.logger.info("Start study with Optuna")

    func = lambda trial: objective(trial, params)
    study = optuna.create_study(direction="minimize", study_name="HistGradientBoostingRegressor")
    study.optimize(func, n_trials=15)
    best_params = study.best_params

    best_model = HistGradientBoostingClassifier(**best_params)
    best_model.fit(params.X_train, params.y_train)

    y_val_pred = best_model.predict(params.X_val)
    val_score = root_mean_squared_error(y_val_pred, params.y_val)

    params.logger.info(f"Best parameters found: {best_params}")
    params.logger.info(f"Validation score: {val_score:.4f}")

    return best_model


def objective(trial, params: namedtuple) -> float:
    param_grid = {
        "loss": "log_loss",
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 50),
        "l2_regularization": trial.suggest_loguniform("l2_regularization", 1e-4, 10),
        "random_state": params.config.random_state,
        "categorical_features": params.schema.catvar_features(),
        "verbose": 0,
    }

    model = HistGradientBoostingClassifier(**param_grid)
    model.fit(params.X_train, params.y_train)

    y_val_pred = model.predict(params.X_val)
    val_score = recall_score(y_val_pred, params.y_val)
    scores = cross_val_score(model, params.X_train, params.y_train, cv=5, scoring="recall")

    return (np.mean(scores) + val_score) / 2
