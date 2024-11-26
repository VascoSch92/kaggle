import warnings

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")


def train_lightlgbm(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    schema,
    config,
    logger,
) -> LGBMClassifier:
    logger.info("Starting LGBM Training")

    def objective(trial):
        param_grid = {
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "random_state": config.random_state,
            "verbose": -1,
        }

        cat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        model = LGBMClassifier(**param_grid)

        model.fit(X_train, y_train)
        scores = cross_val_score(model, X_train, y_train, cv=cat_cv, scoring="accuracy")

        return scores.mean()

    logger.info("Start study")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_model = LGBMClassifier(**study.best_params)
    best_model.fit(X_train, y_train)

    logger.info(f"Best parameters found: {study.best_params}")
    logger.info(f"Best model score: {study.best_value}")

    return best_model
