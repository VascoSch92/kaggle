import warnings
from typing import Any

import optuna
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from tools.config import Config

warnings.filterwarnings("ignore")


def _get_estimator(module_name: str, class_name: str) -> Any:
    mod = __import__(module_name)
    cls = getattr(mod, class_name, None)
    if not cls:
        raise ValueError(f"Class '{class_name}' of module '{module_name}' not found.")
    return cls


def train_ada_boost(config: Config) -> AdaBoostClassifier:
    config.logger.info("Starting AdaBoostClassifier Training")

    X = pd.concat([config.X_train, config.X_val], axis=0, ignore_index=True)
    y = pd.concat([config.y_train, config.y_val], axis=0, ignore_index=True)

    estimator = _get_estimator(
        module_name=config.ada_boost.estimator.module_name,
        class_name=config.ada_boost.estimator.class_name,
    )

    def objective(trial):
        model = AdaBoostClassifier(
            estimator=estimator(),
            n_estimators=trial.suggest_int("n_estimators", 10, 200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 2.0),
            random_state=config.random_state,
        )

        cat_cv = StratifiedKFold(
            n_splits=config.ada_boost.cross_validation.n_splits,
            shuffle=True,
            random_state=config.random_state,
        )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cat_cv,
            scoring=config.ada_boost.cross_validation.scoring,
        )

        return scores.mean()

    config.logger.info(f"Start Optuna study with {config.ada_boost.optuna.n_trials} trials")
    study = optuna.create_study(direction=config.ada_boost.optuna.direction)
    study.optimize(objective, n_trials=config.ada_boost.optuna.n_trials)

    best_model = AdaBoostClassifier(**study.best_params)
    best_model.fit(
        config.X_train,
        config.y_train,
    )

    config.logger.info(f"Best parameters found: {study.best_params}")
    config.logger.info(f"Best model score: {study.best_value}")

    return best_model
