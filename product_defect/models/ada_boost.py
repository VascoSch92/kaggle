from collections import namedtuple

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def train_ada_bost_classifier(params: namedtuple) -> AdaBoostClassifier:
    params.logger.info("Start AdaBoostClassifier Training")

    params.logger.info("Start study with Optuna")

    func = lambda trial: objective(trial, params)
    study = optuna.create_study(direction="maximize", study_name="AdaBoostClassifier")
    study.optimize(func, n_trials=20)
    best_params = study.best_params

    best_model = AdaBoostClassifier(**best_params)
    best_model.fit(params.X_train, params.y_train)

    params.logger.info(f"Best parameters found: {best_params}")

    return best_model


def objective(trial, params: namedtuple) -> float:
    cat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=params.config.random_state)

    model = AdaBoostClassifier(
        estimator=LGBMClassifier(),
        n_estimators=trial.suggest_int("n_estimators", 10, 200),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 2.0),
        random_state=params.config.random_state,
    )

    model.fit(params.X_train, params.y_train)

    y_val_pred = model.predict(params.X_val)
    val_score = recall_score(y_val_pred, params.y_val)
    scores = cross_val_score(model, params.X_train, params.y_train, cv=cat_cv, scoring="recall")
    return (np.mean(scores) + val_score) / 2
