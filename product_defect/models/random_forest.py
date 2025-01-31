from collections import namedtuple

import numpy as np
import optuna
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def train_random_forest_classifier(params: namedtuple) -> RandomForestClassifier:
    params.logger.info("Start RandomForestClassifier Training")

    params.logger.info("Start study with Optuna")

    func = lambda trial: objective(trial, params)
    study = optuna.create_study(direction="maximize", study_name="RandomForestRegressor")
    study.optimize(func, n_trials=20)
    best_params = study.best_params

    best_model = RandomForestClassifier(**best_params)
    best_model.fit(params.X_train, params.y_train)

    params.logger.info(f"Best parameters found: {best_params}")

    return best_model


def objective(trial, params: namedtuple) -> float:
    cat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=params.config.random_state)

    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 10, 200),
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        ccp_alpha=trial.suggest_uniform("ccp_alpha", 0, 1),
        class_weight=trial.suggest_categorical("class_weight", ["balanced", None]),
        random_state=params.config.random_state,
    )

    model.fit(params.X_train, params.y_train)

    y_val_pred = model.predict(params.X_val)
    val_score = recall_score(y_val_pred, params.y_val)
    scores = cross_val_score(model, params.X_train, params.y_train, cv=cat_cv, scoring="recall")
    return (np.mean(scores) + val_score) / 2
