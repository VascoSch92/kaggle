import warnings
from collections import namedtuple

import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")


def train_catboost(params: namedtuple) -> CatBoostClassifier:
    params.logger.info("Starting CatBoost Training")
    params.X_train[params.schema.catvar_features()] = params.X_train[params.schema.catvar_features()].astype(int)
    params.X_val[params.schema.catvar_features()] = params.X_val[params.schema.catvar_features()].astype(int)

    def objective(trial):
        # Define the parameter space to search
        param = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 0.01, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.1, 10.0),
            "cat_features": params.schema.catvar_features(),  # Specify categorical features if any
            "random_seed": params.config.random_state,
            "verbose": False,
        }

        cat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        # Initialize CatBoostClassifier with the suggested parameters
        model = CatBoostClassifier(**param)

        # Fit the model
        model.fit(params.X_train, params.y_train, eval_set)
        scores = cross_val_score(model, params.X_train, params.y_train, cv=cat_cv, scoring="accuracy")
        return scores.mean()

    params.logger.info("Starting study")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    best_model = CatBoostClassifier(**study.best_params, verbose=False)
    best_model.fit(params.X_train, params.y_train)

    params.logger.info(f"Best model parameters {study.best_params}")
    params.logger.info(f"Best model score: {study.best_value}")

    return best_model
