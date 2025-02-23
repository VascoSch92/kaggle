import warnings
from collections import namedtuple

import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")


def train_lightlgbm(params: namedtuple) -> LGBMClassifier:
    params.logger.info("Starting LGBMClassifier Training")

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
            "random_state": params.config.random_state,
            "verbose": -1,
        }

        cat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        model = LGBMClassifier(**param_grid)
        model.fit(
            params.X_train,
            params.y_train,
            eval_set=[(params.X_val, params.y_val)],
            eval_names=["validation"],
            eval_metric="average_precision",
        )

        scores = cross_val_score(model, params.X_train, params.y_train, cv=cat_cv, scoring="accuracy")

        return (scores.mean() + model.evals_result_["validation"]["average_precision"][-1]) / 2

    params.logger.info("Start study")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)

    best_model = LGBMClassifier(**study.best_params)
    best_model.fit(
        params.X_train,
        params.y_train,
        eval_set=[(params.X_val, params.y_val)],
        eval_names=["validation"],
        eval_metric="average_precision",
    )

    params.logger.info(f"Best parameters found: {study.best_params}")
    params.logger.info(f"Best model score: {study.best_value}")
    params.logger.info(f"Best validation score: {best_model.evals_result_['validation']['average_precision'][-1]}")

    return best_model
