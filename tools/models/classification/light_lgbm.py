import warnings

import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd

from tools.config import Config

warnings.filterwarnings("ignore")


def train_light_lgbm(config: Config) -> LGBMClassifier:
    config.logger.info("Starting LGBMClassifier Training")

    X = pd.concat([config.X_train, config.X_val], axis=0, ignore_index=True)
    y = pd.concat([config.y_train, config.y_val], axis=0, ignore_index=True)

    def objective(trial):
        model = LGBMClassifier(
            boosting_type=trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            num_leaves=trial.suggest_int("num_leaves", 20, 150),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            feature_fraction=trial.suggest_uniform("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 1, 100),
            lambda_l1=trial.suggest_loguniform("lambda_l1", 1e-8, 1),
            lambda_l2=trial.suggest_loguniform("lambda_l2", 1e-8, 1),
            random_state=config.random_state,
            verbose=-1,
            num_threads=1,
        )

        cat_cv = StratifiedKFold(
            n_splits=config.light_lgbm.cross_validation.n_splits,
            shuffle=True,
            random_state=config.random_state,
        )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cat_cv,
            scoring=config.light_lgbm.cross_validation.scoring,
        )

        return scores.mean()

    config.logger.info(f"Start Optuna study with {config.light_lgbm.optuna.n_trials} trials")
    study = optuna.create_study(direction=config.light_lgbm.optuna.direction)
    study.optimize(objective, n_trials=config.light_lgbm.optuna.n_trials)

    best_model = LGBMClassifier(**study.best_params)
    best_model.fit(
        config.X_train,
        config.y_train,
    )

    config.logger.info(f"Best parameters found: {study.best_params}")
    config.logger.info(f"Best model score: {study.best_value}")

    return best_model
