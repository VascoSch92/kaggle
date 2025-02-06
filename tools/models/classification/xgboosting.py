import warnings

import optuna
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from tools.config import Config

warnings.filterwarnings("ignore")


def train_xgboost(config: Config) -> XGBClassifier:
    config.logger.info("Starting XGBClassifier Training")

    config.X_train[config.schema.catvar_features()] = config.X_train[config.schema.catvar_features()].astype(int)
    config.X_val[config.schema.catvar_features()] = config.X_val[config.schema.catvar_features()].astype(int)

    X = pd.concat([config.X_train, config.X_val], axis=0, ignore_index=True)
    y = pd.concat([config.y_train, config.y_val], axis=0, ignore_index=True)

    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()
    scale_pos_weight = (negative_count / positive_count).values[0]

    def objective(trial):
        model = XGBClassifier(
            objective="binary:logistic",
            booster=trial.suggest_categorical("booster", ["gbtree", "dart", "gblinear"]),
            eta=trial.suggest_float("eta", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            subsample=trial.suggest_float("subsample", 0.1, 1.),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
            tree_method=trial.suggest_categorical("tree_method", ["auto", "hist", "approx"]),
            scale_pos_weight=scale_pos_weight,
            random_state=config.random_state,
            use_label_encoder=False,
            enable_categorical=True,
        )

        cat_cv = StratifiedKFold(
            n_splits=config.xgboost.cross_validation.n_splits,
            shuffle=True,
            random_state=config.random_state,
        )
        scores = cross_val_score(model, X, y, cv=cat_cv, scoring=config.xgboost.cross_validation.scoring)

        return scores.mean()

    config.logger.info(f"Start Optuna study with {config.xgboost.optuna.n_trials} trials")
    study = optuna.create_study(direction=config.xgboost.optuna.direction)
    study.optimize(objective, n_trials=config.xgboost.optuna.n_trials)

    best_model = XGBClassifier(**study.best_params)
    best_model.fit(config.X_train, config.y_train, eval_set=[(config.X_val, config.y_val)], verbose=False)

    config.logger.info(f"Best model parameters {study.best_params}")
    config.logger.info(f"Best model score: {study.best_value}")

    return best_model
