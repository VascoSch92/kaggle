import warnings

import optuna
import pandas as pd
import lightgbm as lgb
from lightgbm import Booster
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


def train_lightlgbm(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    schema,
    config,
    logger,
) -> Booster:
    def objective(trial):
        # Suggest values for hyperparameters
        param_grid = {
            "objective": "binary",
            "metric": "binary_error",
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
        }

        # Create LightGBM Dataset
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=schema.catvar_features())
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Train LightGBM model
        gbm = lgb.train(
            param_grid,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
        )

        # Predict on validation set
        preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        preds_binary = [1 if p > 0.5 else 0 for p in preds]

        # Calculate accuracy
        accuracy = accuracy_score(y_test, preds_binary)
        return 1 - accuracy  # Minimize the error

    # Create Optuna study
    logger.info("Start study")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    # Print the best parameters
    logger.info(f"Best parameters found: {study.best_params}")

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=schema.catvar_features())
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    best_model = lgb.train(
        study.best_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
    )

    return best_model
