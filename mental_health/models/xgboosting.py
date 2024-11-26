import warnings

import optuna
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


def train_xgboosting(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    schema,
    config,
    logger,
) -> XGBClassifier:
    logger.info("Starting XGBooster Training")

    X_train[schema.catvar_features()] = X_train[schema.catvar_features()].astype(int)
    X_test[schema.catvar_features()] = X_test[schema.catvar_features()].astype(int)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = (negative_count / positive_count).values[0]

    def objective(trial):
        # Define hyperparameter search space
        param = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "random_state": config.random_state,
            "use_label_encoder": False,
            "enable_categorical": True,
            "scale_pos_weight": scale_pos_weight,
        }

        # Initialize the XGBClassifier
        model = XGBClassifier(**param)

        # Train the model
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Predict and evaluate
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        # Return the negative accuracy as Optuna minimizes the objective by default
        return -accuracy

    # Create an Optuna study to maximize the accuracy
    logger.info("Starting study")
    study = optuna.create_study(direction="maximize")

    # Start the optimization
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "logloss"

    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    logger.info(f"Best model parameters {study.best_params}")
    logger.info(f"Best model score: {study.best_value}")

    return best_model
