import warnings
from collections import namedtuple

import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


def train_xgboosting(params: namedtuple) -> XGBClassifier:
    params.logger.info("Starting XGBooster Training")

    params.X_train[params.schema.catvar_features()] = params.X_train[params.schema.catvar_features()].astype(int)
    params.X_val[params.schema.catvar_features()] = params.X_val[params.schema.catvar_features()].astype(int)

    negative_count = (params.y_train == 0).sum()
    positive_count = (params.y_train == 1).sum()
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
            "random_state": params.config.random_state,
            "use_label_encoder": False,
            "enable_categorical": True,
            "scale_pos_weight": scale_pos_weight,
        }

        # Initialize the XGBClassifier
        model = XGBClassifier(**param)

        # Train the model
        model.fit(params.X_train, params.y_train, eval_set=[(params.X_val, params.y_val)], verbose=False)

        # Predict and evaluate
        preds = model.predict(params.X_val)
        accuracy = accuracy_score(params.y_val, preds)

        # Return the negative accuracy as Optuna minimizes the objective by default
        return -accuracy

    # Create an Optuna study to maximize the accuracy
    params.logger.info("Starting study")
    study = optuna.create_study(direction="maximize")

    # Start the optimization
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "logloss"

    best_model = XGBClassifier(**best_params)
    best_model.fit(params.X_train, params.y_train, eval_set=[(params.X_val, params.y_val)], verbose=False)

    params.logger.info(f"Best model parameters {study.best_params}")
    params.logger.info(f"Best model score: {study.best_value}")

    return best_model
