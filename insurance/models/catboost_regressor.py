import warnings
from collections import namedtuple

import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_log_error

warnings.filterwarnings("ignore")


def train_catboost_regressor(params: namedtuple) -> CatBoostRegressor:
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
            "eval_metric": "MSLE",
            "verbose": False,
        }

        model = CatBoostRegressor(**param)
        model.fit(
            params.X_train,
            params.y_train,
            eval_set=[(params.X_val, params.y_val)],
        )

        y_pred = model.predict(params.X_val)
        score = root_mean_squared_log_error(y_pred, params.y_val)

        return score

    params.logger.info("Starting study with Optuna")
    study = optuna.create_study(direction="minimize", study_name="CatoBoostRegressor")
    study.optimize(objective, n_trials=10)

    best_model = CatBoostRegressor(**study.best_params, verbose=False)
    best_model.fit(params.X_train, params.y_train)

    params.logger.info(f"Best model parameters {study.best_params}")
    params.logger.info(f"Best model score: {study.best_value}")

    return best_model
