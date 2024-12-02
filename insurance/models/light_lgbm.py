import warnings
from collections import namedtuple

import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_log_error

warnings.filterwarnings("ignore")


def train_light_lgbm(params: namedtuple) -> LGBMRegressor:
    params.logger.info("Starting LGBMClassifier Training")

    def objective(trial):
        param_grid = {
            "metric": "neg_mean_squared_log_error",
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
            "num_threads": -1,  # Use a single thread,
        }

        model = LGBMRegressor(**param_grid)
        model.fit(
            params.X_train,
            params.y_train,
            eval_set=[(params.X_val, params.y_val)],
            eval_names=["validation"],
            eval_metric="neg_mean_squared_log_error",
        )

        y_pred = model.predict(params.X_val)
        score = root_mean_squared_log_error(y_pred, params.y_val)

        return score

    params.logger.info("Start study")
    study = optuna.create_study(direction="minimize", study_name="LGBMRegressor")
    study.optimize(objective, n_trials=10)

    best_model = LGBMRegressor(**study.best_params)
    best_model.fit(
        params.X_train,
        params.y_train,
        eval_set=[(params.X_val, params.y_val)],
        eval_names=["validation"],
        eval_metric="neg_mean_squared_log_error",
    )

    params.logger.info(f"Best parameters found: {study.best_params}")
    params.logger.info(f"Best model score: {study.best_value}")
    # params.logger.info(
    #     f"Best validation score: {best_model.evals_result_['validation']['neg_mean_squared_log_error'][-1]}"
    # )

    return best_model
