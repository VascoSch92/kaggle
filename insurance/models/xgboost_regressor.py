import warnings
from collections import namedtuple

import optuna
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_log_error

warnings.filterwarnings("ignore")


def train_xgboost_regressor(params: namedtuple) -> XGBRegressor:
    params.logger.info("Starting XGBooster Training")

    params.X_train[params.schema.catvar_features()] = params.X_train[params.schema.catvar_features()].astype(int)
    params.X_val[params.schema.catvar_features()] = params.X_val[params.schema.catvar_features()].astype(int)

    def objective(trial):
        param = {
            "objective": "reg:squaredlogerror",
            "eval_metric": "rmsle",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "random_state": params.config.random_state,
            "use_label_encoder": False,
            "enable_categorical": True,
        }

        model = XGBRegressor(**param)
        model.fit(params.X_train, params.y_train, eval_set=[(params.X_val, params.y_val)], verbose=False)

        y_pred = model.predict(params.X_val)
        score = root_mean_squared_log_error(y_pred, params.y_val)

        return score

    params.logger.info("Starting study with Optuna")
    study = optuna.create_study(direction="minimize", study_name="XGBooster")
    study.optimize(objective, n_trials=15)

    best_model = XGBRegressor(**study.best_params)
    best_model.fit(params.X_train, params.y_train, eval_set=[(params.X_val, params.y_val)], verbose=False)

    params.logger.info(f"Best model parameters {study.best_params}")
    params.logger.info(f"Best model score: {study.best_value}")

    return best_model
