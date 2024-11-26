import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

from mental_health.models.catboost import train_catboost
from mental_health.models.light_lgbm import train_lightlgbm
from mental_health.models.xgboosting import train_xgboosting


def train_booster_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    schema,
    config,
    logger,
):
    logger.info("Starting Booster Training")
    catboost = train_catboost(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        schema=schema,
        config=config,
        logger=logger,
    )
    lightlgbm = train_lightlgbm(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        schema=schema,
        config=config,
        logger=logger,
    )
    xgboosting = train_xgboosting(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        schema=schema,
        config=config,
        logger=logger,
    )
    model = VotingClassifier(
        estimators=[
            ("catboost", catboost),
            ("lightlgbm", lightlgbm),
            ("xgboost", xgboosting),
        ],
        voting='hard',
    )
    model.fit(X_train, y_train)
    return model

