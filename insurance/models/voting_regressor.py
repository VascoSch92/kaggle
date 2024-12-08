from collections import namedtuple

from sklearn.ensemble import VotingRegressor

from insurance.models.catboost_regressor import train_catboost_regressor
from insurance.models.light_lgbm_regressor import train_light_lgbm_regressor


def train_voting_regressor(params: namedtuple) -> VotingRegressor:
    params.logger.info("Starting Booster Training")
    catboost = train_catboost_regressor(params=params)
    lightlgbm = train_light_lgbm_regressor(params=params)
    # xgboosting = train_xgboost_regressor(params=params)
    model = VotingRegressor(
        estimators=[
            ("catboost", catboost),
            ("lightlgbm", lightlgbm),
        ],
        verbose=False,
    )
    model.fit(params.X_train, params.y_train)
    return model
