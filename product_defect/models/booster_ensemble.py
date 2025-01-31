from collections import namedtuple

from sklearn.ensemble import VotingClassifier

from product_defect.models.catboost import train_catboost
from product_defect.models.light_lgbm import train_light_lgbm
from product_defect.models.xgboosting import train_xgboosting


def train_booster_ensemble(params: namedtuple) -> VotingClassifier:
    params.logger.info("Starting Booster Training")
    catboost = train_catboost(params=params)
    lightlgbm = train_light_lgbm(params=params)
    xgboosting = train_xgboosting(params=params)
    model = VotingClassifier(
        estimators=[
            ("catboost", catboost),
            ("lightlgbm", lightlgbm),
            ("xgboost", xgboosting),
        ],
        voting="hard",
    )
    model.fit(params.X_train, params.y_train)
    return model
