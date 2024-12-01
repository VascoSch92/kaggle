from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from spaceship_titanic.models.catboost import train_catboost
from spaceship_titanic.models.light_lgbm import train_light_lgbm
from spaceship_titanic.models.xgboosting import train_xgboosting


def train_booster_stacking(params: "Params") -> StackingClassifier:
    print("Start training of StackingClassifier with CatBoost, Light-LGBM, XGBoosting and LogistcRegression")
    lightlgbm = train_light_lgbm(params=params)
    catboost = train_catboost(params=params)
    xgboosting = train_xgboosting(params=params)
    model = StackingClassifier(
        estimators=[
            ("catboost", catboost),
            ("lightlgbm", lightlgbm),
            ("xgboost", xgboosting),
        ],
        final_estimator=LogisticRegression(verbose=False),
        verbose=False,
    )

    model.fit(params.X_train, params.y_train)
    return model
