from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from product_defect.models.catboost import train_catboost
from product_defect.models.light_lgbm import train_light_lgbm
from product_defect.models.xgboosting import train_xgboosting


def train_stacking_ensemble(params: "Params") -> StackingClassifier:
    params.logger.info("Start training of StackingClassifier.")
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
