from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier

from product_defect.models.xgboosting import train_xgboosting


def train_bagging_ensemble(params: "Params") -> BaggingClassifier:
    params.logger.info("Start BaggingClassifier Training.")
    boosting = train_xgboosting(params=params)

    model = BaggingClassifier(
        estimator=XGBClassifier(**boosting.get_params()),
        n_estimators=10,
        random_state=params.config.random_state,
        verbose=False,
    )

    model.fit(params.X_train, params.y_train)
    return model
