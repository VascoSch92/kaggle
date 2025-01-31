from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier

from product_defect.models.light_lgbm import train_light_lgbm


def train_bagging_ensemble(params: "Params") -> BaggingClassifier:
    params.logger.info("Start BaggingClassifier Training.")
    light_lgbm = train_light_lgbm(params)
    model = BaggingClassifier(
        estimator=LGBMClassifier(**light_lgbm.get_params()),
        n_estimators=5,
        random_state=params.config.random_state,
        verbose=False,
    )

    model.fit(params.X_train, params.y_train)
    return model
