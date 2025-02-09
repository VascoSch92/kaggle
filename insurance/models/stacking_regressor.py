from sklearn.ensemble import StackingRegressor

from insurance.models.light_lgbm_regressor import train_light_lgbm_regressor
from insurance.models.hist_gradient_boosting_regressor import train_hist_gradient_boosting_regressor


def train_stacking_regressor(params) -> StackingRegressor:
    params.logger.info("Starting StackingRegressor Training")
    lightlgbm = train_light_lgbm_regressor(params=params)
    histboost = train_hist_gradient_boosting_regressor(params=params)
    model = StackingRegressor(
        estimators=[
            ("lightlgbm", lightlgbm),
            ("histboost", histboost),
        ],
        final_estimator=lightlgbm,
        verbose=False,
    )

    model.fit(params.X_train, params.y_train)
    return model
