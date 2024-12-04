import os
from typing import Any, Tuple
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import StratifiedKFold, train_test_split

from tools.load import load_schema, load_from_csv
from tools.save import save_submission_as_csv
from tools.task import Data, Task
from tools.logger import log_method_call
from tools.schema import Schema
from insurance.models.xgboost_regressor import train_xgboost_regressor
from insurance.models.catboost_regressor import train_catboost_regressor
from insurance.models.light_lgbm_regressor import train_light_lgbm_regressor
from insurance.models.random_forest_regressor import train_random_forest_regressor


class InsuranceTrain(Task):
    config_path = "./insurance/config.yml"

    def __init__(self) -> None:
        super().__init__()
        self.model = os.environ.get("MODEL")
        self.submission = os.environ.get("STRATIFICATION")

    def run_task(self) -> None:
        dfs = Data(
            train=load_from_csv(self.config.paths.train_preprocessed, logger=self.logger),
            test=load_from_csv(self.config.paths.test_preprocessed, logger=self.logger),
            schema=load_schema(self.config.paths.schema, logger=self.logger),
        )

        X, y = self._select_features_and_labels(dfs=dfs)
        y = self._transform_labels(y=y)

        X_train, X_test, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.config.random_state,
        )

        params = self.get_params(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_val,
            schema=dfs.schema,
        )

        model = self._training_model(params=params)

        submission = self._create_submission(X=X, y=y, dfs=dfs, model=model, params=params)
        save_submission_as_csv(
            path=self.config.paths.data,
            submission=submission,
            logger=self.logger,
        )

    @log_method_call
    def _select_features_and_labels(self, dfs: Data) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = dfs.train[dfs.schema.numeric_features() + dfs.schema.catvar_features()]
        y = dfs.train[dfs.schema.labels]
        return X, y

    @log_method_call
    def _transform_labels(self, y: pd.DataFrame) -> pd.DataFrame:
        return np.log2(y)

    @log_method_call
    def _inverse_transform_labels(self, y: pd.DataFrame) -> pd.DataFrame:
        return np.exp2(y)

    @log_method_call
    def get_params(
        self,
        X_train: pd.DataFrame = None,
        y_train: pd.DataFrame = None,
        X_val: pd.DataFrame = None,
        y_val: pd.DataFrame = None,
        schema: Schema = None,
    ) -> namedtuple:
        Params = namedtuple("Params", ["X_train", "y_train", "X_val", "y_val", "schema", "config", "logger"])
        return Params(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            schema=schema,
            config=self.config,
            logger=self.logger,
        )

    @log_method_call
    def _training_model(self, params: "Params") -> Any:
        match self.model:
            case "--light-lgbm-regressor":
                return train_light_lgbm_regressor(params=params)
            case "--catboost-regressor":
                return train_catboost_regressor(params=params)
            case "--xgboost-regressor":
                return train_xgboost_regressor(params=params)
            case "--random-forest-regressor":
                return train_random_forest_regressor(params=params)
            case _:
                raise KeyError(f"Model {self.model} not found!")

    @log_method_call
    def _create_submission(
        self, X: pd.DataFrame, y: pd.DataFrame, dfs: Data, model: Any, params: "Params"
    ) -> pd.DataFrame:
        match self.submission:
            case "--stratification":
                return self._create_submission_dataframe_with_stratification(X=X, y=y, dfs=dfs, model=model)
            case _:
                self._evaluate_model_on_val_set(X_val=params.X_val, y_val=params.y_val, model=model)
                return self._create_submission_dataframe(dfs=dfs, model=model)

    @log_method_call
    def _evaluate_model_on_val_set(
        self,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        model: Any,
    ) -> None:
        y_pred = model.predict(X_val)

        rmsle = root_mean_squared_log_error(y_val, y_pred=y_pred)
        self.logger.info(f"RMSLE on validation set: {rmsle:.4f}")

    @log_method_call
    def _create_submission_dataframe(self, dfs: Data, model: Any) -> pd.DataFrame:
        submission_pred = model.predict(dfs.test[dfs.schema.numeric_features() + dfs.schema.catvar_features()])
        submission = pd.DataFrame(
            {"id": dfs.test.id, "Premium Amount": self._inverse_transform_labels(y=submission_pred)}
        )
        return submission

    @log_method_call
    def _create_submission_dataframe_with_stratification(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        dfs: Data,
        model: Any,
    ) -> pd.DataFrame:
        stratified_k_fold = StratifiedKFold(10, shuffle=True, random_state=self.config.random_state)
        splits = stratified_k_fold.split(X, y)

        scores, test_predictions = [], []
        for i, (full_train_idx, valid_idx) in enumerate(splits):
            model_fold = model
            X_train_fold, X_valid_fold = X.loc[full_train_idx], X.loc[valid_idx]
            y_train_fold, y_valid_fold = y.loc[full_train_idx], y.loc[valid_idx]

            model_fold.fit(X_train_fold, y_train_fold)

            pred_valid_fold = model_fold.predict(X_valid_fold)

            score = root_mean_squared_log_error(y_valid_fold, pred_valid_fold)
            scores.append(score)
            test_df_pred = model_fold.predict(dfs.test[dfs.schema.numeric_features() + dfs.schema.catvar_features()])
            test_predictions.append(test_df_pred)
            self.logger.info(f"Fold {i + 1} Accuracy Score: {score}")

        self.logger.info(f"mean RMQL Score: {np.mean(scores):.4f}")

        submission = pd.DataFrame(
            {
                "PassengerId": dfs.test.PassengerId,
                "Transported": np.round(np.mean(test_predictions, axis=0)),
            }
        )
        submission["Transported"] = submission.Transported.astype(bool)

        return submission
