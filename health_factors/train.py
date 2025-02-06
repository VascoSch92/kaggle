import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from tools.load import load_schema, load_from_csv
from tools.save import save_submission_as_csv
from tools.task import Data, Task
from tools.logger import log_method_call
from tools.pandas import select_features_and_labels
from tools.models.classification.light_lgbm import train_light_lgbm
from tools.models.classification.xgboosting import train_xgboost


class HealthFactorsTrain(Task):
    config_path = "./health_factors/config.yml"

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

        X, y = select_features_and_labels(dfs=dfs, logger=self.logger)
        X_train, X_test, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.config.train.test_size,
            random_state=self.config.random_state,
        )

        self.config.add(
            {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_test,
                "y_val": y_val,
                "schema": dfs.schema,
                "logger": self.logger,
            }
        )

        model = self._training_model()

        submission = self._create_submission(X=X, y=y, dfs=dfs, model=model)
        save_submission_as_csv(
            path=self.config.paths.data,
            submission=submission,
            logger=self.logger,
        )

    @log_method_call
    def _training_model(self) -> Any:
        match self.model:
            case "--light-lgbm":
                return train_light_lgbm(config=self.config)
            case "--xgboost":
                return train_xgboost(config=self.config)
            case _:
                raise KeyError(f"Model {self.model} not found!")

    @log_method_call
    def _create_submission(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        dfs: Data,
        model: Any,
    ) -> pd.DataFrame:
        match self.submission:
            case "--stratification":
                return self._create_submission_dataframe_with_stratification(X=X, y=y, dfs=dfs, model=model)
            case _:
                self._evaluate_model_on_val_set(X_val=self.config.X_val, y_val=self.config.y_val, model=model)
                return self._create_submission_dataframe(dfs=dfs, model=model)

    @log_method_call
    def _evaluate_model_on_val_set(
        self,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        model: Any,
    ) -> None:
        y_pred = model.predict_proba(X_val)[:, 1]
        accuracy = roc_auc_score(y_true=y_val, y_score=y_pred)
        self.logger.info(f"ROC AUC on validation set: {accuracy:.4f}")

    @log_method_call
    def _create_submission_dataframe(self, dfs: Data, model: Any) -> pd.DataFrame:
        submission_pred = model.predict_proba(dfs.test[dfs.schema.numeric_features() + dfs.schema.catvar_features()])[
            :, 1
        ]
        submission = pd.DataFrame({"ID": dfs.test.ID, "PCOS": submission_pred})
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

            pred_valid_fold = model_fold.predict_proba(X_valid_fold)[:, 1]

            score = roc_auc_score(y_valid_fold, pred_valid_fold)
            scores.append(score)
            test_df_pred = model_fold.predict_proba(
                dfs.test[dfs.schema.numeric_features() + dfs.schema.catvar_features()]
            )[:, 1]
            test_predictions.append(test_df_pred)
            self.logger.info(f"Fold {i + 1} AUC Score: {score}")

        self.logger.info(f"AUC Score: {np.mean(scores):.4f}")

        submission = pd.DataFrame(
            {
                "ID": dfs.test.ID,
                "PCOS": np.mean(test_predictions, axis=0),
            }
        )

        return submission
