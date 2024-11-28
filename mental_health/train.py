import os
import uuid
import pickle
from typing import Any, Tuple
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

from tools.task import Data, Task
from tools.logger import log_method_call
from tools.schema import Schema
from mental_health.models.tabnet import train_tabnet
from mental_health.models.catboost import train_catboost
from mental_health.models.light_lgbm import train_lightlgbm
from mental_health.models.xgboosting import train_xgboosting
from mental_health.models.booster_ensemble import train_booster_ensemble


class MentalHealthTrain(Task):
    config_path = "./mental_health/config.yml"

    def __init__(self) -> None:
        super().__init__()
        self.model = os.environ.get("MODEL")
        self.stratification = os.environ.get("STRATIFICATION")

    def run_task(self) -> None:
        dfs = self._load_datasets()
        dfs.schema = self._load_schema()

        X, y = self._select_features_and_labels(dfs=dfs)
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

        if self.stratification == "--stratification":
            submission = self._create_submission_dataframe_with_stratification(X=X, y=y, dfs=dfs, model=model)
        else:
            self._evaluate_model_on_val_set(X_test=X_test, y_test=y_val, model=model)
            submission = self._create_submission_dataframe(dfs=dfs, model=model)

        self._save_submission(submission=submission)

    @log_method_call
    def _load_datasets(self) -> Data:
        df_train = pd.read_csv(self.config.paths.train_preprocessed)
        df_test = pd.read_csv(self.config.paths.test_preprocessed)

        self.logger.info(f"Dataframe df_train: {len(df_train)} rows x {len(df_train.columns)} columns.")
        self.logger.info(f"Dataframe df_test: {len(df_test)} rows x {len(df_test.columns)} columns.")

        return Data(train=df_train, test=df_test)

    @log_method_call
    def _load_schema(self) -> Schema:
        with Path(self.config.paths.schema).open("rb") as f:
            return pickle.load(f)

    @log_method_call
    def _select_features_and_labels(self, dfs: Data) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = dfs.train[dfs.schema.numeric_features() + dfs.schema.catvar_features()]
        y = dfs.train[dfs.schema.labels]
        return X, y

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
    def _training_model(self, params: namedtuple) -> Any:
        match self.model:
            case "--catboost":
                return train_catboost(params=params)
            case "--light-lgbm":
                return train_lightlgbm(params=params)
            case "--xgboosting":
                return train_xgboosting(params=params)
            case "--booster-ensemble":
                return train_booster_ensemble(params=params)
            case "--tabnet":
                return train_tabnet(params=params)
            case _:
                raise KeyError(f"Model {self.model} not found!")

    @log_method_call
    def _evaluate_model_on_val_set(
        self,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        model: Any,
    ) -> None:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred=y_pred)
        self.logger.info(f"Accuracy on test set: {accuracy:.4f}")
        report = classification_report(y_true=y_test, y_pred=y_pred)
        self.logger.info(f"Classification report \n {report}")

    @log_method_call
    def _create_submission_dataframe(self, dfs: Data, model: Any) -> pd.DataFrame:
        submission_pred = model.predict(dfs.test[dfs.schema.numeric_features() + dfs.schema.catvar_features()])
        submission = pd.DataFrame({"id": dfs.test.id, "Depression": submission_pred})
        return submission

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

            score = accuracy_score(y_valid_fold, pred_valid_fold)
            scores.append(score)
            test_df_pred = model_fold.predict(dfs.test[dfs.schema.numeric_features() + dfs.schema.catvar_features()])
            test_predictions.append(test_df_pred)
            self.logger.info(f"Fold {i + 1} Accuracy Score: {score}")

        self.logger.info(f"mean Accuracy Score: {np.mean(scores):.4f}")

        submission = pd.DataFrame(
            {
                "id": dfs.test.id,
                "Depression": np.round(np.sum(test_predictions, axis=0) / 10),
            }
        )

        return submission

    @log_method_call
    def _save_submission(self, submission: pd.DataFrame) -> None:
        submission_identifier = str(uuid.uuid4())[:8]
        submission.to_csv(Path(self.config.paths.data) / f"submission_{submission_identifier}.csv", index=False)
        self.logger.info(f"Submission {submission_identifier} was successfully saved!")
