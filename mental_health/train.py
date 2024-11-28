import os
import uuid
import pickle
from typing import Any, Tuple
from pathlib import Path

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
        self.pipeline = os.environ.get("PIPELINE")

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

        model = self._training_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_val,
            schema=dfs.schema,
        )

        # self._evaluate_model_on_val_set(X_test=X_test, y_test=y_val, model=model)
        # submission = self._create_submission_dataframe(dfs=dfs, model=model)

        submission = self._create_submission_dataframe_with_stratification(X=X, y=y, dfs=dfs, model=model)
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
    def _training_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        schema: Schema,
    ) -> Any:
        match self.pipeline:
            case "--catboost":
                return train_catboost(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_val,
                    y_test=y_val,
                    schema=schema,
                    config=self.config,
                    logger=self.logger,
                )
            case "--light-lgbm":
                return train_lightlgbm(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_val,
                    y_test=y_val,
                    schema=schema,
                    config=self.config,
                    logger=self.logger,
                )
            case "--xgboosting":
                return train_xgboosting(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_val,
                    y_test=y_val,
                    schema=schema,
                    config=self.config,
                    logger=self.logger,
                )
            case "--booster-ensemble":
                return train_booster_ensemble(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_val,
                    y_test=y_val,
                    schema=schema,
                    config=self.config,
                    logger=self.logger,
                )
            case "--tabnet":
                return train_tabnet(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_val,
                    y_test=y_val,
                    schema=schema,
                    config=self.config,
                    logger=self.logger,
                )
            case _:
                raise KeyError

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
            test_df_pred = model_fold.predict_proba(
                dfs.test[dfs.schema.numeric_features() + dfs.schema.catvar_features()]
            )[:, 1]
            test_predictions.append(test_df_pred)
            self.logger.info(f"Fold {i + 1} Accuracy Score: {score}")

        self.logger.info(f"mean Accuracy Score: {np.mean(scores):.4f}")

        submission = pd.DataFrame(
            {
                "id": dfs.test.id,
                "Depression": np.round(np.mean(test_predictions, axis=0)),
            }
        )

        return submission

    @log_method_call
    def _save_submission(self, submission: pd.DataFrame) -> None:
        submission_identifier = str(uuid.uuid4())[:8]
        submission.to_csv(Path(self.config.paths.data) / f"submission_{submission_identifier}.csv", index=False)
        self.logger.info(f"Submission {submission_identifier} was successfully saved!")
