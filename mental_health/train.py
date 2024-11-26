import os
import uuid
import pickle
from typing import Any, Tuple
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from mental_health.models.booster_ensemble import train_booster_ensemble
from tools.task import Data, Task
from tools.logger import log_method_call
from tools.schema import Schema
from mental_health.models.catboost import train_catboost
from mental_health.models.light_lgbm import train_lightlgbm
from mental_health.models.xgboosting import train_xgboosting


class MentalHealthTrain(Task):
    config_path = "./mental_health/config.yml"

    def __init__(self) -> None:
        super().__init__()
        self.pipeline = os.environ.get("PIPELINE")

    def run_task(self) -> None:
        dfs = self._load_datasets()
        dfs.schema = self._load_schema()

        X, y = self._select_features_and_labels(dfs=dfs)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.config.random_state,
        )

        model = self._training_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            schema=dfs.schema,
        )

        self._evaluate_model_on_test_set(X_test=X_test, y_test=y_test, model=model)
        self._create_and_save_submission(dfs=dfs, model=model)

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
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        schema: Schema,
    ) -> Any:
        match self.pipeline:
            case "--catboost":
                return train_catboost(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    schema=schema,
                    config=self.config,
                    logger=self.logger,
                )
            case "--light-lgbm":
                return train_lightlgbm(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    schema=schema,
                    config=self.config,
                    logger=self.logger,
                )
            case "--xgboosting":
                return train_xgboosting(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    schema=schema,
                    config=self.config,
                    logger=self.logger,
                )
            case "--booster-ensemble":
                return train_booster_ensemble(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    schema=schema,
                    config=self.config,
                    logger=self.logger,
                )
            case _:
                raise KeyError

    @log_method_call
    def _evaluate_model_on_test_set(
        self,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        model: Any,
    ) -> None:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred=y_pred)
        self.logger.info(f"Accuracy on test set: {accuracy:.2f}")
        report = classification_report(y_true=y_test, y_pred=y_pred)
        self.logger.info(f"Classification report \n {report}")

    @log_method_call
    def _create_and_save_submission(self, dfs: Data, model: Any) -> None:
        id = dfs.test.id
        submission_pred = model.predict(dfs.test[dfs.schema.numeric_features() + dfs.schema.catvar_features()])
        output = pd.DataFrame({"id": id, "Depression": submission_pred})
        submission_identifier = str(uuid.uuid4())[:8]
        output.to_csv(Path(self.config.paths.data) / f"submission_{submission_identifier}.csv", index=False)

        self.logger.info(f"Submission {submission_identifier} was successfully saved!")
