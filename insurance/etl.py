from typing import List

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler

from tools.load import load_from_csv
from tools.task import Data, Task
from tools.logger import log_method_call
from tools.schema import Schema, ScalerType, EncodingType


class InsuranceEtl(Task):
    config_path = "./insurance/config.yml"

    def run_task(self) -> None:
        dfs = Data(
            train=load_from_csv(self.config.paths.train, logger=self.logger),
            test=load_from_csv(self.config.paths.test, logger=self.logger),
            schema=self._get_schema(),
        )

        dfs = self._fill_missing_values(dfs=dfs)
        dfs = self._features_engineering(dfs=dfs)

        dfs = self._encoding(dfs=dfs)
        dfs = self._scaling(dfs=dfs)

        self._save_preprocessed_dataframe(dfs=dfs)
        dfs.schema.save_as_pickle(filepath=self.config.paths.schema, logger=self.logger)

    @log_method_call
    def _get_schema(self) -> Schema:
        return Schema(
            meta=["id", "Policy Start Date"],
            numeric={
                "Age": ScalerType.STANDARD_SCALER,
                "Annual Income": ScalerType.STANDARD_SCALER,
                "Health Score": ScalerType.STANDARD_SCALER,
                "Credit Score": ScalerType.STANDARD_SCALER,
            },
            catvar={
                "Marital Status": EncodingType.LABEL_ENCODING,
                "Number of Dependents": EncodingType.ORDINAL_ENCODING,
                "Education Level": EncodingType.LABEL_ENCODING,
                "Occupation": EncodingType.LABEL_ENCODING,
                "Location": EncodingType.LABEL_ENCODING,
                "Policy Type": EncodingType.LABEL_ENCODING,
                "Previous Claims": EncodingType.ORDINAL_ENCODING,
                "Vehicle Age": EncodingType.ORDINAL_ENCODING,
                "Insurance Duration": EncodingType.ORDINAL_ENCODING,
                # "Policy Start Date": None,
                "Customer Feedback": EncodingType.LABEL_ENCODING,
                "Smoking Status": EncodingType.LABEL_ENCODING,
                "Exercise Frequency": EncodingType.LABEL_ENCODING,
                "Property Type": EncodingType.LABEL_ENCODING,
            },
            labels=["Premium Amount"],
        )

    @log_method_call
    def _fill_missing_values(self, dfs: Data) -> Data:
        dfs = self._fill_with_mean(dfs=dfs, columns=["Age", "Annual Income", "Health Score", "Credit Score"])
        dfs = self._fill_with_mode(dfs=dfs, columns=["Insurance Duration", "Vehicle Age"])

        dfs.train["Number of Dependents"].fillna(-1, inplace=True)
        dfs.test["Number of Dependents"].fillna(-1, inplace=True)

        dfs.train["Previous Claims"].fillna(-1, inplace=True)
        dfs.test["Previous Claims"].fillna(-1, inplace=True)

        return dfs

    def _fill_with_mean(self, dfs: Data, columns: List[str]) -> Data:
        self.logger.info(f"Fill with mean features: {columns}")
        mean = dfs.train[columns].mean().to_dict()
        dfs.train[columns] = dfs.train[columns].fillna(mean)
        dfs.test[columns] = dfs.test[columns].fillna(mean)
        return dfs

    def _fill_with_mode(self, dfs: Data, columns: List[str]) -> Data:
        self.logger.info(f"Fill with mode features: {columns}")
        for column in columns:
            mode = dfs.train[column].mode()[0]
            dfs.train[column] = dfs.train[column].fillna(mode)
            dfs.test[column] = dfs.test[column].fillna(mode)
        return dfs

    @log_method_call
    def _features_engineering(self, dfs: Data) -> Data:
        # policy start date extract different features with dates
        # smoking status and exercice frequency can be combined to form a new feature
        return dfs

    @log_method_call
    def _encoding(self, dfs: Data) -> Data:
        dfs = self._ordinal_encoding(dfs=dfs)
        dfs = self._label_encoder(dfs=dfs)
        return dfs

    def _label_encoder(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.catvar_features() if dfs.schema.catvar[f] == EncodingType.LABEL_ENCODING]
        if features:
            self.logger.info(f"Label encoding features: {features}")
            label_encoder = LabelEncoder()
            for feature in features:
                dfs.train[feature] = label_encoder.fit_transform(dfs.train[feature])
                dfs.test[feature] = label_encoder.transform(dfs.test[feature])
        return dfs

    def _ordinal_encoding(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.catvar_features() if dfs.schema.catvar[f] == EncodingType.ORDINAL_ENCODING]
        if features:
            self.logger.info(f"Ordinal encoding features: {features}")
            ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=15)
            dfs.train[features] = ordinal_encoder.fit_transform(dfs.train[features])
            dfs.test[features] = ordinal_encoder.transform(dfs.test[features])
        return dfs

    @log_method_call
    def _scaling(self, dfs: Data) -> Data:
        dfs = self._min_max_scaling(dfs=dfs)
        dfs = self._standard_scaling(dfs=dfs)
        return dfs

    def _min_max_scaling(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.numeric_features() if dfs.schema.numeric[f] == ScalerType.MINMAX_SCALER]
        if features:
            self.logger.info(f"Min max scaling features: {features}")
            scaler = MinMaxScaler()
            dfs.train[features] = scaler.fit_transform(dfs.train[features])
            dfs.test[features] = scaler.transform(dfs.test[features])
        return dfs

    def _standard_scaling(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.numeric_features() if dfs.schema.numeric[f] == ScalerType.STANDARD_SCALER]
        if features:
            self.logger.info(f"Standard scaling features: {features}")
            scaler = StandardScaler()
            dfs.train[features] = scaler.fit_transform(dfs.train[features])
            dfs.test[features] = scaler.transform(dfs.test[features])
        return dfs

    def _save_preprocessed_dataframe(self, dfs: Data) -> None:
        features = dfs.schema.catvar_features() + dfs.schema.numeric_features() + dfs.schema.meta

        dfs.train[features + dfs.schema.labels].to_csv(self.config.paths.train_preprocessed, index=False)
        self.logger.info(f"Processed train dataframe successfully saved at {self.config.paths.train_preprocessed}")

        dfs.test[features].to_csv(self.config.paths.test_preprocessed, index=False)
        self.logger.info(f"Processed test dataframe successfully saved at {self.config.paths.test_preprocessed}")
