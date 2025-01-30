from tools.load import load_from_csv
from tools.task import Data, Task
from tools.logger import log_method_call
from tools.pandas import label_encoder, fill_with_mean, min_max_scaling, ordinal_encoding, standard_scaling
from tools.schema import Schema, ScalerType, EncodingType


class ProductDefectEtl(Task):
    config_path = "./product_defect/config.yml"

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
            meta=["ProductID"],
            numeric={
                "Temperature": ScalerType.MINMAX_SCALER,
                "Pressure": ScalerType.MINMAX_SCALER,
                "AssemblyTime": ScalerType.MINMAX_SCALER,
                "Sensor1Reading": ScalerType.MINMAX_SCALER,
                "Sensor2Reading": ScalerType.MINMAX_SCALER,
                "Sensor3Reading": ScalerType.MINMAX_SCALER,
            },
            catvar={
                "ProductionLine": EncodingType.LABEL_ENCODING,
                "OperatorExperience": EncodingType.ORDINAL_ENCODING,
            },
            labels=["Defective"],
        )

    @log_method_call
    def _fill_missing_values(self, dfs: Data) -> Data:
        dfs = fill_with_mean(
            dfs=dfs, columns=["Temperature", "Pressure", "AssemblyTime", "Sensor1Reading"], logger=self.logger
        )

        dfs.train["OperatorExperience"].fillna(-1, inplace=True)
        dfs.test["OperatorExperience"].fillna(-1, inplace=True)

        return dfs

    @log_method_call
    def _features_engineering(self, dfs: Data) -> Data:
        # Round the experience to the nearest integer, i.e., binarize
        for df in [dfs.train, dfs.test]:
            df["OperatorExperience"] = df["OperatorExperience"].round()
        return dfs

    @log_method_call
    def _encoding(self, dfs: Data) -> Data:
        dfs = ordinal_encoding(dfs=dfs, logger=self.logger)
        dfs = label_encoder(dfs=dfs, logger=self.logger)
        return dfs

    @log_method_call
    def _scaling(self, dfs: Data) -> Data:
        dfs = min_max_scaling(dfs=dfs, logger=self.logger)
        dfs = standard_scaling(dfs=dfs, logger=self.logger)
        return dfs

    def _save_preprocessed_dataframe(self, dfs: Data) -> None:
        features = dfs.schema.catvar_features() + dfs.schema.numeric_features() + dfs.schema.meta

        dfs.train[features + dfs.schema.labels].to_csv(self.config.paths.train_preprocessed, index=False)
        self.logger.info(f"Processed train dataframe successfully saved at {self.config.paths.train_preprocessed}")

        dfs.test[features].to_csv(self.config.paths.test_preprocessed, index=False)
        self.logger.info(f"Processed test dataframe successfully saved at {self.config.paths.test_preprocessed}")
