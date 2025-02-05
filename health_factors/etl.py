from tools.load import load_from_csv
from tools.task import Data, Task
from tools.logger import log_method_call
from tools.pandas import (
    label_encoder,
    fill_with_mode,
    min_max_scaling,
    ordinal_encoding,
    standard_scaling,
    fill_with_given_value,
)
from tools.schema import Schema, ScalerType, EncodingType
from health_factors._categorizator import (
    AGE_CATEGORIES,
    SLEEP_CATEGORIES,
    NO_YES_CATEGORIES,
    EXERCISE_TYPES_CATEGORIES,
    categorize,
)


class HealthFactorsEtl(Task):
    config_path = "./health_factors/config.yml"

    def run_task(self) -> None:
        dfs = Data(
            train=load_from_csv(self.config.paths.train, logger=self.logger),
            test=load_from_csv(self.config.paths.test, logger=self.logger),
            schema=self._get_schema(),
        )

        dfs = self._features_engineering(dfs=dfs)
        dfs = self._fill_missing_values(dfs=dfs)

        dfs = self._encoding(dfs=dfs)
        dfs = self._scaling(dfs=dfs)

        dfs = self._set_label(dfs=dfs)
        self._save_preprocessed_dataframe(dfs=dfs)
        dfs.schema.save_as_pickle(filepath=self.config.paths.schema, logger=self.logger)

    @log_method_call
    def _get_schema(self) -> Schema:
        return Schema(
            meta=["ID"],
            numeric={
                "Weight_kg": ScalerType.MINMAX_SCALER,
            },
            catvar={
                "Age": EncodingType.LABEL_ENCODING,
                "Hormonal_Imbalance": EncodingType.LABEL_ENCODING,
                "Hyperandrogenism": EncodingType.LABEL_ENCODING,
                "Hirsutism": EncodingType.LABEL_ENCODING,
                "Conception_Difficulty": EncodingType.LABEL_ENCODING,
                "Insulin_Resistance": EncodingType.LABEL_ENCODING,
                "Exercise_Frequency": EncodingType.ORDINAL_ENCODING,
                "Exercise_Type": EncodingType.ORDINAL_ENCODING,
                "Exercise_Duration": EncodingType.ORDINAL_ENCODING,
                "Sleep_Hours": EncodingType.ORDINAL_ENCODING,
                "Exercise_Benefit": EncodingType.ORDINAL_ENCODING,
            },
            labels=["PCOS"],
        )

    @log_method_call
    def _features_engineering(self, dfs: Data) -> Data:
        dfs = self._fix_age_column(dfs=dfs)
        dfs = self._fix_yes_or_no_columns(dfs=dfs)
        dfs = self._add_specific_sport_types(dfs=dfs)
        dfs = self._fix_exercise_types_column(dfs=dfs)
        dfs = self._fix_sleep_hours_columns(dfs=dfs)
        return dfs

    @log_method_call
    def _fix_age_column(self, dfs: Data) -> Data:
        for df in [dfs.train, dfs.test]:
            df["Age"] = df["Age"].apply(
                lambda x: categorize(str(x).lower(), categories=AGE_CATEGORIES),
            )
        return dfs

    @log_method_call
    def _fix_yes_or_no_columns(self, dfs: Data) -> Data:
        for df in [dfs.train, dfs.test]:
            df["Hormonal_Imbalance"] = df["Hormonal_Imbalance"].apply(
                lambda x: categorize(str(x).lower(), categories=NO_YES_CATEGORIES),
            )
            df["Hyperandrogenism"] = df["Hyperandrogenism"].apply(
                lambda x: categorize(str(x).lower(), categories=NO_YES_CATEGORIES),
            )
            df["Hirsutism"] = df["Hirsutism"].apply(
                lambda x: categorize(str(x).lower(), categories=NO_YES_CATEGORIES),
            )
            df["Conception_Difficulty"] = df["Conception_Difficulty"].apply(
                lambda x: categorize(str(x).lower(), categories=NO_YES_CATEGORIES),
            )
            df["Insulin_Resistance"] = df["Insulin_Resistance"].apply(
                lambda x: categorize(str(x).lower(), categories=NO_YES_CATEGORIES),
            )
        return dfs

    @log_method_call
    def _add_specific_sport_types(self, dfs: Data) -> Data:
        for df in [dfs.train, dfs.test]:
            df["cardio"] = "cardio" in df["Exercise_Type"].str.lower()
            df["flexibility"] = "flexibility" in df["Exercise_Type"].str.lower()
            df["balance"] = "balance" in df["Exercise_Type"].str.lower()
            df["strength"] = "strength" in df["Exercise_Type"].str.lower()

        dfs.schema.catvar.update(
            {
                "cardio": EncodingType.LABEL_ENCODING,
                "flexibility": EncodingType.LABEL_ENCODING,
                "balance": EncodingType.LABEL_ENCODING,
                "strength": EncodingType.LABEL_ENCODING,
            }
        )
        return dfs

    @log_method_call
    def _fix_exercise_types_column(self, dfs: Data) -> Data:
        for df in [dfs.train, dfs.test]:
            df["Exercise_Type"] = df["Exercise_Type"].apply(
                lambda x: categorize(str(x).lower(), categories=EXERCISE_TYPES_CATEGORIES),
            )
        return dfs

    @log_method_call
    def _fix_sleep_hours_columns(self, dfs: Data) -> Data:
        for df in [dfs.train, dfs.test]:
            df["Sleep_Hours"] = df["Sleep_Hours"].apply(
                lambda x: categorize(str(x).lower(), categories=SLEEP_CATEGORIES),
            )
        return dfs

    @log_method_call
    def _fill_missing_values(self, dfs: Data) -> Data:
        dfs = fill_with_mode(
            dfs=dfs,
            columns=["Weight_kg"],
            logger=self.logger,
        )

        dfs = fill_with_given_value(
            dfs=dfs,
            columns=[
                "Age",
                "Hormonal_Imbalance",
                "Hyperandrogenism",
                "Hirsutism",
                "Conception_Difficulty",
                "Insulin_Resistance",
                "Exercise_Type",
                "Exercise_Frequency",
                "Exercise_Duration",
                "Sleep_Hours",
                "Exercise_Benefit",
            ],
            value="missing",
            logger=self.logger,
        )

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

    @log_method_call
    def _set_label(self, dfs: Data) -> Data:
        dfs.train[dfs.schema.labels[0]] = dfs.train[dfs.schema.labels[0]].map({"Yes": 1, "No": 0})
        return dfs

    def _save_preprocessed_dataframe(self, dfs: Data) -> None:
        features = dfs.schema.catvar_features() + dfs.schema.numeric_features() + dfs.schema.meta

        dfs.train[features + dfs.schema.labels].to_csv(self.config.paths.train_preprocessed, index=False)
        self.logger.info(f"Processed train dataframe successfully saved at {self.config.paths.train_preprocessed}")

        dfs.test[features].to_csv(self.config.paths.test_preprocessed, index=False)
        self.logger.info(f"Processed test dataframe successfully saved at {self.config.paths.test_preprocessed}")
