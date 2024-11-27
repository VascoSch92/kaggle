import pickle
from typing import Optional
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler

from tools.task import Data, Task
from tools.logger import log_method_call
from tools.schema import Schema, ScalerType, EncodingType
from mental_health._maps import DEGREES, CITY_TO_POPULATION, SLEEP_DURATION_MAP


class MentalHealthEtl(Task):
    config_path = "./mental_health/config.yml"

    def run_task(self) -> None:
        dfs = self._load_datasets()
        dfs.schema = self._create_schema()

        dfs = self._complete_the_data(dfs=dfs)
        dfs = self._fill_missing_values(dfs=dfs)
        dfs = self._features_engineering(dfs=dfs)
        dfs = self._encoding(dfs=dfs)
        dfs = self._scaling(dfs=dfs)

        self._save_preprocessed_dataframes(dfs=dfs)
        self._save_schema(dfs=dfs)

    @log_method_call
    def _load_datasets(self) -> Data:
        df_train = pd.read_csv(self.config.paths.train)
        df_test = pd.read_csv(self.config.paths.test)

        self.logger.info(f"Dataframe df_train: {len(df_train)} rows x {len(df_train.columns)} columns.")
        self.logger.info(f"Dataframe df_test: {len(df_test)} rows x {len(df_test.columns)} columns.")

        return Data(train=df_train, test=df_test)

    @log_method_call
    def _create_schema(self) -> Schema:
        return Schema(
            meta=["id", "Name"],
            numeric={
                "Age": ScalerType.MINMAX_SCALER,
                "Academic Pressure": None,
                "Work Pressure": None,
                "Study Satisfaction": None,
                "Job Satisfaction": None,
                "CGPA": ScalerType.MINMAX_SCALER,
                "Work/Study Hours": ScalerType.MINMAX_SCALER,
                "Financial Stress": None,
            },
            catvar={
                "Gender": EncodingType.LABEL_ENCODING,
                "City": EncodingType.ORDINAL_ENCODING,
                "Working Professional or Student": EncodingType.LABEL_ENCODING,
                "Profession": EncodingType.ORDINAL_ENCODING,
                "Sleep Duration": None,
                "Dietary Habits": EncodingType.ORDINAL_ENCODING,
                "Degree": EncodingType.ORDINAL_ENCODING,
                "Have you ever had suicidal thoughts ?": EncodingType.LABEL_ENCODING,
                "Family History of Mental Illness": EncodingType.LABEL_ENCODING,
            },
            labels=["Depression"],
        )

    @log_method_call
    def _complete_the_data(self, dfs: Data) -> Data:
        """See notebook for details."""
        dfs = self._correct_misspelled_cities_names(dfs=dfs)
        dfs = self._complete_profession_column(dfs=dfs)
        dfs = self._correct_profession_column(dfs=dfs)
        dfs = self._complete_pressure_column(dfs=dfs)
        dfs = self._complete_satisfaction_column(dfs=dfs)
        dfs = self._complete_cgpa_column(dfs=dfs)
        dfs = self._clean_dietary_habits(dfs=dfs)

        return dfs

    @log_method_call
    def _correct_misspelled_cities_names(self, dfs: Data) -> Data:
        def correct(city: str) -> str:
            if city in {"Golkata", "Molkata", "Tolkata", "Rolkata"}:
                return "Kolkata"
            if city in {"Ghopal", "Mhopal"}:
                return "Bhopal"
            if city == "Khaziabad":
                return "Ghaziabad"
            return city

        dfs.train["City"] = dfs.train["City"].apply(lambda c: correct(c))
        dfs.test["City"] = dfs.test["City"].apply(lambda c: correct(c))
        return dfs

    @log_method_call
    def _complete_profession_column(self, dfs: Data) -> Data:
        dfs.train["Profession"] = dfs.train["Profession"].fillna(dfs.train["Working Professional or Student"])
        dfs.test["Profession"] = dfs.test["Profession"].fillna(dfs.test["Working Professional or Student"])
        return dfs

    @log_method_call
    def _correct_profession_column(self, dfs: Data) -> Data:
        def correct(profession: str) -> str:
            if profession in {"Finanancial Analyst"}:
                return "Financial Analyst"
            if profession in {"City Manager"}:
                return "Manager"
            if profession in {"Family Consultant", "City Consultant"}:
                return "Consultant"
            if profession in {"MBA", "BCA"}:
                return "Business Analyst"
            if profession in {"Dev"}:
                return "Software Engineer"
            if profession in {"M.Pharm"}:
                return "Pharmacist"
            if profession in {"Doctor", "Medical Doctor"}:
                return "Doctor"
            if profession in {"Civil Engineer", "Mechanical Engineer"}:
                return "Engineer"
            if profession in {"UX/UI Designer"}:
                return "Graphic Designer"
            if profession in {"Educational Consultant"}:
                return "Consultant"
            if profession in {"B.Ed", "M.Ed"}:
                return "Teacher"
            if profession in {"Chemist"}:
                return "Pharmacist"
            if profession in {"Judge", "LLM"}:
                return "Lawyer"
            if profession in {"MCA", "ME", "M.Tech", "BE"}:
                return "Engineer"
            if profession in {"PhD", "Research Analyst"}:
                return "Researcher"
            if profession in {"Unemployed"}:
                return "Unemployed"
            if profession in {"Student"}:
                return "Student"
            return profession

        dfs.train["Profession"] = dfs.train["Profession"].apply(lambda c: correct(c))
        dfs.test["Profession"] = dfs.test["Profession"].apply(lambda c: correct(c))
        return dfs

    @log_method_call
    def _complete_satisfaction_column(self, dfs: Data) -> Data:
        dfs.train["Satisfaction"] = dfs.train["Study Satisfaction"].fillna(dfs.train["Job Satisfaction"])
        dfs.test["Satisfaction"] = dfs.test["Study Satisfaction"].fillna(dfs.test["Job Satisfaction"])

        dfs.schema.numeric.update({"Satisfaction": None})
        del dfs.schema.numeric["Study Satisfaction"]
        del dfs.schema.numeric["Job Satisfaction"]

        mean = dfs.train["Satisfaction"].mean()
        dfs.train["Satisfaction"] = dfs.train["Satisfaction"].fillna(mean)
        dfs.test["Satisfaction"] = dfs.test["Satisfaction"].fillna(mean)

        return dfs

    @log_method_call
    def _complete_pressure_column(self, dfs: Data) -> Data:
        dfs.train["Pressure"] = dfs.train["Academic Pressure"].fillna(dfs.train["Work Pressure"])
        dfs.test["Pressure"] = dfs.test["Academic Pressure"].fillna(dfs.test["Work Pressure"])

        dfs.schema.numeric.update({"Pressure": None})
        del dfs.schema.numeric["Academic Pressure"]
        del dfs.schema.numeric["Work Pressure"]

        mean = dfs.train["Pressure"].mean()
        dfs.train["Pressure"] = dfs.train["Pressure"].fillna(mean)
        dfs.test["Pressure"] = dfs.test["Pressure"].fillna(mean)

        return dfs

    @log_method_call
    def _complete_cgpa_column(self, dfs: Data) -> Data:
        dfs.train["CGPA"] = dfs.train["CGPA"].fillna(0)
        dfs.test["CGPA"] = dfs.test["CGPA"].fillna(0)
        return dfs

    @log_method_call
    def _clean_dietary_habits(self, dfs: Data) -> Data:
        def dietary_habits(diet: str) -> str:
            diet = str(diet).lower()
            if "unhealthy" in diet:
                return "unhealthy"
            elif "healthy" in diet:
                return "healthy"
            else:
                return "moderate"

        dfs.train["Dietary Habits"] = dfs.train["Dietary Habits"].apply(lambda x: dietary_habits(x))
        dfs.test["Dietary Habits"] = dfs.test["Dietary Habits"].apply(lambda x: dietary_habits(x))

        return dfs

    @log_method_call
    def _fill_missing_values(self, dfs: Data) -> Data:
        mean = dfs.train["Financial Stress"].mean()
        dfs.train["Financial Stress"] = dfs.train["Financial Stress"].fillna(mean)
        dfs.test["Financial Stress"] = dfs.test["Financial Stress"].fillna(mean)
        return dfs

    @log_method_call
    def _features_engineering(self, dfs: Data) -> Data:
        dfs = self._binarize_degree_column(dfs=dfs)
        dfs = self._add_city_population(dfs=dfs)
        dfs = self._add_city_size(dfs=dfs)
        dfs = self._add_sleep_hours(dfs=dfs)
        dfs = self._add_stress(dfs=dfs)
        return dfs

    @log_method_call
    def _binarize_degree_column(self, dfs: Data) -> Data:
        def categorize_degree(degree: str) -> Optional[str]:
            degree = str(degree).replace(".", "").replace(" ", "").replace("_", "").replace("-", "").upper()
            if degree in DEGREES["bachelor"]:
                return "bachelor"
            elif degree in DEGREES["master"]:
                return "master"
            elif degree in DEGREES["doctorate"]:
                return "doctorate"
            elif degree in DEGREES["high_school"]:
                return "high_school"
            else:
                return None

        dfs.train["Degree"] = dfs.train["Degree"].apply(lambda x: categorize_degree(degree=x))
        dfs.test["Degree"] = dfs.test["Degree"].apply(lambda x: categorize_degree(degree=x))

        mode = dfs.train["Degree"].mode()[0]
        dfs.train["Degree"] = dfs.train["Degree"].fillna(mode)
        dfs.test["Degree"] = dfs.test["Degree"].fillna(mode)
        return dfs

    @log_method_call
    def _add_city_population(self, dfs: Data) -> Data:
        dfs.train["City Population"] = dfs.train.City.map(CITY_TO_POPULATION)
        dfs.test["City Population"] = dfs.test.City.map(CITY_TO_POPULATION)
        dfs.schema.numeric.update({"City Population": ScalerType.STANDARD_SCALER})

        mean = dfs.train["City Population"].mean()
        dfs.train["City Population"] = dfs.train["City Population"].fillna(mean)
        dfs.test["City Population"] = dfs.test["City Population"].fillna(mean)
        return dfs

    @log_method_call
    def _add_city_size(self, dfs: Data) -> Data:
        bins = [0, 1000000, 3000000, 10000000, float("inf")]
        labels = ["Small", "Medium", "Large", "Mega"]

        dfs.train["City Size"] = pd.cut(dfs.train["City Population"], bins=bins, labels=labels)
        dfs.test["City Size"] = pd.cut(dfs.test["City Population"], bins=bins, labels=labels)
        dfs.schema.catvar.update({"City Size": EncodingType.ORDINAL_ENCODING})
        return dfs

    @log_method_call
    def _add_sleep_hours(self, dfs: Data) -> Data:
        dfs.train["Sleep Hours"] = dfs.train["Sleep Duration"].map(SLEEP_DURATION_MAP)
        dfs.test["Sleep Hours"] = dfs.test["Sleep Duration"].map(SLEEP_DURATION_MAP)
        dfs.schema.numeric.update({"Sleep Hours": ScalerType.MINMAX_SCALER})
        del dfs.schema.catvar["Sleep Duration"]

        mean = dfs.train["Sleep Hours"].mean()
        dfs.train["Sleep Hours"] = dfs.train["Sleep Hours"].fillna(mean)
        dfs.test["Sleep Hours"] = dfs.test["Sleep Hours"].fillna(mean)

        return dfs

    @log_method_call
    def _add_stress(self, dfs: Data) -> Data:
        dfs.train["Stress"] = (
            dfs.train["Work/Study Hours"]
            + dfs.train["Pressure"]
            + dfs.train["Financial Stress"]
            - dfs.train["Satisfaction"]
            + 3
        )
        dfs.test["Stress"] = (
            dfs.test["Work/Study Hours"]
            + dfs.test["Pressure"]
            + dfs.test["Financial Stress"]
            - dfs.test["Satisfaction"]
            + 3
        )
        dfs.schema.numeric.update({"Stress": ScalerType.MINMAX_SCALER})
        return dfs

    @log_method_call
    def _encoding(self, dfs: Data) -> Data:
        dfs = self._ordinal_encoding(dfs=dfs)
        dfs = self._label_encoder(dfs=dfs)
        return dfs

    @log_method_call
    def _label_encoder(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.catvar_features() if dfs.schema.catvar[f] == EncodingType.LABEL_ENCODING]
        label_encoder = LabelEncoder()
        for feature in features:
            dfs.train[feature] = label_encoder.fit_transform(dfs.train[feature])
            dfs.test[feature] = label_encoder.transform(dfs.test[feature])
        return dfs

    @log_method_call
    def _ordinal_encoding(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.catvar_features() if dfs.schema.catvar[f] == EncodingType.ORDINAL_ENCODING]
        ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        dfs.train[features] = ordinal_encoder.fit_transform(dfs.train[features])
        dfs.test[features] = ordinal_encoder.transform(dfs.test[features])
        return dfs

    @log_method_call
    def _scaling(self, dfs: Data) -> Data:
        dfs = self._min_max_scaling(dfs=dfs)
        dfs = self._standard_scaling(dfs=dfs)
        return dfs

    @log_method_call
    def _min_max_scaling(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.numeric_features() if dfs.schema.numeric[f] == ScalerType.MINMAX_SCALER]
        scaler = MinMaxScaler()
        dfs.train[features] = scaler.fit_transform(dfs.train[features])
        dfs.test[features] = scaler.transform(dfs.test[features])
        return dfs

    @log_method_call
    def _standard_scaling(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.numeric_features() if dfs.schema.numeric[f] == ScalerType.STANDARD_SCALER]
        scaler = StandardScaler()
        dfs.train[features] = scaler.fit_transform(dfs.train[features])
        dfs.test[features] = scaler.transform(dfs.test[features])
        return dfs

    @log_method_call
    def _save_preprocessed_dataframes(self, dfs: Data) -> None:
        features = dfs.schema.meta + dfs.schema.catvar_features() + dfs.schema.numeric_features()
        dfs.train[features + dfs.schema.labels].to_csv(self.config.paths.train_preprocessed, index=False)
        dfs.test[features].to_csv(self.config.paths.test_preprocessed, index=False)

    @log_method_call
    def _save_schema(self, dfs: Data) -> None:
        with Path(self.config.paths.schema).open("wb") as f:
            pickle.dump(dfs.schema, f)
