from typing import List

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler

from tools.load import load_from_csv
from tools.task import Data, Task
from tools.logger import log_method_call
from tools.schema import Schema, ScalerType, EncodingType


class SpaceshipTitanicEtl(Task):
    config_path = "./spaceship_titanic/config.yml"

    def run_task(self) -> None:
        dfs = Data(
            train=load_from_csv(self.config.paths.train, logger=self.logger),
            test=load_from_csv(self.config.paths.test, logger=self.logger),
            schema=self._get_schema(),
        )

        dfs = self._correct_columns(dfs=dfs)
        dfs = self._fill_missing_values(dfs=dfs)
        dfs = self._features_engineering(dfs=dfs)

        dfs = self._encoding(dfs=dfs)
        dfs = self._scaling(dfs=dfs)

        self._save_preprocessed_dataframe(dfs=dfs)
        dfs.schema.save_as_pickle(filepath=self.config.paths.schema, logger=self.logger)

    @log_method_call
    def _fill_missing_values(self, dfs: Data) -> Data:
        dfs.train.HomePlanet.fillna(value="Unknown", inplace=True)
        dfs.test.HomePlanet.fillna(value="Unknown", inplace=True)
        dfs.train.Destination.fillna(value="Unknown", inplace=True)
        dfs.test.Destination.fillna(value="Unknown", inplace=True)
        dfs.train.VIP.fillna(value="Unknown", inplace=True)
        dfs.train.VIP = dfs.train.VIP.astype(str)
        dfs.test.VIP.fillna(value="Unknown", inplace=True)
        dfs.test.VIP = dfs.test.VIP.astype(str)
        return dfs

    def _correct_columns(self, dfs: Data) -> Data:
        dfs = self._correct_expense_columns(dfs=dfs)
        dfs = self._fill_with_mean(dfs=dfs, columns=["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"])
        dfs = self._correct_cryo_sleep_column(dfs=dfs)
        return dfs

    def _correct_cryo_sleep_column(self, dfs: Data) -> Data:
        # If they bought something, then they were not in cryo sleep
        # Select rows where CryoSleep is NaN
        mask = dfs.train.CryoSleep.isna()

        # Update those rows where any of the specified columns are 0
        dfs.train.loc[mask, "CryoSleep"] = (
            (dfs.train.loc[mask, "RoomService"] == 0)
            & (dfs.train.loc[mask, "FoodCourt"] == 0)
            & (dfs.train.loc[mask, "ShoppingMall"] == 0)
            & (dfs.train.loc[mask, "Spa"] == 0)
            & (dfs.train.loc[mask, "VRDeck"] == 0)
        )

        mask = dfs.test.CryoSleep.isna()
        dfs.test.loc[mask, "CryoSleep"] = (
            (dfs.test.loc[mask, "RoomService"] == 0)
            & (dfs.test.loc[mask, "FoodCourt"] == 0)
            & (dfs.test.loc[mask, "ShoppingMall"] == 0)
            & (dfs.test.loc[mask, "Spa"] == 0)
            & (dfs.test.loc[mask, "VRDeck"] == 0)
        )

        return dfs

    @log_method_call
    def _correct_expense_columns(self, dfs: Data) -> Data:
        # Fill with 0 because if they are in cryo sleep, then they cannot buy stuff.
        expense_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        dfs.train[dfs.train.CryoSleep == True][expense_columns].fillna(value="Unknown", inplace=True)
        dfs.test[dfs.test.CryoSleep == True][expense_columns].fillna(value="Unknown", inplace=True)
        return dfs

    @log_method_call
    def _fill_with_mean(self, dfs: Data, columns: List[str]) -> Data:
        mean = dfs.train[columns].mean().to_dict()
        dfs.train[columns] = dfs.train[columns].fillna(mean)
        dfs.test[columns] = dfs.test[columns].fillna(mean)
        return dfs

    @log_method_call
    def _get_schema(self) -> Schema:
        return Schema(
            meta=["PassengerId", "Name"],
            numeric={
                "Age": ScalerType.STANDARD_SCALER,
                "RoomService": ScalerType.STANDARD_SCALER,
                "FoodCourt": ScalerType.STANDARD_SCALER,
                "ShoppingMall": ScalerType.STANDARD_SCALER,
                "Spa": ScalerType.STANDARD_SCALER,
                "VRDeck": ScalerType.STANDARD_SCALER,
            },
            catvar={
                "HomePlanet": EncodingType.LABEL_ENCODING,
                "CryoSleep": EncodingType.ORDINAL_ENCODING,
                "Cabin": EncodingType.LABEL_ENCODING,
                "Destination": EncodingType.ORDINAL_ENCODING,
                "VIP": EncodingType.LABEL_ENCODING,
            },
            labels=["Transported"],
        )

    @log_method_call
    def _features_engineering(self, dfs: Data) -> Data:
        dfs = self._extract_cabin_data(dfs=dfs)
        dfs = self._extract_group_data(dfs=dfs)
        dfs = self._extract_amount_billed(dfs=dfs)
        return dfs

    @log_method_call
    def _extract_cabin_data(self, dfs: Data) -> Data:
        dfs.train[["Deck", "Number", "Side"]] = dfs.train["Cabin"].str.split("/", expand=True)
        dfs.train.drop(columns=["Cabin"], inplace=True)

        dfs.test[["Deck", "Number", "Side"]] = dfs.test["Cabin"].str.split("/", expand=True)
        dfs.test.drop(columns=["Cabin"], inplace=True)

        dfs = self._fill_with_mode(dfs=dfs, columns=["Deck", "Number", "Side"])

        dfs.schema.catvar.update(
            {"Deck": EncodingType.LABEL_ENCODING, "Number": None, "Side": EncodingType.LABEL_ENCODING}
        )
        del dfs.schema.catvar["Cabin"]
        return dfs

    @log_method_call
    def _fill_with_mode(self, dfs: Data, columns: List[str]) -> Data:
        for column in columns:
            mode = dfs.train[column].mode()[0]
            dfs.train[column] = dfs.train[column].fillna(mode)
            dfs.test[column] = dfs.test[column].fillna(mode)
        return dfs

    @log_method_call
    def _extract_group_data(self, dfs: Data) -> Data:
        dfs.train["Group"] = dfs.train["PassengerId"].str.split("_").str[0]
        dfs.test["Group"] = dfs.test["PassengerId"].str.split("_").str[0]
        dfs.schema.catvar.update({"Group": None})

        dfs.train["IdInGroup"] = dfs.train["PassengerId"].str.split("_").str[1]
        dfs.test["IdInGroup"] = dfs.test["PassengerId"].str.split("_").str[1]
        dfs.schema.catvar.update({"IdInGroup": None})

        map_group_size_train = dfs.train.Group.value_counts().to_dict()
        dfs.train["GroupSize"] = dfs.train["Group"].map(map_group_size_train)
        map_group_size_test = dfs.test.Group.value_counts().to_dict()
        dfs.test["GroupSize"] = dfs.test["Group"].map(map_group_size_test)
        dfs.schema.catvar.update({"GroupSize": None})

        return dfs

    @log_method_call
    def _extract_amount_billed(self, dfs: Data) -> Data:
        dfs.train["TotalAmountBilled"] = (
            dfs.train.RoomService + dfs.train.FoodCourt + dfs.train.ShoppingMall + dfs.train.Spa + dfs.train.VRDeck
        )
        dfs.test["TotalAmountBilled"] = (
            dfs.test.RoomService + dfs.test.FoodCourt + dfs.test.ShoppingMall + dfs.test.Spa + dfs.test.VRDeck
        )
        dfs.schema.numeric.update({"TotalAmountBilled": ScalerType.STANDARD_SCALER})
        return dfs

    @log_method_call
    def _encoding(self, dfs: Data) -> Data:
        dfs = self._ordinal_encoding(dfs=dfs)
        dfs = self._label_encoder(dfs=dfs)
        return dfs

    @log_method_call
    def _label_encoder(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.catvar_features() if dfs.schema.catvar[f] == EncodingType.LABEL_ENCODING]
        if features:
            label_encoder = LabelEncoder()
            for feature in features:
                dfs.train[feature] = label_encoder.fit_transform(dfs.train[feature])
                dfs.test[feature] = label_encoder.transform(dfs.test[feature])
        return dfs

    @log_method_call
    def _ordinal_encoding(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.catvar_features() if dfs.schema.catvar[f] == EncodingType.ORDINAL_ENCODING]
        if features:
            ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=15)
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
        if features:
            scaler = MinMaxScaler()
            dfs.train[features] = scaler.fit_transform(dfs.train[features])
            dfs.test[features] = scaler.transform(dfs.test[features])
        return dfs

    @log_method_call
    def _standard_scaling(self, dfs: Data) -> Data:
        features = [f for f in dfs.schema.numeric_features() if dfs.schema.numeric[f] == ScalerType.STANDARD_SCALER]
        if features:
            scaler = StandardScaler()
            dfs.train[features] = scaler.fit_transform(dfs.train[features])
            dfs.test[features] = scaler.transform(dfs.test[features])
        return dfs

    @log_method_call
    def _save_preprocessed_dataframe(self, dfs: Data) -> None:
        features = dfs.schema.catvar_features() + dfs.schema.numeric_features() + dfs.schema.meta

        dfs.train[features + dfs.schema.labels].to_csv(self.config.paths.train_preprocessed, index=False)
        dfs.test[features].to_csv(self.config.paths.test_preprocessed, index=False)
