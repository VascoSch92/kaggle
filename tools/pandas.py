from typing import List
from logging import Logger

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler

from tools.task import Data
from tools.schema import ScalerType, EncodingType


def fill_with_mode(dfs: Data, columns: List[str], logger: Logger) -> Data:
    logger.info(f"Fill with mode features: {columns}")
    for column in columns:
        mode = dfs.train[column].mode()[0]
        dfs.train[column] = dfs.train[column].fillna(mode)
        dfs.test[column] = dfs.test[column].fillna(mode)
    return dfs


def fill_with_mean(dfs: Data, columns: List[str], logger: Logger) -> Data:
    logger.info(f"Fill with mean features: {columns}")
    mean = dfs.train[columns].mean().to_dict()
    dfs.train[columns] = dfs.train[columns].fillna(mean)
    dfs.test[columns] = dfs.test[columns].fillna(mean)
    return dfs


def label_encoder(dfs: Data, logger: Logger) -> Data:
    features = [f for f in dfs.schema.catvar_features() if dfs.schema.catvar[f] == EncodingType.LABEL_ENCODING]
    if features:
        logger.info(f"Label encoding features: {features}")
        label_encoder_transformer = LabelEncoder()
        for feature in features:
            dfs.train[feature] = label_encoder_transformer.fit_transform(dfs.train[feature])
            dfs.test[feature] = label_encoder_transformer.transform(dfs.test[feature])
    return dfs


def ordinal_encoding(dfs: Data, logger: Logger) -> Data:
    features = [f for f in dfs.schema.catvar_features() if dfs.schema.catvar[f] == EncodingType.ORDINAL_ENCODING]
    if features:
        logger.info(f"Ordinal encoding features: {features}")
        ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=15)
        dfs.train[features] = ordinal_encoder.fit_transform(dfs.train[features])
        dfs.test[features] = ordinal_encoder.transform(dfs.test[features])
    return dfs


def min_max_scaling(dfs: Data, logger: Logger) -> Data:
    features = [f for f in dfs.schema.numeric_features() if dfs.schema.numeric[f] == ScalerType.MINMAX_SCALER]
    if features:
        logger.info(f"Min max scaling features: {features}")
        scaler = MinMaxScaler()
        dfs.train[features] = scaler.fit_transform(dfs.train[features])
        dfs.test[features] = scaler.transform(dfs.test[features])
    return dfs


def standard_scaling(dfs: Data, logger: Logger) -> Data:
    features = [f for f in dfs.schema.numeric_features() if dfs.schema.numeric[f] == ScalerType.STANDARD_SCALER]
    if features:
        logger.info(f"Standard scaling features: {features}")
        scaler = StandardScaler()
        dfs.train[features] = scaler.fit_transform(dfs.train[features])
        dfs.test[features] = scaler.transform(dfs.test[features])
    return dfs
