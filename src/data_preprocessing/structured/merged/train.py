import os.path as osp

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from settings.settings import MODELS_DIR
from src.data_preprocessing.structured.basic_preprocessing import load_data
from src.data_preprocessing.structured.merged.merged import filter_by_feature_importance
from src.data_preprocessing.structured.ml_classifier import MLClassifier
from src.data_transformation.dataset_separation import separate_datasets
from src.data_transformation.imputation import get_imputer
from src.data_transformation.outlier_removal import remove_outliers


MODELS = [
    # 'bagging',
    # 'extraTrees',
    # 'randomForest',
    # 'naiveBayes',
    # 'logisticRegression',
    # 'svm',
    # 'xgboost',
    # 'lda',
    'mlp'
]

MERGED_MODELS_DIR = osp.join(MODELS_DIR, 'merged')


def preprocess_and_separate_dataset(df: DataFrame):
    features_to_keep = ['IRR', 'DYSILL', 'HACHIN', 'BILLS', 'SMOKYRS', 'EVENTS', 'PAYATTN', 'ANX', 'PARK', 'NPIQINF',
                        'GDS', 'TRAVEL', 'CVDIMAG', 'NORMAL', 'PACKSPER', 'DEL', 'APA', 'DEP', 'HYPERCHO', 'REMDATES',
                        'MEALPREP', 'STOVE', 'STROKCOG', 'NORMCOG', 'COGOTH', 'GAMES', 'MOT', 'TAXES', 'DISN',
                        'QUITSMOK', 'MEDS', 'DEP2YRS', 'DEPOTHR', 'DEPD', 'NITE', 'CVDCOG', 'AGIT', 'HYPERTEN',
                        'HXHYPER', 'SHOPPING', 'dx1']

    df = df.loc[:, df.columns.isin(features_to_keep)]


    # split into train and test datasets
    train_df, test_df = train_test_split(df, test_size=.2)

    # create numpy arrays
    feature_cols = list(train_df.columns)
    feature_cols.remove('dx1')
    train_features = train_df[feature_cols].values
    train_labels = np.ravel(train_df['dx1'])
    test_features = test_df[feature_cols].values
    test_labels = np.ravel(test_df['dx1'])

    # fill na
    # df.fillna(df.mean(), inplace=True)
    imputer = get_imputer('KNN')
    train_features = imputer.fit_transform(train_features)
    test_features = imputer.fit_transform(test_features)

    # perform normalization on feature values for train and test
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # features_to_keep = filter_by_feature_importance(threshold=0.005, dataset=train_df,
    #                                                 X_train=train_features, y_train=train_labels)

    train_features, train_labels = remove_outliers(model_name='isf', X=train_features, y=train_labels)

    # after removing outliers return data to original values
    train_features = scaler.inverse_transform(train_features)
    train_df = pd.DataFrame(train_features, columns=feature_cols)
    train_df['dx1'] = pd.Series(train_labels)

    # split data into healthy and diagnosed parts
    healthy_df_train, diagnosed_df_train = separate_datasets(train_df, 'dx1')
    healthy_df_test, diagnosed_df_test = separate_datasets(test_df, 'dx1')

    healthy_df_train.reset_index(inplace=True)
    healthy_df_test.reset_index(inplace=True)
    diagnosed_df_train.reset_index(inplace=True)
    diagnosed_df_test.reset_index(inplace=True)

    return healthy_df_train, healthy_df_test, diagnosed_df_train, diagnosed_df_test


def load_merged_data():
    merged = load_data()['merged']
    merged.dropna(axis="rows", subset=["dx1"], inplace=True)
    merged.drop(labels=["ID", "Subject"], axis="columns", inplace=True)
    return merged


def train(model_name, X, y):
    model = MLClassifier(model_name=model_name,
                         model_dir=MERGED_MODELS_DIR,
                         load=False)
    # model.train(X, y)
    model.load()
    model.evaluate(X, y)


if __name__ == '__main__':

    df = load_merged_data()
    healthy_df_train, healthy_df_test, diagnosed_df_train, diagnosed_df_test = preprocess_and_separate_dataset(df)

    X = np.array(healthy_df_train.drop('dx1', axis='columns'))
    y = np.ravel(healthy_df_train['dx1'])

    for model_name in MODELS:
        train(model_name, X, y)
