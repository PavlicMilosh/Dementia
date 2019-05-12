import os
import os.path as osp

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from settings.settings import MODELS_DIR, DATA_VISUALISATION_DIR
from src.data_preprocessing.structured.basic_preprocessing import load_data
from src.data_preprocessing.structured.ml_classifier import MLClassifier
from src.data_transformation.dataset_separation import separate_datasets
from src.data_transformation.imputation import get_imputer
from src.data_transformation.sampling import get_oversampler

MODELS = [
    'bagging',
    'extraTrees',
    'randomForest',
    # 'naiveBayes',
    # 'logisticRegression',
    'svm',
    'xgboost',
    # 'lda',
    # 'mlp'
]

MERGED_MODELS_DIR = osp.join(MODELS_DIR, 'merged')


def visualize_data(x, y, dims=2, name='data.csv'):
    kpca = KernelPCA(n_components=dims, kernel='rbf', fit_inverse_transform=True, gamma=10)
    X_kpca = kpca.fit_transform(x)

    df = DataFrame()
    df['x_pca_0'] = pd.Series(X_kpca[:, 0])
    df['x_pca_1'] = pd.Series(X_kpca[:, 1])
    if dims == 3:
        df['x_pca_2'] = pd.Series(X_kpca[:, 2])
    df['y'] = pd.Series(y)

    if not os.path.exists(DATA_VISUALISATION_DIR):
        os.makedirs(DATA_VISUALISATION_DIR)
    df.to_csv(osp.join(DATA_VISUALISATION_DIR, name))

    d = {
        'Alzheimer Dementia': 0,
        'Non AD Dementia': 1,
        'Uncertain Dementia': 2,
        'Cognitively Normal': 3
    }

    # # add jitter in rapidminer
    #
    # if dimensions == 2:
    #     plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
    # else:
    #     plt.scatter(X_kpca[:, 0], X_kpca[:, 1], X_kpca[:, 2], c=y)
    #
    # plt.title('pca')
    # plt.colorbar()
    # plt.show()
    print()


def preprocess(df: DataFrame, separate=False):
    # find features to keep based on their importance
    # features_to_keep = filter_by_feature_importance(threshold=0.005, dataset=train_df,
    # X_train=train_features, y_train=train_labels)
    # features_to_keep = ['IRR', 'DYSILL', 'HACHIN', 'BILLS', 'SMOKYRS', 'EVENTS', 'PAYATTN', 'ANX', 'PARK', 'NPIQINF',
    #                     'GDS', 'TRAVEL', 'CVDIMAG', 'NORMAL', 'PACKSPER', 'DEL', 'APA', 'DEP', 'HYPERCHO', 'REMDATES',
    #                     'MEALPREP', 'STOVE', 'STROKCOG', 'NORMCOG', 'COGOTH', 'GAMES', 'MOT', 'TAXES', 'DISN',
    #                     'QUITSMOK', 'MEDS', 'DEP2YRS', 'DEPOTHR', 'DEPD', 'NITE', 'CVDCOG', 'AGIT', 'HYPERTEN',
    #                     'HXHYPER', 'SHOPPING', 'dx1']
    # df = df.loc[:, df.columns.isin(features_to_keep)]

    # split into train and test datasets
    train_df, test_df = train_test_split(df, test_size=.2)

    # create numpy arrays
    feature_cols = list(train_df.columns)
    feature_cols.remove('dx1')
    train_features = train_df[feature_cols].values
    train_labels = np.ravel(train_df['dx1'])
    test_features = test_df[feature_cols].values
    test_labels = np.ravel(test_df['dx1'])

    # fill missing values using KNN
    imputer = get_imputer('KNN')
    train_features = imputer.fit_transform(train_features)
    test_features = imputer.fit_transform(test_features)

    # perform normalization on feature values for train and test
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # visualize_data(train_features, train_labels, dims=2, name='original-2d.csv')
    # visualize_data(train_features, train_labels, dims=3, name='original-3d.csv')

    # oversample train dataset
    sampler = get_oversampler('adasyn')
    train_features, train_labels = sampler.fit_resample(train_features, train_labels)

    # visualize_data(train_features, train_labels, dims=2, name='oversampled-2d.csv')
    # visualize_data(train_features, train_labels, dims=3, name='oversampled-3d.csv')

    # remove outliers
    # train_features, train_labels = remove_outliers(model_name='isf', X=train_features, y=train_labels)

    # after removing outliers return train and test data to original values
    # train_features = scaler.inverse_transform(train_features)
    train_df = pd.DataFrame(train_features, columns=feature_cols)
    train_df['dx1'] = pd.Series(train_labels)

    test_features = scaler.inverse_transform(test_features)
    test_df = pd.DataFrame(test_features, columns=feature_cols)
    test_df['dx1'] = pd.Series(test_labels)

    # features_to_keep = filter_by_feature_importance(threshold=0.005, dataset=train_df)

    if separate:
        # split data into healthy and diagnosed parts
        healthy_df_train, diagnosed_df_train = separate_datasets(train_df, 'dx1')
        healthy_df_test, diagnosed_df_test = separate_datasets(test_df, 'dx1')

        # reset indexes
        healthy_df_train.reset_index(inplace=True)
        healthy_df_test.reset_index(inplace=True)
        diagnosed_df_train.reset_index(inplace=True)
        diagnosed_df_test.reset_index(inplace=True)

        return healthy_df_train, healthy_df_test, diagnosed_df_train, diagnosed_df_test

    return train_df, test_df


def load_merged_data():
    merged = load_data()['merged']
    merged.dropna(axis="rows", subset=["dx1"], inplace=True)
    merged.drop(labels=["ID", "Subject"], axis="columns", inplace=True)
    return merged


def train(model_name, X, y, X_test, y_test):
    model = MLClassifier(model_name=model_name,
                         model_dir=MERGED_MODELS_DIR,
                         load=False)
    model.train(X, y)
    print('=' * 10 + '\n\n\n')
    print("{} finished training".format(model_name))

    print("\nMODEL\n")
    model.print_model()

    print("\nTRAIN RESULTS\n")
    model.evaluate(X, y)

    print("\nTEST RESULTS\n")
    model.evaluate(X_test, y_test)

    print('=' * 10 + '\n\n\n')


if __name__ == '__main__':

    df = load_merged_data()
    # healthy_df_train, healthy_df_test, diagnosed_df_train, diagnosed_df_test = preprocess(df, separate=True)
    #
    # X = np.array(healthy_df_train.drop('dx1', axis='columns'))
    # y = np.ravel(healthy_df_train['dx1'])
    # X_test = np.array(healthy_df_test.drop('dx1', axis='columns'))
    # y_test = np.ravel(healthy_df_test['dx1'])

    train_df, test_df = preprocess(df)
    X = np.array(train_df.drop('dx1', axis='columns'))
    y = np.ravel(train_df['dx1'])
    X_test = np.array(test_df.drop('dx1', axis='columns'))
    y_test = np.ravel(test_df['dx1'])

    for model_name in MODELS:
        train(model_name, X, y, X_test, y_test)
