import itertools

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import xgboost as xgb
import random

from settings.settings import RANDOM_STATE
from src.data_preprocessing.structured.basic_preprocessing import remove_others, load_data
from src.data_transformation.dataset_separation import separate_datasets


def preprocess_merged_remove_rows(df):
    return df.dropna()


def preprocess_merged_mean_imputing(X):
    imputer = SimpleImputer()
    imputer.fit(X)
    imputer.transform(X)


def encode(dataset, y_label):
    encoder = LabelEncoder()
    dataset[y_label] = encoder.fit_transform(dataset[y_label])


def split_and_encode(dataset, y_label):
    # Get X and y parts of the dataset
    y = dataset[y_label]
    X = dataset.drop(labels=y_label, axis="columns")

    # Encode target variable
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)

    return X, y


def encode(dataset, y_col):
    # Encode target variable
    encoder = LabelEncoder()
    dataset[y_col] = encoder.fit_transform(dataset[y_col])

    return dataset


def filter_by_feature_importance(threshold, dataset, seed=14, X_train=None, y_train=None):
    # Feature importance
    forest = ExtraTreesClassifier(n_estimators=500, random_state=seed)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    feature_importances = pd.DataFrame(forest.feature_importances_,
                                       index=dataset.drop(['index', 'dx1'], axis='columns').columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking

    # print("Feature ranking:")
    # for f in range(X.shape[1]):
    #     print("%d. feature %s (%f)" % (f + 1, list(dataset)[indices[f]], importances[indices[f]]))
    #
    # # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(X_train.shape[1]), importances[indices],
    #         color="r", yerr=std[indices], align="center")
    # plt.xticks(range(X_train.shape[1]), indices)
    # plt.xlim([-1, X_train.shape[1]])
    # plt.show()

    features_to_keep = set(itertools.compress(list(dataset), [i > threshold for i in importances]))
    features_to_keep.add('dx1')

    if dataset is None:
        return features_to_keep
    else:
        return remove_others(dataset, features_to_keep), features_to_keep


def isolation_forest_outlier_removal(X, y, seed,
                                     n_estimators=150, max_samples=0.8, max_features=0.8,
                                     contamination="auto"):

    clf = IsolationForest(n_estimators=n_estimators,
                          max_samples=max_samples,
                          contamination=contamination,
                          max_features=max_features,
                          behaviour="new",
                          random_state=seed,
                          n_jobs=-1)

    results = clf.fit_predict(X.values)

    outliers = 0
    for r in results:
        if r == -1:
            outliers += 1

    removing_indices = [i for i in range(0, len(results)) if results[i] == -1]
    X_train_new = X.drop(X.index[removing_indices])
    y_train_new = [y[yi] for yi in range(0, len(y)) if results[yi] == 1]

    return X_train_new, y_train_new


def bagging_cv(X_train, y_train, seed, verbose=3):

    # Results:
    #                       DEFAULT      Z-SCORE      OUTLIERS
    # n_estimators          250          150          150
    # warm_start            True         True         True
    # max_samples           0.6          0.6          0.6
    # --------------------------------------------------------
    # f1-micro              0.9220       0.9268       0.9403

    clf = BaggingClassifier(n_estimators=140, random_state=seed)

    params = {
        'n_estimators': list(range(100, 1500, 50)),
        'warm_start': [True, False],
        'max_samples': [0.6, 0.8, 1.0]
    }

    gCV = GridSearchCV(estimator=clf,
                       param_grid=params,
                       scoring='f1_micro',
                       n_jobs=-1,
                       refit=True,
                       cv=3,
                       verbose=verbose,
                       return_train_score='warn')

    return gCV.fit(X_train.values, y_train)


def random_forest_cv(X_train, y_train, seed, verbose=3):

    # Results:
    #                       DEFAULT      Z-SCORE      OUTLIERS
    # n_estimators          300          850          350
    # criterion             gini         gini         gini
    # min_samples_split     4            2            14
    # max_depth             18           12           10
    # --------------------------------------------------------
    # f1-micro              0.9285       0.9293       0.9452

    clf = RandomForestClassifier(n_estimators=140, random_state=seed)

    params = {
        'n_estimators': list(range(100, 1500, 50)),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': list(range(2, 20, 2)),
        'max_depth': list(range(2, 20, 2))
    }

    gCV = GridSearchCV(estimator=clf,
                       param_grid=params,
                       scoring='f1_micro',
                       n_jobs=-1,
                       refit=True,
                       cv=3,
                       verbose=verbose,
                       return_train_score='warn')

    return gCV.fit(X_train.values, y_train)


def extra_trees_cv(X_train, y_train, seed, verbose=3):

    # Results:
    #                       DEFAULT      Z-SCORE      OUTLIERS
    # n_estimators          1000         850          350
    # min_samples_split     6            2            14
    # max_depth             12           12           10
    # criterion             gini         gini         gini
    # --------------------------------------------------------
    # f1-micro              0.9244       0.9293       0.9452

    clf = ExtraTreesClassifier(n_estimators=140, random_state=seed)

    params = {
        'n_estimators': list(range(100, 1500, 50)),
        'criterion': ['gini', 'entropy'],
        'min_samples_split': list(range(2, 20, 2)),
        'max_depth': list(range(2, 20, 2))
    }

    gCV = GridSearchCV(estimator=clf,
                       param_grid=params,
                       scoring='f1_micro',
                       n_jobs=-1,
                       refit=True,
                       cv=3,
                       verbose=verbose,
                       return_train_score='warn')

    return gCV.fit(X_train.values, y_train)


def gradient_boosting_cv(X_train, y_train, seed, verbose=3):

    # Results:
    #                       DEFAULT      Z-SCORE      OUTLIERS
    # n_estimators          140                       100
    # min_samples_split     2                         14
    # max_depth             2                         18
    # --------------------------------------------------------
    # f1-micro              0.9204                    0.9439

    clf = GradientBoostingClassifier(n_estimators=140,
                                     min_samples_split=2,
                                     max_depth=2,
                                     random_state=seed)

    params = {
        'n_estimators': list(range(100, 1000, 50)),
        'min_samples_split': list(range(2, 20, 2)),
        'max_depth': list(range(2, 20, 2))
    }

    gCV = GridSearchCV(estimator=clf,
                       param_grid=params,
                       scoring='f1_micro',
                       n_jobs=-1,
                       refit=True,
                       cv=3,
                       verbose=verbose,
                       return_train_score='warn')

    return gCV.fit(X_train.values, y_train)


def xgboost_cv(X_train, y_train, seed, verbose=3):

    # Results:
    #                       DEFAULT      Z-SCORE      OUTLIERS
    # learning_rate         0.1                       .1
    # max_depth             16                        10
    # min_child_weight      1                         5
    # subsample             0.8                       .8
    # n_estimators          100                       100
    # --------------------------------------------------------
    # f1-micro              0.93                      0.9477

    # merged = encode(merged, 'dx1')
    # train, test = train_test_split(merged, test_size=0.2)

    clf = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

    params = {'nthread': [4],
              'learning_rate': [0.1],
              'max_depth': list(range(2, 22, 2)),
              'min_child_weight': list(range(1, 6)),
              'subsample': [0.8, 1],
              'n_estimators': list(range(100, 1000, 50)),
              'seed': [seed]}

    gCV = GridSearchCV(estimator=clf,
                       scoring='f1_micro',
                       param_grid=params,
                       n_jobs=-1,
                       refit=True,
                       cv=3,
                       verbose=verbose,
                       return_train_score='warn')

    return gCV.fit(X_train, y_train)


def separated_datasets():
    merged = load_data()['merged']
    merged.dropna(axis="rows", subset=["dx1"], inplace=True)
    merged.drop(labels=["ID", "Subject"], axis="columns", inplace=True)

    merged = encode(merged, 'dx1')

    train, test = train_test_split(merged, test_size=.2)

    x_cols = list(train.columns)
    x_cols.remove('dx1')

    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train[x_cols].values)
    scaled_test_features = scaler.transform(test[x_cols].values)

    train[x_cols] = pd.DataFrame(scaled_train_features, index=train[x_cols].index, columns=train[x_cols].columns)
    test[x_cols] = pd.DataFrame(scaled_test_features, index=test[x_cols].index, columns=test[x_cols].columns)

    merged, features = filter_by_feature_importance(threshold=0.005,
                                                    dataset=merged,
                                                    X_train=train.drop('dx1', axis=1),
                                                    y_train=train['dx1'])

    train = train[features]
    test = test[features]

    X_train, y_train = isolation_forest_outlier_removal(train.drop('dx1', axis=1), train['dx1'], RANDOM_STATE)

    healthy_df_train, diagnosed_df_train = separate_datasets(X_train, y_train)
    healthy_df_test, diagnosed_df_train = separate_datasets(test.drop('dx1', axis=1), test['dx1'])


def main():

    seed = 14
    random.seed(seed)

    merged = load_data()["merged"]
    merged.dropna(axis="rows", subset=["dx1"], inplace=True)
    merged.drop(labels=["ID", "Subject"], axis="columns", inplace=True)
    merged = preprocess_merged_remove_rows(merged)

    # Train test split
    X, y = split_and_encode(merged, "dx1")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Perform z-score normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    merged, features_to_keep = filter_by_feature_importance(0.005, merged, X_train, y_train, seed)
    X, y = split_and_encode(merged, "dx1")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Removing outliers using the Isolation Forest classifier
    X_train, y_train = isolation_forest_outlier_removal(X_train, y_train, seed)

    # Try out different models and see their performance

    # # Bagging
    # bcv = bagging_cv(X_train, y_train, seed, verbose=0)
    #
    # # RandomForest
    # rfcv = random_forest_cv(X_train, y_train, seed, verbose=0)
    #
    # # ExtraTrees
    # etcv = random_forest_cv(X_train, y_train, seed, verbose=0)

    # GradientBoosting
    gbcv = gradient_boosting_cv(X_train, y_train, seed, verbose=0)

    # XGBoost
    xgbcv = xgboost_cv(X_train, y_train, seed, verbose=0)


if __name__ == '__main__':
    main()

