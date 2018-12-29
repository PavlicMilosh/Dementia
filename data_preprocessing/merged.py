import itertools

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from data_preprocessing.basic_preprocessing import remove_others
from data_preprocessing.basic_preprocessing import load_data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
import xgboost as xgb



def preprocess_merged_remove_rows(df):
    return df.dropna()


def preprocess_merged_mean_imputing(X):
    imputer = SimpleImputer()
    imputer.fit(X)
    imputer.transform(X)


def preprocess_merged_models_imputing(X):
    pass


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


def filter_by_feature_importance(threshold, dataset, X_train, y_train, seed):
    # Feature importance
    forest = ExtraTreesClassifier(n_estimators=500, random_state=seed)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
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

    return remove_others(dataset, features_to_keep)


if __name__ == '__main__':
    seed = 14

    merged = load_data()["merged"]
    merged.dropna(axis="rows", subset=["dx1"], inplace=True)
    merged.drop(labels=["ID", "Subject"], axis="columns", inplace=True)

    merged = preprocess_merged_remove_rows(merged)

    X, y = split_and_encode(merged, "dx1")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Perform z-score normalization
    scaler = StandardScaler().fit_transform(X_train)

    merged = filter_by_feature_importance(0.005, merged, X_train, y_train, seed)

    X, y = split_and_encode(merged, "dx1")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Try out different models and see their performance

    # Bagging

    # RandomForest

    # ExtraTrees

    # GradientBoosting
    gradient_boosting = GradientBoostingClassifier(n_estimators=140,
                                                   min_samples_split=2,
                                                   max_depth=2,
                                                   random_state=seed)
    params = {
        'n_estimators': list(range(100, 1000, 50)),           # 140
        'min_samples_split': list(range(2, 20, 2)),           # 2
        'max_depth': list(range(2, 20, 2))                    # 2
    }
    gCV1 = GridSearchCV(estimator=gradient_boosting,
                        param_grid=params,
                        scoring='f1-micro',
                        n_jobs=-1,
                        refit=True,
                        cv=3,
                        verbose=2,
                        return_train_score='warn')
    gCV1 = gCV1.fit(X_train, y_train)

    # XGBoost

    merged = encode(merged, 'dx1')

    # train, test = train_test_split(merged, test_size=0.2)

    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

    params = {'nthread': [4],
              'learning_rate': [0.1],                           # 0.1
              'max_depth': list(range(2, 22, 2)),               # 16
              'min_child_weight': list(range(1, 6)),            # 1
              'subsample': [0.8, 1],                            # 0.8
              'n_estimators': list(range(100, 1000, 50)),      # 100
              'seed': [seed]}

    gCV2 = GridSearchCV(estimator=xgb_model,
                        scoring='f1_micro',
                        param_grid=params,
                        n_jobs=-1,
                        refit=True,
                        cv=3,
                        verbose=2,
                        return_train_score='warn')
    gCV2 = gCV2.fit(X_train, y_train)



