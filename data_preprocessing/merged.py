import itertools

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from data_preprocessing.basic_preprocessing import remove_others
from data_preprocessing.basic_preprocessing import load_data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt


def preprocess_merged_remove_rows(X):
    return X.dropna()


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

    # Encode dx1 (target variable)
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)

    return X, y


def filter_by_feature_importance(threshold, dataset, X_train, y_train):
    # Feature importance
    forest = ExtraTreesClassifier(n_estimators=500, random_state=0)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking

    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, list(dataset)[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    features_to_keep = set(itertools.compress(list(dataset), [i > threshold for i in importances]))
    features_to_keep.add('dx1')

    return remove_others(dataset, features_to_keep)


if __name__ == '__main__':
    merged = load_data()["merged"]
    merged.dropna(axis="rows", subset=["dx1"], inplace=True)
    merged.drop(labels=["ID", "Subject"], axis="columns", inplace=True)

    X, y = split_and_encode(merged, "dx1")

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Perform z-score normalization
    scaler = StandardScaler().fit_transform(X_train)

    merged = filter_by_feature_importance(0.005, merged, X_train, y_train)

    X, y = split_and_encode(merged, "dx1")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Try out different models and see their performance

    seed = 14

    #Bagging
    rsCV = RandomizedSearchCV()

    #RandomForest

    #ExtraTrees

    #GradientBoosting

    #XGBoost

