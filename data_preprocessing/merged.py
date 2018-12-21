from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

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


if __name__ == '__main__':
    merged = load_data()["merged"]
    merged.dropna(axis="rows", subset=["dx1"], inplace=True)
    merged.drop(labels=["ID", "Subject"], axis="columns", inplace=True)

    # Deal with missing values
    merged = preprocess_merged_remove_rows(merged)

    # Get X and y parts of the dataset
    y = merged["dx1"]
    X = merged.drop(labels="dx1", axis="columns")

    # Encode dx1 (target variable)
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Perform z-score normalization
    scaler = StandardScaler().fit_transform(X_train)

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
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    # Try out different models and see their performance
