from sklearn.model_selection import train_test_split

from data_preprocessing.basic_preprocessing import load_data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def preprocess_adrc_remove_rows(X):
    X.dropna(inplace=True)


def preprocess_adrc_mean_imputing(X):
    imputer = SimpleImputer()
    imputer.fit(X)
    imputer.transform(X)


def preprocess_adrc_models_imputing(X):
    pass


if __name__ == '__main__':
    adrc = load_data()["adrc"]
    adrc.drop(labels=["dx2", "dx3", "dx4", "dx5", "ADRC_ADRCCLINICALDATA ID", "Subject"],
              axis="columns",
              inplace=True)
    adrc.dropna(axis="rows", subset=["dx1"], inplace=True)

    # Get X and y parts of the dataset
    y = adrc["dx1"]
    X = adrc.drop(labels="dx1", axis="columns")

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    # Encode dx1 (target variable)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Perform z-score normalization
    scaler = StandardScaler().fit_transform(X)

    # Try out different models and see their performance
