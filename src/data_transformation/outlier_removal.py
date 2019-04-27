from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from settings.settings import RANDOM_STATE


def remove_outliers(model_name, X, y, **add_params):
    """
    For given X and y, removes detected outliers using either Isolation Forest or Local Outlier Factor.

    :param model_name: str - 'isf' (for Isolation Forest), 'lof' (for Local Outlier Factor)
    :param X: numpy array
    :param y: numpy array
    :param add_params: additional_params for Isolation Forest / Local Outlier Factor models
    :return: X, y with removed outliers
    """
    model_name = model_name.lower()

    if model_name == 'isf':
        # n_estimators=150, max_samples=0.8, max_features=0.8, contamination="auto"
        clf = IsolationForest(behaviour='new', random_state=RANDOM_STATE, n_jobs=-1, **add_params)
    elif model_name == 'lof':
        clf = LocalOutlierFactor(n_jobs=-1, **add_params)
    else:
        raise Exception("Choose one of predefined models ('isf' or 'lof')")

    results = clf.fit_predict(X)

    outliers = len(list(filter(lambda x: x == -1, results)))
    print("Isolation forest found {} outliers".format(outliers))

    removing_indices = [i for i in range(0, len(results)) if results[i] == -1]
    X_new = np.delete(X, removing_indices, axis=0)
    y_new = [y[yi] for yi in range(0, len(y)) if results[yi] == 1]

    return X_new, y_new
