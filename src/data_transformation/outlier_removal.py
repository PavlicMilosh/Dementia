from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from settings.settings import RANDOM_STATE


def remove_outliers(model_name, X, y, **add_params):

    model_name = model_name.lower()

    if model_name == 'isf':
        # n_estimators=150, max_samples=0.8, max_features=0.8, contamination="auto"
        clf = IsolationForest(behaviour='new', random_state=RANDOM_STATE, n_jobs=-1, **add_params)
    elif model_name == 'lof':
        clf = LocalOutlierFactor(n_jobs=-1, **add_params)
    else:
        print('Choose one of predefined models')
        return

    results = clf.fit_predict(X.values)

    outliers = len(list(filter(lambda x: x == -1, results)))
    print("Isolation forest found {} outliers".format(outliers))

    removing_indices = [i for i in range(0, len(results)) if results[i] == -1]
    X_new = X.drop(X.index[removing_indices])
    y_new = [y[yi] for yi in range(0, len(y)) if results[yi] == 1]

    return X_new, y_new