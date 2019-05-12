import itertools
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from settings.settings import RANDOM_STATE, DATA_VISUALISATION_DIR


def plot_feature_importance(features_names, importances, std):
    indices = np.argsort(importances)

    plt.figure(figsize=(6, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title("Feature importances")

    ax = plt.gca()
    ax.grid(True)

    plt.barh(range(len(indices)),
             importances[indices],
             color="r",
             xerr=std[indices],
             align="center")

    plt.yticks(range(len(indices)), [features_names[i] for i in indices])
    plt.ylim([-1, len(indices)])
    plt.xlabel('Relative Importance')
    plt.show()
    plt.savefig(osp.join(DATA_VISUALISATION_DIR, 'importance_graph.png'))


def print_feature_importance(feature_names, importances):
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for i in range(len(indices)):
        print("{}. feature {} ({})".format(i + 1, feature_names[indices[i]], importances[indices[i]]))


def filter_by_feature_importance(threshold, dataset: pd.DataFrame):
    X = np.array(dataset.drop('dx1', axis='columns'))
    y = np.ravel(dataset['dx1'])

    # Feature importance
    forest = ExtraTreesClassifier(n_estimators=500, random_state=RANDOM_STATE)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    feature_importances = pd.DataFrame(forest.feature_importances_,
                                       index=dataset.drop(['dx1'], axis='columns').columns,
                                       columns=['importance'])\
        .sort_values('importance', ascending=False)

    # Plot and print the feature importances of the forest
    plot_feature_importance(list(dataset.columns), importances, std)
    print_feature_importance(list(dataset.columns), importances)

    features_to_keep = set(itertools.compress(list(dataset), [i > threshold for i in importances]))
    features_to_keep.add('dx1')

    return features_to_keep