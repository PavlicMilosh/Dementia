import os
import os.path as osp

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from settings.settings import RANDOM_STATE, CV_FOLDS


class BaseModel(object):

    def __init__(self):
        self.params = {}


    def fit(self, x_train, y_train):
        self.grid_search = GridSearchCV(self.model,
                                        param_grid=self.params,
                                        scoring='f1_micro',
                                        n_jobs=-1,
                                        refit=True,
                                        cv=CV_FOLDS,
                                        verbose=0)

        self.grid_search.fit(x_train, y_train)

        self.model = self.grid_search.best_estimator_


    def predict(self, inputs):
        return self.model.predict(inputs)


    def evaluate(self, y, y_predicted, model_name):
        f1 = f1_score(y, y_predicted, average='micro')
        print('{} model achieved {} F-measure score.'.format(model_name, f1))
        print('*' * 80)
        return f1


    def load_model(self, dir_path):
        model_path = osp.join(dir_path, 'model')
        lab2ind_path = osp.join(dir_path, 'lab2ind')
        ind2lab_path = osp.join(dir_path, 'ind2lab')

        if not osp.exists(model_path):
            raise Exception("'model' file is not present inside given path: {}".format(dir_path))
        if not osp.exists(lab2ind_path):
            raise Exception("'lab2ind' file is not present inside given path: {}".format(dir_path))
        if not osp.exists(ind2lab_path):
            raise Exception("'ind2lab' file is not present inside given path: {}".format(dir_path))

        try:
            self.model = joblib.load(model_path)
            self.ind2lab = joblib.load(ind2lab_path)
            self.lab2ind = joblib.load(lab2ind_path)
        except Exception as e:
            print(e)
            print("Couldn't load model from given dir {}!".format(dir_path))
            exit(1)


    def save_model(self, dir_path):
        try:
            os.makedirs(osp.dirname(dir_path), exist_ok=True)
        except Exception as e:
            print(e)
            print("Couldn't save model on given dir {}!".format(dir_path))


    def print_search_results(self):
        params_str = ""
        for k, v in self.grid_search.best_params_.items():
            params_str += 18 * " " + "{:<20} : {}\n".format(k, v)
        print("=" * 150)
        print("Best score:       {}\n".format(self.grid_search.best_score_))
        print("Best params:")
        print(params_str)
        print("Best estimator:\n")
        print(self.grid_search.best_estimator_)
        print("=" * 150)


class Bagging(BaseModel):

    def __init__(self):
        super(Bagging).__init__()
        self.params = {'n_estimators': list(range(100, 251, 50)),
                       'warm_start': [True],
                       'max_samples': [0.4, 0.6, 0.8]
                      }
        self.model = BaggingClassifier(n_estimators=140, random_state=RANDOM_STATE)


class RandomForest(BaseModel):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.params = {'n_estimators': [200, 250, 350, 400], # list(range(200, 400, 100)),
                       'criterion': ['gini'],  # ['gini', 'entropy'],
                       'min_samples_split': list(range(5, 16, 5)),
                       'max_depth': list(range(10, 21, 5))
                       }

        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=3, random_state=RANDOM_STATE)


class ExtraTrees(BaseModel):
    def __init__(self):
        super(ExtraTrees, self).__init__()
        self.params = { 'n_estimators': [200, 250, 350, 400], # list(range(100, 1500, 50)),
                        'criterion': ['gini'], # , 'entropy'],
                        'min_samples_split': list(range(5, 16, 5)),
                        'max_depth': list(range(2, 20, 2))
                      }
        self.model = ExtraTreesClassifier(n_estimators=140, random_state=RANDOM_STATE)


class NaiveBayes(BaseModel):
    def __init__(self):
        super(NaiveBayes, self).__init__()
        self.params = {'alpha': [.01, .05, .1, .2, .5, 1]}
        self.model = MultinomialNB()


class LogisticRegresssion(BaseModel):
    def __init__(self):
        super(LogisticRegresssion, self).__init__()
        self.params = {'penalty': ['l2'],
                       'C': [.1, .2, .5, 1],
                       'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                       'max_iter': list(range(100, 1001, 100))
                       }
        self.model = LogisticRegression(solver='warn', multi_class='auto', random_state=RANDOM_STATE)


class SVM(BaseModel):
    def __init__(self):
        super(SVM, self).__init__()
        self.params = {'C': [.01, .05, .1, .2, .5, 1, 2],
                       'kernel': ['linear', 'rbf'],
                       'decision_function_shape': ['ovo', 'ovr']
                       }
        self.model = SVC(gamma='scale', random_state=RANDOM_STATE)


class XGBoost(BaseModel):
    def __init__(self):
        super(XGBoost, self).__init__()
        self.params = {'nthread': [1],
                       'learning_rate': [0.1],
                       'max_depth': list(range(4, 13, 4)),
                       'min_child_weight': list(range(2, 7, 2)),
                       'subsample': [0.8],
                       'n_estimators': list(range(50, 201, 50))
                       }
        self.model = XGBClassifier(objective='multi:softmax', num_class=3, n_jobs=-1, random_state=RANDOM_STATE)


class LDA(BaseModel):
    def __init__(self):
        super(LDA, self).__init__()
        self.params = {
                       'learning_decay': [.5, .6, .7, .8, .9, 1]
                       }
        self.model = LatentDirichletAllocation(n_components=5, n_jobs=-1, random_state=RANDOM_STATE)


class MLP(BaseModel):
    def __init__(self, hidden_size=(2048,)):
        super(MLP, self).__init__()
        self.model = MLPClassifier(hidden_layer_sizes=hidden_size)
