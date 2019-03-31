import os
import os.path as osp

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
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

    def evaluate(self, y, y_predicted, model_name):
        f1 = f1_score(y, y_predicted, average='micro')
        print('{} model achieved {} F-measure score.'.format(model_name, f1))
        print('*' * 80)
        return f1


    def load_model(self, path):
        try:
            self.model = joblib.load(path)
        except Exception as e:
            print(e)
            print("Couldn't load model from path {}!".format(path))
            exit(1)


    def save_model(self, path):
        try:
            os.makedirs(osp.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
        except Exception as e:
            print(e)
            print("Couldn't save model on path {}!".format(path))


class Bagging(BaseModel):

    def __init__(self):
        super(Bagging).__init__()
        self.params = {'n_estimators': list(range(100, 1500, 50)),
                       'warm_start': [True, False],
                       'max_samples': [0.6, 0.8, 1.0]
                      }
        self.model = BaggingClassifier(n_estimators=140, random_state=RANDOM_STATE)


class RandomForest(BaseModel):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.params = {'n_estimators': list(range(100, 1000, 200)),
                       'criterion': ['gini', 'entropy'],
                       'min_samples_split': list(range(2, 20, 2)),
                       'max_depth': list(range(2, 20, 2))
                       }

        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=3, random_state=RANDOM_STATE)


class ExtraTrees(BaseModel):
    def __init__(self):
        super(ExtraTrees, self).__init__()
        self.params = { 'n_estimators': list(range(100, 1500, 50)),
                        'criterion': ['gini', 'entropy'],
                        'min_samples_split': list(range(2, 20, 2)),
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
                       'subsample': [0.8, 1],
                       'n_estimators': list(range(100, 1000, 200))
                       }
        self.model = XGBClassifier(objective='multi:softmax', num_class=5, n_jobs=-1, random_state=RANDOM_STATE)


class LDA(BaseModel):
    def __init__(self):
        super(LDA, self).__init__()
        self.params = {
                       'learning_decay': [.5, .6, .7, .8, .9, 1]
                       }
        self.model = LatentDirichletAllocation(n_components=5, n_jobs=-1, random_state=RANDOM_STATE)
