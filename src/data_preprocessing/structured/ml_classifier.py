import os

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os.path as osp
from src.models.models import get_model


class MLClassifier:

    def __init__(self, model_name, model_dir, load=False):
        self.model_name = model_name
        self.model_dir = model_dir
        self.pipeline = None
        self.lab2ind = None
        self.ind2lab = None

        if load:
            self.load()


    def train(self, X, y, **fit_params):
        self.lab2ind = {yi: idx for idx, yi in enumerate(np.unique(y))}
        self.ind2lab = {idx: yi for yi, idx in self.lab2ind.items()}
        y = np.array(list(map(lambda yi: self.lab2ind[yi], y)))


        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', get_model(self.model_name))
        ])

        # print report
        self.pipeline.fit(X, y, **fit_params)
        predicted = self.pipeline.predict(X)

        # can do this because the order of items is consistent in one instance of dictionary
        indexes = list(self.ind2lab.keys())
        labels = list(self.ind2lab.values())
        # print(str(classification_report(y, predicted, labels=indexes, target_names=labels)))


    def predict(self, X):
        return self.pipeline.predict(X)


    def evaluate(self, X, y):
        predicted = self.pipeline.predict(X)
        y = np.array(list(map(lambda yi: self.lab2ind[yi], y)))
        indexes = list(self.ind2lab.keys())
        labels = list(self.ind2lab.values())
        # print("acuracy: " + str(accuracy_score(y, predicted)))
        # print("micro-f1: " + str(f1_score(y, predicted, average='micro')))

        report = classification_report(y, predicted, labels=indexes, target_names=labels)
        print(str(report))


    def print_model(self):
        self.pipeline.named_steps['model'].print_search_results()

    def load(self):
        pipeline_path = osp.join(self.model_dir, 'final')
        lab2ind_path = osp.join(self.model_dir, 'lab2ind')
        ind2lab_path = osp.join(self.model_dir, 'ind2lab')
        if not osp.exists(pipeline_path):
            raise Exception("'final' file is not present inside given path: {}".format(self.model_dir))
        if not osp.exists(lab2ind_path):
            raise Exception("'lab2ind' file is not present inside given path: {}".format(self.model_dir))
        if not osp.exists(ind2lab_path):
            raise Exception("'ind2lab' file is not present inside given path: {}".format(self.model_dir))
        try:
            self.pipeline = joblib.load(pipeline_path)
            self.ind2lab = joblib.load(ind2lab_path)
            self.lab2ind = joblib.load(lab2ind_path)
        except Exception as e:
            print(e)
            print("Couldn't load model from given dir {}!".format(self.model_dir))
            exit(1)


    def save(self):
        pipeline_path = osp.join(self.model_dir, 'final')
        lab2ind_path = osp.join(self.model_dir, 'lab2ind')
        ind2lab_path = osp.join(self.model_dir, 'ind2lab')

        try:
            os.makedirs(self.model_dir, exist_ok=True)
            joblib.dump(self.pipeline, pipeline_path)
            joblib.dump(self.lab2ind, lab2ind_path)
            joblib.dump(self.ind2lab, ind2lab_path)
        except Exception as e:
            print(e)
            print("Couldn't save model on given dir {}!".format(self.model_dir))
