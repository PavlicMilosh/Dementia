from sklearn.impute import SimpleImputer

class Imputer:

    def __init__(self):
        pass

    def fit(self, X, y):
        self.model.fit(X, y)

    def transform(self, X):
        return self.model.transform(X)


class KNNImputation(Imputer):

    def __init__(self):
        pass


class MeanImputation(Imputer):

    def __init__(self):
        self.model = SimpleImputer(strategy='mean')


