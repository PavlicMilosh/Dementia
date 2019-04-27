from src.models.base_models import RandomForest, NaiveBayes, LogisticRegresssion, SVM, XGBoost, Bagging, \
    ExtraTrees, LDA, MLP


def get_model(model_name):
    if model_name == 'bagging':
        return Bagging()
    elif model_name == 'extraTrees':
        return ExtraTrees()
    elif model_name == 'randomForest':
        return RandomForest()
    elif model_name == 'naiveBayes':
        return NaiveBayes()
    elif model_name == 'logisticRegression':
        return LogisticRegresssion()
    elif model_name == 'svm':
        return SVM()
    elif model_name == 'xgboost':
        return XGBoost()
    elif model_name == 'lda':
        return LDA()
    elif model_name == 'mlp':
        return MLP()
    else:
        print('Choose one of predefined models')

