#!/usr/bin/python                                                               
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train(features_train, labels_train):
    svc = SVC(kernel = 'rbf', gamma = 10, C = 10)
    fit = svc.fit(features_train, labels_train)
    return fit

def test(features_test, labels_test, fit):
    pred = fit.predict(features_test)
    return accuracy_score(pred, labels_test)


