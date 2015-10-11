#!/usr/bin/python
from sklearn.naive_bayes import GaussianNB
import numpy as np

def classify(features_train, labels_train):   
    gnb = GaussianNB()
    fit = gnb.fit(features_train, labels_train)
    return fit

def accuracy(features_test, labels_test, fit):
    pred = fit.predict(features_test)
    return np.sum(pred == labels_test) / float(len(pred))
