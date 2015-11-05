#!/usr/bin/python
import sys
import pickle
import pprint
import numpy as np
sys.path.append("./tools/")
sys.path.append("./learning/")
from preprocess import *
import ClassifyNB, ClassifySVM, RFClassifier, PCA
from feature_format import targetFeatureSplit, featureFormat
from learningtester import test_classifier, dump_classifier_and_data

# Ignore the new feature as it messes up PCA
data_dict = pickle.load(open("data/own_data_dict.pkl", "r"))

features_list = getallFeatures(data_dict)
data = featureFormat(data_dict, features_list, sort_keys = True)

# Scale features:
mins = np.min(data, axis=0)
maxs = np.max(data, axis=0)
data = (data-mins)/(maxs-mins)

labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    stratifiedShuffleSplit(features, labels)

### Do some PCA
pca = PCA.doPCA(features_train, n = 4)
transformed_train = pca.transform(features_train)

# Do some hyperparam validation:
best_svc, svc_grid_scores = ClassifySVM.gridsearch(
    transformed_train, labels_train
)

svmfit = ClassifySVM.train(transformed_train, labels_train, best_svc)

test_classifier(svmfit, data)

dump_classifier_and_data(svmfit, data_dict, features_list)
