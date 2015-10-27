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
data_dict = pickle.load(open("data/data_dict_no_new_feature.pkl", "r"))

features_list = ['poi',
                 'expenses',
                 'director_fees',
                 'salary',
                 'to_messages',
                 'restricted_stock_deferred',
                 'from_poi_to_this_person',
                 'shared_receipt_with_poi'
                 ]

data = featureFormat(data_dict, features_list, sort_keys = True)

# Scale features:
mins = np.min(data, axis=0)
maxs = np.max(data, axis=0)
data = (data-mins)/(maxs-mins)

labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    stratifiedShuffleSplit(features, labels)

# Do some hyperparam validation:
best_svc, svc_grid_scores = ClassifySVM.gridsearch(
    features_train, labels_train
)
pprint.pprint(best_svc)

svmfit = ClassifySVM.train(features_train, labels_train, best_svc)

test_classifier(svmfit, data)

dump_classifier_and_data(svmfit, data_dict, features_list)
