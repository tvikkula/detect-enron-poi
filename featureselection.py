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

data_dict = pickle.load(open("data/data_dict_no_new_feature.pkl", "r"))

features_list = getallFeatures(data_dict)

data = featureFormat(data_dict, features_list, sort_keys = True)

# Scale features:
mins = np.min(data, axis=0)
maxs = np.max(data, axis=0)
data = (data-mins)/(maxs-mins)

labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    stratifiedShuffleSplit(features, labels)

results = {}
for k in xrange(1, len(features_list)):
    print k
    # Get feature importance:
    importance, selector = getFeatureImportance(features_train, labels_train, features_list, k)

    ### Next try with selector
    selector_train = selector.transform(features_train)
    # Do some hyperparam validation:
    best_svc, svc_grid_scores = ClassifySVM.gridsearch(
        selector_train, labels_train
    )

    svmfit = ClassifySVM.train(selector_train, labels_train, best_svc)

    precision, recall, f1, f2 = test_classifier(svmfit, data)
    results[k] = [precision, recall, f1, f2]

pprint.pprint(results)
pickle.dump(results, open('featureselection_orig.pkl', "w") )

pcaresults = {}
for k in xrange(1, 7):
    print k
    ### Do some PCA
    pca = PCA.doPCA(features_train, n = k)
    transformed_train = pca.transform(features_train)

    # Do some hyperparam validation:
    best_svc, svc_grid_scores = ClassifySVM.gridsearch(
        transformed_train, labels_train
    )

    svmfit = ClassifySVM.train(transformed_train, labels_train, best_svc)

    precision, recall, f1, f2 = test_classifier(svmfit, data)
    pcaresults[k] = [precision, recall, f1, f2]

pprint.pprint(pcaresults)
pickle.dump(pcaresults, open('pcacomponentselection_orig.pkl', "w") )