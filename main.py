#!/usr/bin/python

import sys
import pickle
import pprint
import numpy as np
sys.path.append("./tools/")
sys.path.append("./learning/")
from preprocess import *
import ClassifyNB, ClassifySVM, RFClassifier, PCA
from feature_format import targetFeatureSplit
from tester import test_classifier

### Load the numpy array with the dataset
data = np.load('data/enrondata_normalized.npy')
features_list = np.load('data/features_list.npy').tolist()

# Courtesy of http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
#, sets "divide-by-zero"-cases to 0
with np.errstate(divide='ignore', invalid='ignore'):
    c = np.divide(np.add(data[:,15], data[:,19]), np.add(data[:,13], data[:,2]))
    c[c == np.inf] = 0
    c = np.nan_to_num(c)
c = c.reshape(143,1)
fresh_data = np.append(data, c, 1)

# Add the principal components

# Create a list of feature names, handy for plot labeling etc.
features_only_list = list(features_list)
features_only_list.remove('poi')
features_only_list.append('email_ratio_with_poi')
features_list.append('email_ratio_with_poi')
labels, features = targetFeatureSplit(fresh_data)

features_train, features_test, labels_train, labels_test = \
    stratifiedShuffleSplit(features, labels)

### Do some PCA
pca = PCA.doPCA(features_train, n = 3)
transformed_train = pca.transform(features_train)
transformed_test = pca.transform(features_test)
#features_train = transformed_train
#features_test = transformed_test

# Now we have only PCA features:
#features_only_list = ['pca'+str(i) for i in range(len(features_train[0]))]
print features_only_list

# Get feature importance:
importance, selector = getFeatureImportance(features_train, labels_train, features_only_list, k = 9)
plt.figure()
plotFeatureImportance(importance, features_only_list, plt)
plt.show()

### Next try with selector
selector_train = selector.transform(features_train)

# Do some hyperparam validation:
best_svc, svc_grid_scores = ClassifySVM.gridsearch(selector_train, labels_train)
print("Best SVMC estimator found by grid search:")
pprint.pprint(best_svc)


# Do fits based on hyperparam validation
nbfit = ClassifyNB.train(selector_train, labels_train)
svmfit = ClassifySVM.train(selector_train, labels_train, best_svc)


### Probably better to test with precision and recall:
print 'Naive bayes:'
test_classifier(nbfit, data)
print 'SVM:'
test_classifier(svmfit, data)


dump_classifier_and_data(svmfit, my_dataset, features_list)
