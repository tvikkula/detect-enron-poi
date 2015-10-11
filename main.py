#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("./tools/")
sys.path.append("./learning/")
from preprocess import *
import ClassifyNB, ClassifySVM, RFClassifier
from feature_format import targetFeatureSplit
from tester import test_classifier

### Load the numpy array with the dataset
data = np.load('data/enrondata_normalized.npy')
features_list = np.load('data/features_list.npy')

# meh..
features_only_list = np.delete(features_list, 0)
print features_only_list

labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    trainTestSplit(features, labels, test_size=0.3)

# Get feature importance:
importance = getFeatureImportance(features_train, labels_train, features_only_list)
pprint.pprint(importance)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

nbfit = ClassifyNB.train(features_train, labels_train)
svmfit = ClassifySVM.train(features_train, labels_train)
rffit = RFClassifier.train(features_train, labels_train)
### Probably better to test with precision and recall:
print 'Naive bayes:'
test_classifier(nbfit, data)
print 'SVM:'
test_classifier(svmfit, data)
print 'Random Forest:'
test_classifier(rffit, data)







### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)
