#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("./tools/")
from preprocess import *
from feature_format import targetFeatureSplit
from tester import dump_classifier_and_data

### Load the numpy array with the dataset
data = np.load('enrondata_normalized.npy')

labels, features = targetFeatureSplit(data)
# Change labels to ints:
labels = map(lambda x: int(x), labels)
#pprint.pprint(labels)
print len(labels)
print len(features)
print len(features[0])
features_train, features_test, labels_train, labels_test = \
    trainTestSplit(features, labels, test_size=0.3)

# Get feature importance:
getFeatureImportance(features_train, labels_train)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

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
