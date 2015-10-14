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

# Drop features:
deletables = ['other', 'from_messages', 'loan_advances', 'deferral_payments', \
              'total_payments', 'deferred_income', 'total_stock_value',\
              'from_this_person_to_poi', 'exercised_stock_options',\
              'bonus', 'long_term_incentive',\
              'restricted_stock',\
              'to_messages']
for deletable in deletables:
    data = np.delete(data, features_list.index(deletable), 0)
    features_list.remove(deletable)

# Add the principal components

# Create a list of feature names, handy for plot labeling etc.
features_only_list = features_list
features_only_list.remove('poi')

labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    stratifiedShuffleSplit(features, labels)

### Do some PCA
pca = PCA.doPCA(features_train, n = 3)
transformed_train = pca.transform(features_train)
transformed_test = pca.transform(features_test)
#features_train = transformed_train
#features_test = transformed_test
#PCA.plotPCA(transformed_train)

# Now we have only PCA features:
#features_only_list = ['pca'+str(i) for i in range(len(features_train[0]))]
print features_only_list
## PCA as new features yeilded the same results
## Using PCA only yeilded better results with an SVM. The decision boundary was made
## significantly simpler by reducing dimensionality.

# Get feature importance:
importance = getFeatureImportance(features_train, labels_train, features_only_list)
plt.figure()
plotFeatureImportance(importance, features_only_list, plt)
plt.show()
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Do some hyperparam validation:
best_svc, svc_grid_scores = ClassifySVM.gridsearch(features_train, labels_train)
print("Best SVMC estimator found by grid search:")
pprint.pprint(best_svc)
#pprint.pprint(svc_grid_scores[0:10])
#best_rfc, rfc_grid_scores = RFClassifier.gridsearch(features_train, labels_train)
#print("Best RFC estimator found by grid search:")
#pprint.pprint(best_rfc)
#pprint.pprint(rfc_grid_scores[0:10])


# Do fits based on hyperparam validation
nbfit = ClassifyNB.train(features_train, labels_train)
svmfit = ClassifySVM.train(features_train, labels_train, best_svc)
#rffit = RFClassifier.train(features_train, labels_train, best_rfc)

### Probably better to test with precision and recall:
print 'Naive bayes:'
test_classifier(nbfit, data)
print 'SVM:'
test_classifier(svmfit, data)
#print 'Random Forest:'
#test_classifier(rffit, data)

# Linear SVM and Naive Bayes seems to be best.


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
