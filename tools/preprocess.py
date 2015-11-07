#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr

# So that pyplot axis labels don't overflow out of screen
rcParams.update({'figure.autolayout': True})

def getallFeatures(data_dict):
    features_list = data_dict.itervalues().next().keys()
    features_list.remove('poi')
    features_list.remove('email_address')
    features_list.insert(0, 'poi')
    return features_list

def trainTestSplit(features, labels, test_size=0.2):
    from sklearn.cross_validation import train_test_split
    return train_test_split(features, labels, test_size=test_size, random_state=42)

def stratifiedShuffleSplit(features, labels, folds=1000):
    from sklearn.cross_validation import StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(labels, folds)
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
    return features_train, features_test, labels_train, labels_test

def minmaxScale(data):
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    data_normal = (data-mins)/(maxs-mins)
    return data_normal

def defineCorrelation(x, y, features_list, data):
    # Calculate spearman between x and y:
    indexX = features_list.index(x)
    indexY = features_list.index(y)
    xarr = [row[indexX] for row in data]
    yarr = [row[indexY] for row in data]
    return spearmanr(xarr, yarr)

def getFeatureImportance(features_train, labels_train, features_list, k):
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k)
    selector.fit(features_train, labels_train)
    ### Extend function to return a sorted list with keysnames and scores_&pvalues_.
    scores = zip(features_list, selector.scores_, selector.pvalues_)
    scores.sort(key = lambda x: x[1], reverse=True)
    return scores, selector

def plotFeatureImportance(importance, features_only_list, plt):
    plt.bar(np.arange(len(features_only_list)), [i[1] for i in importance])
    plt.xticks(np.arange(0.5, len(features_only_list), 1), [i[0] for i in importance], rotation=90)
    return plt

def outlierRegression(features_train, labels_train):
    '''
    Train the model with all features, print errors of all training data
    '''
    from sklearn import linear_model
    cls = linear_model.LinearRegression()

    reg = cls.fit(features_train, labels_train)
    print reg.intercept_
    print reg.coef_
    predictions = reg.predict(features_train)
    return predictions

# Note: Only use this for regression problems
def outlierCleaner(predictions, features_train, labels_train, amount_omitted = 0.1):
    '''
    Run the outlier cleaner on the training data and the labels. Check results by viewing
    residuals of each row. Omit highest residuals. The amount omitted is in percentages.
    '''
    # Outlier ideas:
    #  plot distributions?
    #
    cleaned_data = map(lambda x,y,z:(x,y,z),
                       features_train, labels_train, (predictions-labels_train)**2)

    cleaned_data.sort(key=lambda x:x[2])
    # Mean of the cleaned_data residuals:
    print reduce(lambda x,y: x[2] + y[2], cleaned_data) / len(cleaned_data)
    # Median of the cleaned_data residuals:
    print cleaned_data[len(cleaned_data)/2]
    print cleaned_data[len(cleaned_data)-20:len(cleaned_data)]
    cleaned_data = cleaned_data[0:len(cleaned_data)*(1-amount_omitted)]
    return cleaned_data
