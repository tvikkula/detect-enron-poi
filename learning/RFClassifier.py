from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

def train(features_train, labels_train, rfc = None):
    if (rfc == None):
        rfc = RandomForestClassifier(
            n_estimators = 2, min_samples_split = 2,
            n_jobs = -1, criterion = 'entropy',
            max_features = None
        )
    fit = rfc.fit(features_train, labels_train)
    return fit 

def test(features_test, labels_test, model):
    pred = model.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    return acc
                         
def gridsearch(features_train, labels_train, n):
    clf = RandomForestClassifier(
        n_estimators = n,
        n_jobs = -1
    )
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'min_samples_split': [1, 2, 5, 10, 20, 50],
        'max_features': ['sqrt', 'log2', None]
    }
    clf = GridSearchCV(clf, param_grid, scoring='f1')
    clf.fit(features_train, labels_train)
    scores = clf.grid_scores_
    # Sort by mean (note, it's using namedtuples)
    scores.sort(key=lambda x:x.mean_validation_score, reverse=True)
    return clf.best_estimator_, scores