from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train(features_train, labels_train):
    clf = RandomForestClassifier(
        n_estimators = 10, min_samples_split = 2,
        n_jobs = -1, criterion = 'entropy',
        max_features = None
    )
    fit = clf.fit(features_train, labels_train)
    return fit 

def test(features_test, labels_test, model):
    pred = model.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    return acc
                         
