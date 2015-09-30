#!/usr/bin/python

def getallFeatures(data_dict):
    features_list = data_dict.itervalues().next().keys()
    print features_list
    features_list.remove('poi')
    features_list.remove('email_address')
    features_list.insert(0, 'poi')
    return features_list

def trainTestSplit(features, labels, test_size=0.3):
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=test_size, random_state=42)

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

def outlierCleaner(predictions, features_train, labels_train, amount_omitted = 0.1):
    '''
    Run the outlier cleaner on the training data and the labels. Check results by viewing
    residuals of each row. Omit highest residuals. The amount omitted is in percentages.
    '''
    cleaned_data = map(lambda x,y,z:(x,y,z),
                       features, labels, (predictions-net_worths)**2)

    cleaned_data.sort(key=lambda x:x[2])
    # Mean of the cleaned_data residuals:
    print reduce(lambda x,y: x[2] + y[2], cleaned_data) / len(cleaned_data)
    # Median of the cleaned_data residuals:
    print cleaned_data[len(cleaned_data)/2]
    print cleaned_data[len(cleaned_data)-20:len(cleaned_data)]
    cleaned_data = cleaned_data[0:len(cleaned_data)*(1-amount_omitted)]
    return cleaned_data
