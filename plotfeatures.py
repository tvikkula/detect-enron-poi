from matplotlib import pyplot as plt
import pickle
import pprint
files = {"own features": "featureselection_own.pkl",
         "orig. features": "featureselection_orig.pkl",
         "own PCA": "pcacomponentselection_own.pkl",
         "orig. PCA": "pcacomponentselection_orig.pkl"}
for key,value in files.iteritems():
    selectk = pickle.load(open(value, "r"))
    selectpca = pickle.load(open(value, "r"))
    pprint.pprint(selectk)
    plt.plot(selectk.keys(), [i[0] for i in selectk.values()], marker='o',
             linestyle='--', label='Precision')
    plt.plot(selectk.keys(), [i[1] for i in selectk.values()], marker='o',
             linestyle='--', label='Recall')
    plt.plot(selectk.keys(), [i[2] for i in selectk.values()], marker='o',
             linestyle='--', label='F1')
    plt.plot(selectk.keys(), [i[3] for i in selectk.values()], marker='o',
             linestyle='--',label='F2')
    plt.xticks(range(1,len(selectk.keys())))
    plt.xlabel('# of Features')
    plt.ylabel('Score value')
    plt.title('Scores by # of features using ' + key)
    plt.legend()
    plt.show()