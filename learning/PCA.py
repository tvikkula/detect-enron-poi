from sklearn.decomposition import RandomizedPCA as PCA
from matplotlib import pyplot as plt

def doPCA(data, n):
    pca = PCA(n_components=n)
    pca.fit(data)
    print pca.explained_variance_ratio_
    return pca

def plotPCA(transformed_data):
    # Better hope that there will be at most 8 components:
    colorlist = ['b','g','r','c','m','y','k','w']
    for row in transformed_data:
        for component in range(len(row)):
            plt.scatter(row[component], row[component],
                        color=colorlist[component])
    plt.show()