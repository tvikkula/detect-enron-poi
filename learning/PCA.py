from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
def doPCA(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    print pca.explained_variance_ratio_
    first_pc = pca.components_[0]
    second_pc = pca.components_[1]
    return pca, first_pc, second_pc

def plotPCA(transformed_data, first_pc, second_pc):
    for i in transformed_data:
        plt.scatter(first_pc[0]*i[0], first_pc[1]*i[0], color='r')
        plt.scatter(second_pc[0]*i[1], second_pc[1]*i[1], color='c')
    plt.show()
