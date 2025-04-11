import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from OutlierValuesDetection import get_data
from Printing import create_directory, create_workbook, save_to_excel_1d
from sklearn.manifold import TSNE
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams["font.sans-serif"] = ["Bahnschrift"]
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.unicode_minus"] = False


# Clustering based on the values of all features
def clustering_features(data, n_clusters):
    ml_features = KMeans(n_clusters=n_clusters, random_state=0)
    ml_features.fit(data)
    labels_of_features = ml_features.labels_
    cla_of_features = []
    for i in range(8):
        samples = [index for index, x in enumerate(labels_of_features) if x == i]
        cla_of_features.append(samples)
    return cla_of_features


def plotting_clusters_fea(n_clusters, data, data_label, label_index, index, columns, text):
    plt.subplot(1, 4, index)
    markers = ['o', ',', 'v', '^', '<', '>', 'p', 'H']
    colors = ['blue', 'orange', 'red', 'purple', 'pink', 'green', 'grey', 'cyan']
    labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'cluster6', 'cluster7', 'cluster8']
    for i in range(n_clusters):
        samples = data_label[i]
        x = data[samples, 0]
        y = data[samples, 1]
        print(len(x))
        plt.scatter(x, y, c=colors[i], s=50, label=labels[i], alpha=0.4, edgecolors='black', marker=markers[i])
        for j in range(len(x)):
            if samples[j] in label_index:
                plt.annotate(str(samples[j]), xy=(x[j], y[j]), xytext=(x[j] + 0.1, y[j] + 0.1))
                plt.scatter(x[j], y[j], c=colors[i], s=80, label=None, alpha=0.4, edgecolors='black')
    plt.legend(ncol=columns[index-1], loc='upper right')
    plt.text(2.6, 7, text[index - 1])
    # plt.text(-4.8, 7, text[index - 1])
    plt.ylim((-8, 8))
    plt.grid(True, linestyle='-.')
    plt.xlabel('Component1', fontdict={'size': 11})
    plt.ylabel('Component2', fontdict={'size': 11})


def plotting_clusters_tar(n_clusters, data, data_label, index, columns, text):
    plt.subplot(1, 4, index)
    markers = ['o', ',', 'v', '^', '<', '>', 'p', 'H']
    colors = ['blue', 'orange', 'red', 'purple', 'pink', 'green', 'grey', 'cyan']
    labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'cluster6', 'cluster7', 'cluster8']
    maximum = []
    for i in range(n_clusters):
        samples = data_label[i]
        x = samples
        y = data[samples]
        plt.scatter(x, y, c=colors[i], s=50, label=labels[i], alpha=0.4, edgecolors='black', marker=markers[i])
        maximum.append(np.max(data[samples]))
    plt.legend(ncol=columns[index - 1], loc='upper right')
    plt.ylim((0.6, 1.95))
    plt.yticks(np.round(maximum, 2) + 0.02)
    plt.grid(True, linestyle='-.', axis='y')
    plt.text(-3, 1.85, text[index - 1])
    plt.xlabel('Sample Number')
    plt.ylabel('Energy Barrier (eV)')


# Clustering based on the values of target
def clustering_target(data, n_clusters):
    ml_target = KMeans(n_clusters=n_clusters, random_state=0)
    ml_target.fit(np.array(data).reshape((-1, 1)))
    labels_of_target = ml_target.labels_
    cla_of_target = []
    for i in range(8):
        samples = [index for index, x in enumerate(labels_of_target) if x == i]
        cla_of_target.append(samples)
    return cla_of_target


if __name__ == "__main__":
    path = './Data/MD1.xlsx'
    original_data, duplicate_data, columns = get_data(path, 'data')
    features = StandardScaler().fit_transform(duplicate_data[:, :-1])
    target = duplicate_data[:, -1]
    all = StandardScaler().fit_transform(duplicate_data)

    classes_of_all = clustering_features(all, 2)
    classes_of_features = clustering_features(features, 2)
    classes_of_target = clustering_target(target, 2)

    # Write the clustering results
    '''
    wb_name = './Result/Revised_85/Clustering/Clusters_2.xlsx'
    create_workbook(wb_name)
    for i in range(len(classes_of_features)):
        save_to_excel_1d(classes_of_features[i], str(i) + 'th', wb_name, 'clusters_of_features', i + 1, 2)
        save_to_excel_1d(classes_of_target[i], str(i) + 'th', wb_name, 'clusters_of_target', i + 1, 2)
        save_to_excel_1d(classes_of_all[i], str(i) + 'th', wb_name, 'clusters_of_all', i + 1, 2)
    '''

    # Plotting for clusters on all features / data
    '''
    low_dim_data = TSNE(n_components=2, random_state=80).fit_transform(all)
    # save_path = './Result/Revised_85/Clustering/Clustering_all_data.tif'
    save_path = './Result/Revised_85/Clustering/Clustering_all_features.tif'
    # text = ['(A) K = 8', '(B) K = 6', '(C) K = 4', '(D) K = 2']
    text = ['(a) K = 8', '(b) K = 6', '(c) K = 4', '(d) K = 2']
    legend_columns = [2, 2, 1, 1]
    plt.figure(figsize=(20, 4))
    for i, x in enumerate([8, 6, 4, 2]):
        # classes_of_all = clustering_features(all, x)
        classes_of_features = clustering_features(features, x)
        # plotting_clusters_fea(x, low_dim_data, classes_of_all, [], i + 1, legend_columns, text)
        plotting_clusters_fea(x, low_dim_data, classes_of_features, [], i + 1, legend_columns, text)
    plt.savefig(save_path)
    plt.show()
    '''

    # Plotting for clusters on the values of target
    # '''
    save_path = './Result/MD1/Clustering/Clustering_target.tif'
    text = ['(1) K = 8', '(2) K = 6', '(3) K = 4', '(4) K = 2']
    legend_columns = [2, 2, 2, 2]
    plt.figure(figsize=(20, 4))
    for i, x in enumerate([8, 6, 4, 2]):
        classes_of_target = clustering_target(target, x)
        plotting_clusters_tar(x, target, classes_of_target, i+1, legend_columns, text)
    plt.savefig(save_path)
    plt.show()
    # '''
