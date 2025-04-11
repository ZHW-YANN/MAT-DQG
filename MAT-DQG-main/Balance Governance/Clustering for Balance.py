import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from SingleDimensionalOutlierDetection import get_data
from Printing import create_directory, create_workbook, save_to_excel_1d
from sklearn.manifold import TSNE
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.unicode_minus"] = False


# 基于所有特征数据的聚类
def clustering_features(data, n_clusters):
    '''
    :param data: 所有特征数据
    :param n_clusters: 类簇数目
    :return:类别标签
    '''
    ml_features = KMeans(n_clusters=n_clusters, random_state=0)
    ml_features.fit(data)
    labels_of_features = ml_features.labels_
    cla_of_features = []
    for i in range(n_clusters):
        samples = [index for index, x in enumerate(labels_of_features) if x == i]
        cla_of_features.append(samples)
    return cla_of_features


# 基于目标性能数据的聚类
def clustering_target(data, n_clusters):
    '''
    :param data: 目标性能数据
    :param n_clusters: 类簇数目
    :return: 类别标签
    '''
    ml_target = KMeans(n_clusters=n_clusters, random_state=0)
    ml_target.fit(np.array(data).reshape((-1, 1)))
    labels_of_target = ml_target.labels_
    cla_of_target = []
    for i in range(8):
        samples = [index for index, x in enumerate(labels_of_target) if x == i]
        cla_of_target.append(samples)
    return cla_of_target


# 基于全部特征数据的聚类散点图
def plotting_clusters_fea(n_clusters, data, data_label, label_index, index, columns, text):
    '''
    :param n_clusters: 聚类数目
    :param data: 所有特征数据
    :param data_label: 类簇标签
    :param label_index: 需要特殊标注的数据点的index
    :param index: 子图序号
    :param columns: 组合图中图注的列数
    :param text: 存储所有子图标题（x轴下方）的数组
    :return:
    '''
    plt.subplot(1, 4, index)
    markers = ['o', ',', 'v', '^', '<', '>', 'p', 'H']
    colors = ['blue', 'orange', 'red', 'purple', 'pink', 'green', 'grey', 'cyan']
    # labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'cluster6', 'cluster7', 'cluster8']
    labels = ['类簇1', '类簇2', '类簇3', '类簇4', '类簇5', '类簇6', '类簇7', '类簇8']
    for i in range(n_clusters):
        samples = data_label[i]
        x = data[samples, 0]
        y = data[samples, 1]
        plt.scatter(x, y, c=colors[i], s=50, label=labels[i], alpha=0.4, edgecolors='black', marker=markers[i])
        for j in range(len(x)):
            if samples[j] in label_index:
                plt.annotate(str(samples[j]), xy=(x[j], y[j]), xytext=(x[j] + 0.1, y[j] + 0.1))
                plt.scatter(x[j], y[j], c=colors[i], s=80, label=None, alpha=0.4, edgecolors='black')
    plt.legend(ncol=columns[index-1], loc='upper right', fontsize=13)
    # plt.text(2.6, 7, text[index - 1])
    # plt.text(-4.8, 7, text[index - 1])
    plt.ylim((-8, 8))
    plt.grid(True, linestyle='-.')
    plt.xlabel('PC1\n' + str(text[index - 1]), fontdict={'size': 18})
    plt.ylabel('PC2', fontdict={'size': 18})


# 基于目标性能数据的聚类散点图
def plotting_clusters_tar(n_clusters, data, data_label, index, columns, text):
    '''
    :param n_clusters: 聚类数目
    :param data:目标性能数据
    :param data_label:类簇标签
    :param index:子图序号
    :param columns:组合图中图注的列数
    :param text:存储所有子图标题（x轴下方）的数组
    :return:
    '''
    plt.subplot(1, 4, index)
    markers = ['o', ',', 'v', '^', '<', '>', 'p', 'H']
    colors = ['blue', 'orange', 'red', 'purple', 'pink', 'green', 'grey', 'cyan']
    # labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'cluster6', 'cluster7', 'cluster8']
    labels = ['类簇1', '类簇2', '类簇3', '类簇4', '类簇5', '类簇6', '类簇7', '类簇8']
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
    # plt.text(-3, 1.85, text[index - 1])
    # plt.xlabel('Sample Number')
    # plt.ylabel('Energy Barrier (eV)')
    plt.xlabel('样本编号\n' + text[index - 1] + '\n', fontdict={'size': 16})
    plt.ylabel('能量势垒 (eV)', fontdict={'size': 16})


if __name__ == "__main__":
    path = './Data/data_revised_85.xlsx'
    original_data, duplicate_data, columns = get_data(path, 'data')
    features = StandardScaler().fit_transform(duplicate_data[:, :-1])
    target = duplicate_data[:, -1]
    all = StandardScaler().fit_transform(duplicate_data)

    # 类簇数目分别同时取2、4、6、8
    classes_of_all = clustering_features(all, 2)
    classes_of_features = clustering_features(features, 2)
    classes_of_target = clustering_target(target, 2)

    # 写入聚类结果到表格
    '''
    wb_name = './Result/Revised_85/Clustering/Clusters.xlsx'
    create_workbook(wb_name)
    for i in range(len(classes_of_features)):
        save_to_excel_1d(classes_of_features[i], str(i) + 'th', wb_name, 'clusters_of_features', i + 1, 2)
        save_to_excel_1d(classes_of_target[i], str(i) + 'th', wb_name, 'clusters_of_target', i + 1, 2)
        save_to_excel_1d(classes_of_all[i], str(i) + 'th', wb_name, 'clusters_of_all', i + 1, 2)
    '''

    # 绘制全部特征数据/全部数据的聚类散点图
    '''
    low_dim_data = TSNE(n_components=2, random_state=80).fit_transform(all)
    # save_path = './Result/Revised_85/Clustering/Clustering_all_data.tif'
    save_path = './Result/Revised_85/Clustering/Clustering_all_features.tif'
    # text = ['(A) K = 8', '(B) K = 6', '(C) K = 4', '(D) K = 2']
    text = ['(a) K = 8', '(b) K = 6', '(c) K = 4', '(d) K = 2']
    legend_columns = [2, 2, 2, 1]
    plt.figure(figsize=(16, 3))
    for i, x in enumerate([8, 6, 4, 2]):
        # classes_of_all = clustering_features(all, x)
        classes_of_features = clustering_features(features, x)
        plotting_clusters_fea(x, low_dim_data, classes_of_features, [], i + 1, legend_columns, text)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.1, wspace=0.2)
    plt.margins(0, 0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=1)
    # plt.show()
    '''

    # 绘制目标性能数据的聚类散点图
    # '''
    save_path = './Result/Revised_85/Clustering/Clustering_target.tif'
    text = ['(1) K = 8', '(2) K = 6', '(3) K = 4', '(4) K = 2']
    legend_columns = [2, 2, 2, 2]
    plt.figure(figsize=(20, 3))
    for i, x in enumerate([8, 6, 4, 2]):
        classes_of_target = clustering_target(target, x)
        plotting_clusters_tar(x, target, classes_of_target, i+1, legend_columns, text)
    plt.subplots_adjust(top=1, bottom=0.01, hspace=0, wspace=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    # '''
