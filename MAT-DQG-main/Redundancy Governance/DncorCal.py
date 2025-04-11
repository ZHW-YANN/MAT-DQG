import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import spearmanr
from minepy import MINE
from Printting import create_directory, create_workbook, save_to_excel_1d
import copy


class DataDrivenNCOR(object):
    def __init__(self, data, KD_NCOR, threshold, k, wb):
        self.data = data
        self.duplicates = self.eliminate_duplicate(KD_NCOR)
        self.threshold = threshold
        self.k = k
        self.wb = wb
        self.n_samples, self.n_features, self.target, self.features, self.feature_names = self.load_data()

    # 基于KNCOR生成高度相关特征对
    def eliminate_duplicate(self, KD_NCOR):
        duplicates = []
        for x in KD_NCOR:
            for index_1, y in enumerate(x):
                for z in y:
                    for index_2 in range(index_1 + 1, len(x)):
                        for p in x[index_2]:
                            duplicates.append([z, p])
        return duplicates

    def load_data(self):
        data = pd.DataFrame(self.data)
        features = preprocessing.MinMaxScaler().fit_transform(np.array(data.values[:, :-1]))
        target = np.array(data.values[:, -1])
        feature_names = np.array(pd.DataFrame(pd.read_excel(self.wb, sheet_name='data')).columns[:-1])
        n_samples = data.shape[0]
        n_features = data.shape[1] - 1

        return n_samples, n_features, target, features, feature_names

    # 利用MIC计算特征之间的相关性
    def evaluate_mic(self, wb, sheet):
        name1s = []
        name2s = []
        coors = []
        feature_couples = []

        mine = MINE(alpha=0.6, c=15)
        features_indices = range(self.n_features)
        for i in range(self.n_features):
            feature_name1 = self.feature_names[i]
            for j in range(i + 1, self.n_features):
                feature_name2 = self.feature_names[j]
                name1s.append(feature_name1 + '(' + str(features_indices[i]) + ')')
                name2s.append(feature_name2 + '(' + str(features_indices[j]) + ')')
                feature_couples.append((features_indices[i], features_indices[j]))
                mine.compute_score(self.features[:, i], self.features[:, j])
                if [i, j] in self.duplicates:   # 保证识别出的DNCOR不与KNCOR重复
                    coor = 0
                else:
                    coor = mine.mic()
                coors.append(coor)
        sort_indices = np.argsort(coors)
        compared_indices = list(filter(lambda x: coors[x] >= self.threshold, sort_indices))
        compared_features = list(
            filter(None, [value if index in compared_indices else None for index, value in enumerate(feature_couples)]))

        create_directory(wb)
        create_workbook(wb)
        save_to_excel_1d(name1s, 'Feature1', wb, sheet, 1, 2)
        save_to_excel_1d(name2s, 'Feature2', wb, sheet, 2, 2)
        save_to_excel_1d(coors, 'MIC Value', wb, sheet, 3, 2)
        return compared_features, coors

    # 利用SCC计算特征之间的相关性
    def evaluate_scc(self, wb, sheet):
        name1s = []
        name2s = []
        coors = []
        abs_coors = []
        feature_couples = []

        features_indices = range(self.n_features)
        for i in range(self.n_features):
            feature_name1 = self.feature_names[i]
            for j in range(i + 1, self.n_features):
                feature_name2 = self.feature_names[j]
                name1s.append(feature_name1 + '(' + str(features_indices[i]) + ')')
                name2s.append(feature_name2 + '(' + str(features_indices[j]) + ')')
                feature_couples.append((features_indices[i], features_indices[j]))
                if [i, j] in self.duplicates:
                    coor = 0
                else:
                    coor = spearmanr(self.features[:, i], self.features[:, j])[0]
                coors.append(coor)
                abs_coors.append(abs(coor))
        sort_indices = np.argsort(abs_coors)
        compared_indices = list(filter(lambda x: abs_coors[x] >= self.threshold, sort_indices))
        compared_features = list(
            filter(None, [value if index in compared_indices else None for index, value in enumerate(feature_couples)]))

        create_directory(wb)
        create_workbook(wb)
        save_to_excel_1d(name1s, 'Feature1', wb, sheet, 1, 2)
        save_to_excel_1d(name2s, 'Feature2', wb, sheet, 2, 2)
        save_to_excel_1d(coors, 'PCC Value', wb, sheet, 3, 2)
        return compared_features, abs_coors

    # 利用SCC计算特征之间的相关性
    def all_filter(self, wb):
        if self.n_samples < self.k:
            compared_features, abs_coors = self.evaluate_mic(wb, 'MIC')
            print('MIC: ')
        else:
            compared_features, abs_coors = self.evaluate_scc(wb, 'SCC')
            print('SCC: ')
        print('Compared features：', compared_features)
        redundant_paris = copy.deepcopy(compared_features)

        compared_features = np.array(list(set([tuple(t) for t in compared_features])))
        compared_features = compared_features.tolist()

        # 基于高度相关特征对生成多元高度相关特征
        absolute_compared_features = []
        for x in list(compared_features):
            temp = list(copy.deepcopy(x))
            for i in range(self.n_features):
                flag = False
                for y in temp:
                    if [y, i] in compared_features or [i, y] in compared_features:
                        flag = True
                    else:
                        flag = False
                        break
                if flag:
                    temp.append(i)
                    temp.sort()
            if temp not in absolute_compared_features:
                absolute_compared_features.append(temp)
        return absolute_compared_features, redundant_paris

    # 多元高度相关特征的存储
    def feature_selection(self, path, wb_name, sheet_name):
        result, redundant_paris = self.all_filter(path)
        result_str = copy.deepcopy(result)
        i = 0
        for group in result:
            j = 0
            for index in group:
                result_str[i][j] = self.feature_names[index]
                j += 1
            i += 1
        print('Data-driven NCOR:', result)
        print('Data-driven NCOR:', result_str)
        create_directory(wb_name)
        create_workbook(wb_name)
        save_to_excel_1d(result, 'Data-driven NCOR Indices', wb_name, sheet_name, 1, 2)
        save_to_excel_1d(result_str, 'Data-driven NCOR Names', wb_name, sheet_name, 2, 2)
        return result, redundant_paris  # 多元高度相关特征，高度相关的特征对


# 将高度相关特征转换为NCOR
def cal_NCOR(old_relevance_rules):
    relevance_rules = []
    for index, value in enumerate(old_relevance_rules):
        temp = []
        for x in value:
            temp.append([x])
        relevance_rules.append(temp)
    return relevance_rules


if __name__ == '__main__':
    data = pd.DataFrame(pd.read_excel('../DataInput/original_data.xlsx', sheet_name='data')).values
    DDNCOR = DataDrivenNCOR(data, 0.9, 200, '../DataInput/original_data.xlsx')
    path = '../ResultsForValidation/Data_Driven_NCOR/correlation-0.xlsx'
    result_wb = '../ResultsForValidation/Data_Driven_NCOR/NCOR-0.xlsx'
    result_sheet = 'Data-driven NCOR'
    DD_NCOR = cal_NCOR(DDNCOR.feature_selection(path, result_wb, result_sheet))