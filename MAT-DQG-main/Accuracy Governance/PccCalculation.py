import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import pearsonr
from Printing import create_directory, create_workbook, save_to_excel_1d

if __name__ == '__main__':
    data = pd.DataFrame(pd.read_excel('./Data/data_revised_85.xlsx', sheet_name='data')).values
    feature_names = np.array(pd.DataFrame(pd.read_excel('./Data/data_revised_85.xlsx', sheet_name='data')).columns[:-1])
    features = preprocessing.StandardScaler().fit_transform(np.array(data[:, :-1]))
    features = np.array(features).transpose()
    target = np.array(data[:, -1])

    coors = []
    for x in features:
        coors.append(pearsonr(x, target)[0])

    wb = './Result/Revised_85/FeatureSelection/PCC.xlsx'
    sheet = 'data'
    create_directory(wb)
    create_workbook(wb)
    save_to_excel_1d(feature_names, 'Feature Name', wb, sheet, 1, 2)
    save_to_excel_1d(coors, 'PCC', wb, sheet, 2, 2)
