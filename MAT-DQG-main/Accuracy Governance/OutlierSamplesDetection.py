import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from OutlierValuesDetection import get_data, create_directory
from Printing import create_workbook, save_to_excel_1d, save_to_excel_2d
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD


def detection_Others(data, method):
    if method == 'KNN':
        ml = KNN(n_neighbors=8, radius=0.1)
    elif method == 'LOF':
        ml = LOF()
    elif method == 'OCSVM':
        ml = OCSVM()
    else:
        ml = MCD()
    ml.fit(data)
    y_pred = ml.labels_
    outliers_indices = [index for index, x in enumerate(y_pred) if x == 1]

    wb_name = 'Result/Revised_85/Outliers/' + method + '_Samples.xlsx'
    create_workbook(wb_name)
    save_to_excel_1d(outliers_indices, 'Outliers', wb_name, 'result', 1, 2)

    print(method + ' Detection: ', outliers_indices)


def detection_IForest(times, data):
    outlier_indices = []
    for i in range(times):
        ml = IForest(contamination=0.1, random_state=i)
        ml.fit(data)
        y_pred = ml.labels_                 # binary labels (0: inliers, 1: outliers)
        outliers_temp = [index for index, x in enumerate(y_pred) if x == 1]
        outlier_indices.append(outliers_temp)

    wb_name = 'Result/Revised_85/Outliers/IF_Samples.xlsx'
    create_workbook(wb_name)
    save_to_excel_2d(np.array(outlier_indices).T, [str(x) + ' th' for x in range(times)], wb_name, 'result', 1, 2)
    outlier_indices_count = Counter(np.array(outlier_indices).flatten())

    print('IForset Detection: ', outlier_indices_count)


if __name__ == '__main__':
    path = './Data/data_revised_85.xlsx'
    original_data, duplicate_data, columns = get_data(path, 'data')
    processing_data = StandardScaler().fit_transform(duplicate_data)

    # detection_IForest(10, processing_data)
    detection_Others(processing_data, 'LOF')

