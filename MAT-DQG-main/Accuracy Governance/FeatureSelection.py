import numpy as np
import pandas as pd
from Printing import create_directory, create_workbook, save_to_excel_1d
from Training import model_selection

if __name__ == '__main__':
    wb_data = './Data/data_revised_85.xlsx'
    wb_splitting = './Data/data_split_85.xlsx'
    data = pd.read_excel(wb_data, sheet_name='data').values
    train_indices = pd.read_excel(wb_splitting, sheet_name='Train').values.T
    test_indices = pd.read_excel(wb_splitting, sheet_name='Test').values.T
    pcc_values = pd.DataFrame(pd.read_excel('./Result/Revised_85/FeatureSelection/PCC.xlsx', sheet_name='data')).values[:, -1]
    thresholds = np.arange(0.10, 0.70, 0.02)

    wb = './Result/Revised_85/FeatureSelection/ThresholdSensitivity.xlsx'
    sheet = 'data'
    create_directory(wb)
    create_workbook(wb)

    for i, t in enumerate(thresholds):
        feature = []
        for index, x in enumerate(pcc_values):
            if np.abs(x) > t:
                feature.append(index)
        save_to_excel_1d(feature, str(i) + ' th', wb, sheet, i+1, 2)

        feature.append(45)
        new_data = data[:, feature]
        models = ['MLR', 'SVR', 'KNN', 'GPR', 'Ridge', 'LASSO', 'RF']
        path = 'Result/Revised_85/FeatureSelection/Threshold_' + str(i)
        create_directory(path)
        model_selection(new_data, train_indices, test_indices, models, path, 10)





