import re
import os
import copy
import numpy as np
import pandas as pd
import sympy
from scipy import stats
from Printing import create_directory, create_workbook, save_to_excel_1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

mpl.rcParams["font.sans-serif"] = ["Bahnschrift"]
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.unicode_minus"] = False


def get_data(path, sheet_name):
    data = pd.read_excel(path, sheet_name=sheet_name)
    columns = data.columns
    original_data = np.array(data)
    duplicate_data = copy.deepcopy(original_data[:, ])
    return original_data, duplicate_data, columns


# Quartile distribution method
def quartile_outlier(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    threshold_1 = Q1 - outlier_step
    threshold_2 = Q3 + outlier_step
    outlier_indices, outliers = [], []

    for i in range(len(data)):
        if data[i] < threshold_1 or data[i] > threshold_2:
            outliers.append(data[i])
            outlier_indices.append(i)

    return outlier_indices, outliers


if __name__ == '__main__':
    path = './Data/data_revised_90.xlsx'
    method = 'quartile_distribution'
    original_data, duplicate_data, columns = get_data(path, 'data')
    features = pd.DataFrame(pd.read_excel(path, sheet_name='data')).columns
    all_outliers_indices = []
    t = 1

    wb = 'Result/Revised_90/Outliers/Values.xlsx'
    # create_directory(wb)
    # create_workbook(wb)
    # for i in range(len(original_data) - 1):
    #     outlier_indices, outliers = quartile_outlier(duplicate_data[:, i])
    #     save_to_excel_1d(outlier_indices, str(i) + ' th', wb, 'index', i + 1, 2)
    #     save_to_excel_1d(outliers, str(i) + ' th', wb, 'value', i + 1, 2)

    # '''
    for i in [0, 2, 40]:
        outlier_indices, outliers = quartile_outlier(duplicate_data[:, i])
        all_outliers_indices.append(outlier_indices)
        if i == 0:
            plt.figure(figsize=(20, 5))
            t = 1
        if i in [0, 2, 40] and i != 0:
            t = t + 1
        if i in [0, 2, 40]:
            titles = ["(a) ", "(b) ", "(c) "]
            plt.subplot(1, 3, t)
            plt.title(titles[t - 1] + '%s' % features[i], fontdict={'size': 15})
            plt.scatter(range(len(duplicate_data[:, i])), duplicate_data[:, i], c='#3E606F', s=50, edgecolors='k', alpha=0.85, marker='o')
            plt.scatter(outlier_indices, outliers, c='#8C6161', s=50, edgecolors='k', alpha=0.85, marker=',')
            x_major_locator = MultipleLocator(5)
            ax = plt.gca()
            ax.xaxis.set_major_locator(x_major_locator)
            plt.grid(color='grey', linestyle='--')

            plt.xlabel('Sample Number')
            plt.ylabel(features[i])
    plt.savefig('Result/Revised_90/Plotting/OutlierValues.tif')
    # '''
