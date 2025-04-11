import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams["font.sans-serif"] = ["Bahnschrift"]
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.unicode_minus"] = False

if __name__ == "__main__":
    # all_feature_sets = pd.read_excel('./Result/Removed_85/FeatureSelection/ThresholdSensitivity.xlsx', sheet_name='data').values
    all_feature_sets = pd.read_excel('./Result/Revised_85/FeatureSelection/ThresholdSensitivity.xlsx',sheet_name='data').values

    features = []
    for i in range(np.array(all_feature_sets).shape[1]):
        temp = all_feature_sets[:, i][np.logical_not(np.isnan(all_feature_sets[:, i]))]
        features.append(np.array(temp).astype('int8'))
    length = np.full(30, 44) - [len(x) for x in features]
    x = np.arange(0.1, 0.70, 0.02)

    # models = ['LASSO', 'RF']
    models = ['Ridge', 'SVR', 'KNN', 'GPR']
    all_rmses, all_deviations = [], []
    for m in models:
        avg_rmses, deviations = [], []
        for j in range(len(x)):
            # rmses = pd.read_excel('./Result/Removed_85/FeatureSelection/Threshold_' + str(j) + '/' + m + '.xlsx', sheet_name='performance').values[:, 0]
            rmses = pd.read_excel('./Result/Revised_85/FeatureSelection/Threshold_' + str(j) + '/' + m + '.xlsx',sheet_name='performance').values[:, 0]
            avg_rmses.append(np.average(rmses))
            deviations.append(np.std(rmses))
        all_rmses.append(avg_rmses)
        all_deviations.append(deviations)

    colors = ['palevioletred', 'tab:orange', 'tab:green', 'tab:purple', '#8C6161', '#193441']
    markers = ['^', 'v', '.', '*']
    fig, left_axis = plt.subplots(figsize=(7, 6))
    right_axis = left_axis.twinx()
    a2, = right_axis.plot(x, length, color='mediumblue', marker='D', alpha=0.6)
    for r, d, c, mar, m in zip(all_rmses, all_deviations, colors, markers, models):
        if m in models:
            left_axis.plot(x, r, color=c, marker=mar, label=m, zorder=2)
            min = np.min(r)
            min_index = x[np.argmin(r)]
            left_axis.plot([0.08, min_index], [min, min], color='silver', dashes=[0.5, 1], zorder=1)
            left_axis.plot([min_index, min_index], [0, min], color='silver', dashes=[0.5, 1], zorder=1)
            left_axis.scatter([min_index], [min], color='red', marker=mar, zorder=3)
            if m == 'SVR':
                left_axis.text(min_index - 0.020, min + 0.003, "%.4f" % min, zorder=3)
            else:
                left_axis.text(min_index - 0.015, min + 0.002, "%.4f" % min, zorder=3)

            pot = length[np.argmin(r)]
            right_axis.plot([min_index, min_index], [-25, pot], color='silver', dashes=[0.5, 1])
            right_axis.scatter([min_index], [pot], color='red', marker='D', zorder=3)
            right_axis.text(min_index + 0.005, pot-3, "%d" % pot)

    left_axis.set_ylim(0.035, 0.15)
    left_axis.set_ylabel('Average RMSE (eV)')
    left_axis.set_xlabel('Threshold')
    left_axis.legend(ncol=1, loc='upper left')
    # right_axis.text(0.51, 33, '(a) Semi-revised data2', font={'size': 11})
    right_axis.text(0.55, 33, '(b) Revised data', font={'size': 11})
    right_axis.spines['right'].set_color('mediumblue')
    right_axis.xaxis.label.set_color('mediumblue')
    right_axis.tick_params(axis='y', colors='mediumblue')
    right_axis.set_ylabel('Number of Removed Descriptors', color='mediumblue')


    xticks = [0.08]
    xticks.extend(x)
    xticks.append(0.70)
    plt.xlim(0.08, 0.70)
    plt.xticks(np.arange(np.min(xticks), np.max(xticks), 0.05))
    plt.ylim(-30, 36)
    # plt.savefig('./Result/Removed_85/FeatureSelection/Figures/Sensitivity.tif')
    plt.savefig('./Result/Revised_85/FeatureSelection/Figures/Sensitivity.tif')
    plt.show()
