import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Bahnschrift"]
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.unicode_minus"] = False

if __name__ == '__main__':
    data = pd.DataFrame(pd.read_excel('./Data/data_revised_90.xlsx'))
    duplicate_data = data.values
    columns = data.columns
    processing_data = StandardScaler().fit_transform(duplicate_data)
    color = cm.get_cmap('viridis', len(columns))

    df = pd.DataFrame(processing_data)
    plt.figure(figsize=(16, 7.5))
    f = df.boxplot(sym='o',  # 异常点形状，参考marker
                   vert=True,  # 是否垂直
                   whis=1.5,  # IQR，默认1.5，也可以设置区间比如[5,95]，代表强制上下边缘为数据95%和5%位置
                   patch_artist=True,  # 上下四分位框内是否填充，True为填充
                   meanline=False, showmeans=True,  # 是否有均值线及其形状
                   showbox=True,  # 是否显示箱线
                   showcaps=True,  # 是否显示边缘线
                   showfliers=True,  # 是否显示异常值
                   notch=False,  # 中间箱体是否缺口
                   return_type='dict',  # 返回类型为字典
                   )
    plt.xlabel('Number', fontsize=15)
    plt.ylabel('Standardized Value', fontsize=15)
    plt.grid(False)
    plt.xticks(np.arange(1, len(columns) + 1), np.arange(1, len(columns) + 1))
    i = 0
    for box in f['boxes']:
        box.set(color='black', linewidth=1)  # 箱体边框颜色
        box.set(facecolor=color.colors[i], alpha=0.5)  # 箱体内部填充颜色
        i += 1
    for whisker in f['whiskers']:
        whisker.set(color='k', linewidth=0.8, linestyle='--')
    for cap in f['caps']:
        cap.set(color='black', linewidth=1)
    for median in f['medians']:
        median.set(color='black', linewidth=1)
    for mean in f['means']:
        mean.set(marker='+', markerfacecolor='dimgrey', markeredgecolor='dimgrey', markersize=4)
    for flier in f['fliers']:
        flier.set(marker='o', markerfacecolor='lightseagreen', markeredgecolor='lightseagreen', markersize=2)
    plt.tight_layout()
    plt.savefig('./Result/Revised_90/Plotting/BoxPlot.tif')
    plt.show()
