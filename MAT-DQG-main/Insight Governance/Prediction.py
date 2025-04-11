import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import sklearn.preprocessing as prep
import pandas as pd
import numpy as np
import os
from get_model.BayesOptModels import predictors
from get_model.utils import load_data, get_data, create_wb, save_to_excel_y, save_to_excel_1d, save_to_excel_2d, create_directory
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pickle
import joblib


dataset_list = ['dataset8']
# dataset_list = os.listdir('./input_data')

if __name__ == '__main__':
    # Loading data
    for data_name in dataset_list:

        load_data_wb = './input_data/' + data_name + '.xlsx'
        data_ori = load_data(load_data_wb, sheet_name='data')  # oringial dataset
        data_rev = load_data(load_data_wb, sheet_name='data-rev') # revised dataset

        # Creating directory
        path_write = './output_data/' + data_name

        isExists = os.path.exists(path_write)
        if not isExists:
            os.mkdir(path_write)
        all_data_ori = pd.DataFrame(data_ori)


        # Setting methods
        model_list = ['LASSO', 'Ridge', 'SVR',  'KNN', 'RF', 'GPR', 'MLR']
        root_path = './output_data/' + data_name

        wb_name = root_path + '/prediction-CV10.xlsx'
        # wb_name = root_path + '/prediction-CV10-rev.xlsx'

        create_wb(wb_name)
        model_name, rmse_train, rmse, mape, r2, pcc_train, pcc_test = [], [], [], [], [], [], []
        for m in model_list:
            print("method: ", m)
            print("dataset:", data_name)

            train_X, test_X, train_Y, test_Y = train_test_split(all_data_ori.iloc[:, :-1],
                                                                all_data_ori.iloc[:, -1], test_size=0.2,
                                                                random_state=999, shuffle=True)
            scaler = prep.MinMaxScaler()
            pro_train_x = scaler.fit_transform(train_X)
            pro_test_x = scaler.transform(test_X)

            if (os.path.exists(root_path + '/Model_'+str(m) +'.pkl')):
                print('Model exists!')
                'leverage the model trained with 10-fold cross-validation'
                model = joblib.load(open(root_path + '/Model_' + str(m) + '.pkl', 'rb'), mmap_mode='r')
            else:
                'train the model with 10-fold cross-validation'
                model = predictors(pro_train_x, train_Y, kf=10, type=m, random_state=999)
                joblib.dump(model, root_path + f'/Model_' + str(m) + '.pkl')

            best_model = model.fit(pro_train_x, train_Y)
            predict_org_y = best_model.predict(pro_train_x)
            predict_y = best_model.predict(pro_test_x)

            corr_train = pearsonr(train_Y, predict_org_y)  # 训练集PCC
            corr_test = pearsonr(test_Y, predict_y)  # 测试集PCC
            train_rmse = np.sqrt(mean_squared_error(train_Y, predict_org_y))
            test_rmse = np.sqrt(mean_squared_error(test_Y, predict_y))
            test_mape = sum(abs((predict_y - test_Y) / test_Y) / len(test_Y))
            test_r2 = r2_score(test_Y, predict_y)

            print('ML model: ', model_name)
            print('CV RMSE: %.5f    Test RMSE: %.5f    Test MAPE: %.5f    Test R2: %.5f' % (train_rmse, test_rmse, test_mape, test_r2))
            print('Train PCC: %.5f Test PCC: %.5f' % (corr_train[0], corr_test[0]))

            model_name.append(m)
            rmse_train.append(train_rmse)
            rmse.append(test_rmse)
            mape.append(test_mape)
            r2.append(test_r2)
            pcc_train.append(corr_train[0])
            pcc_test.append(corr_test[0])

            sheet_name = 'result'
            save_to_excel_1d(model_name, 'Model', wb_name, sheet_name, 1, 2)
            save_to_excel_1d(rmse_train, 'CV RMSE', wb_name, sheet_name, 2, 2)
            save_to_excel_1d(rmse, 'Test RMSE', wb_name, sheet_name, 3, 2)
            save_to_excel_1d(mape, 'Test MAPE', wb_name, sheet_name, 4, 2)
            save_to_excel_1d(r2, 'Test R2', wb_name, sheet_name, 5, 2)
            save_to_excel_1d(pcc_train, 'Train PCC', wb_name, sheet_name, 6, 2)
            save_to_excel_1d(pcc_test, 'Test PCC', wb_name, sheet_name, 7, 2)
