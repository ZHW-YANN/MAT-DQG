from get_model.ModelsCV import predictors
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from fancyimpute import IterativeImputer
import numpy as np
import json
import os
from json import JSONEncoder
import re
import pandas as pd
from flask import Flask, jsonify, request, make_response, abort, send_file
from flask_cors import CORS
from dataBase.add import insert_record,save_data_info

app = Flask(__name__)

CORS(app, supports_credentials=True)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:7081"}})

# Completeness
@app.route('/completeness/report', methods=['GET'])
def completeness_report():
    file_path = request.args.get('filePath')
    file_name = request.args.get('fileName')
    # 分割路径和文件名
    # dir_name, file_name = os.path.split(file_path)
    # base_name, ext_name = os.path.splitext(file_name)

    reg_ex_1 = re.compile(re.escape(os.path.sep))
    list_of_token_1 = reg_ex_1.split(file_path)
    if not list_of_token_1[0].endswith(os.path.sep):
        list_of_token_1[0] += os.path.sep
    relevant_part = list_of_token_1[:5]
    dir_name = os.path.join(*relevant_part)

    completeness_dir = os.path.join(dir_name, 'completeness')
    if not os.path.exists(completeness_dir):
        os.makedirs(completeness_dir)


    df = pd.read_excel(file_path,sheet_name='data')
    property_column = df['Property'] if 'Property' in df.columns else None
    #
    if 'Property' in df.columns:
        df.drop('Property', axis=1, inplace=True)
    #
    saved_columns = {}
    for column in df.columns:
        if df[column].dtype == object:
            saved_columns[column] = df[column].copy()
    def remove_units(value):
        if pd.isnull(value):
            return value
        return pd.to_numeric(re.sub(r'[^\d.]+', '', str(value)), errors='coerce')

    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].apply(remove_units)
    # Convert to the appropriate numerical type
    df = df.apply(pd.to_numeric, errors='coerce')

    # Calculate the missing rate
    missing_rate = df.isnull().mean()


    if property_column is not None:
        df['Property'] = property_column

    for column, data in saved_columns.items():
        df[column] = data
    data_origin = {col: df[col].fillna('').tolist() for col in df.columns}
    columns = df.columns.tolist()


    missing_rate_dict = missing_rate.to_dict()


    response_data = {
        'message': 'File processed successfully',
        'columns': columns,
        'data': data_origin,
        'missing_rate': missing_rate_dict
    }


    return jsonify(response_data)

@app.route('/completeness/medicine', methods=['GET'])
def completeness_medicine():
    file_path = request.args.get('filePath')
    file_name = request.args.get('fileName')
    userName = request.args.get('userName')
    # 分割路径和文件名
    # dir_name, file_name = os.path.split(file_path)
    # base_name, ext_name = os.path.splitext(file_name)
    reg_ex_1 = re.compile(re.escape(os.path.sep))
    list_of_token_1 = reg_ex_1.split(file_path)
    if not list_of_token_1[0].endswith(os.path.sep):
        list_of_token_1[0] += os.path.sep
    relevant_part = list_of_token_1[:5]
    dir_name = os.path.join(*relevant_part)


    completeness_dir = os.path.join(dir_name, 'completeness')
    if not os.path.exists(completeness_dir):
        os.makedirs(completeness_dir)


    df = pd.read_excel(file_path,sheet_name='data')
    property_column = df['Property'] if 'Property' in df.columns else None

    if 'Property' in df.columns:
        df.drop('Property', axis=1, inplace=True)


    saved_columns = {}
    for column in df.columns:
        if df[column].dtype == object:
            saved_columns[column] = df[column].copy()

    def remove_units(value):
        if pd.isnull(value):
            return value
        return pd.to_numeric(re.sub(r'[^\d.]+', '', str(value)), errors='coerce')

    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].apply(remove_units)

    df = df.apply(pd.to_numeric, errors='coerce')

    all_data = pd.DataFrame(df)
    target_c = all_data.columns[-1]
    column = all_data.columns


    imputer = IterativeImputer()

    data_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    data_imputed1 = data_imputed.copy()
    data_complete = {col: data_imputed[col].fillna('').tolist() for col in data_imputed.columns}
    new_file = f"D3_{file_name}"
    new_file_path = os.path.join(completeness_dir, new_file)


    if property_column is not None:
        data_imputed1['Property'] = property_column

    for column, data in saved_columns.items():
        data_imputed1[column] = data
    complete_df = data_imputed1.copy()
    complete_df.insert(0, 'No', range(len(complete_df)))

    with pd.ExcelWriter(new_file_path, engine='xlsxwriter') as writer:
        data_imputed1.to_excel(writer, index=False, sheet_name='data')
        complete_df.to_excel(writer, index=False,
                             sheet_name='complete information')

    insert_record(userName, 'Completeness ', 'Impute missing data', new_file_path, file_name, new_file, 1)

    results = []
    method_list = ['KNN', 'MLR']
    for m in method_list:

        labeled_data = df[df[target_c].notnull()]
        labeled_X = labeled_data.drop(target_c, axis=1).reset_index(drop=True)  # 去除含缺失值的列后重新排序
        labeled_Y = labeled_data[target_c].reset_index(drop=True)  # 含缺失值的列就是Y

        train_X, test_X, train_Y, test_Y = train_test_split(labeled_X, labeled_Y, test_size=0.15, random_state=42)
        model = predictors(train_X, train_Y, kf=10, type=m)[1]
        model.fit(train_X, train_Y)
        predict_y = model.predict(test_X)
        test_rmse1 = np.sqrt(mean_squared_error(test_Y, predict_y))
        test_r2_1 = r2_score(test_Y, predict_y)

        train_X_imputed, test_X_imputed, train_Y_imputed, test_Y_imputed = train_test_split(data_imputed.drop(target_c, axis=1), data_imputed[target_c], test_size=0.15, random_state=42)
        model_new = predictors(train_X_imputed, train_Y_imputed, kf=10, type=m)[1]
        model_new.fit(train_X_imputed, train_Y_imputed)
        predict_y_new = model_new.predict(test_X_imputed)
        test_rmse2 = np.sqrt(mean_squared_error(test_Y_imputed, predict_y_new))
        test_r2_2 = r2_score(test_Y_imputed, predict_y_new)

        results.append({
            'method': m,
            'original_rmse': test_rmse1,
            'imputed_rmse': test_rmse2,
            'original_r2': test_r2_1,
            'imputed_r2': test_r2_2
        })


    response_data = {
        'message': 'File processed successfully',
        'data': json.dumps(data_complete),
        'results': results
    }


    return jsonify(response_data)