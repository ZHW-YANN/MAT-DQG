import json
import os
import sys
import re

import pandas as pd
from scipy.stats import skew, kurtosis

def normalize(series):
    """MinMax Normalization"""
    return (series - series.min()) / (series.max() - series.min())

def calculate_dataset_score(file_path, file_name, user_name):
    """Calculate the comprehensive score of the dataset"""
    try:
        # Verify the existence of the file
        if not os.path.exists(file_path):
            return json.dumps({"error": f"文件不存在: {file_path}"})

        # Loading the dataset
        data = pd.read_excel(file_path)

        # Check the data type of each column and discard columns containing character-type data.
        numeric_columns = []
        for column in data.columns:
            # Check whether the data type of the column is a numeric type (int or float)
            if pd.api.types.is_numeric_dtype(data[column]):
                numeric_columns.append(column)
            # else:
            #     print(f"Column '{column}' contains non-numeric data and will be dropped.")

        # Retain only columns of numeric type
        data = data[numeric_columns]

        # Calculate the skewness, kurtosis, range, and standard deviation for each column.
        # skewness = data.apply(skew)
        # kurt = data.apply(kurtosis)
        data_range = data.max() - data.min()
        std_dev = data.std()

        # Normalization of each indicator
        # skewness_norm = normalize(skewness)
        # kurt_norm = normalize(kurt)
        range_norm = normalize(data_range)
        std_dev_norm = normalize(std_dev)

        # weights
        weights = {
            'skewness': 0.25,
            'kurtosis': 0.25,
            'range': 0.5,
            'std_dev': 0.5
        }

        # Calculate the comprehensive score for each column
        column_scores = (
            # weights['skewness'] * skewness_norm +
            # weights['kurtosis'] * kurt_norm +
            weights['range'] * range_norm +
            weights['std_dev'] * std_dev_norm
        )

        # Calculate the comprehensive score of the entire dataset
        dataset_score = column_scores.mean()


        process_data = pd.DataFrame({
            '列名': data.columns,
            # '偏度': skewness_norm,
            # '峰度': kurt_norm,
            '范围': range_norm,
            '标准差': std_dev_norm,
            'Score': column_scores
        })

        dir_name, file_name = os.path.split(file_path)
        base_name, ext_name = os.path.splitext(file_name)
        new_file_name = f"evaluate_{base_name}{ext_name}"

        reg_ex_1 = re.compile(re.escape(os.path.sep))
        list_of_token_1 = reg_ex_1.split(file_path)
        if not list_of_token_1[0].endswith(os.path.sep):
            list_of_token_1[0] += os.path.sep
        relevant_part = list_of_token_1[:5]
        dir_name = os.path.join(*relevant_part)


        standardability_dir = os.path.join(dir_name, 'standardability')
        if not os.path.exists(standardability_dir):
            os.makedirs(standardability_dir)


        new_file_path = os.path.join(standardability_dir, new_file_name)

        process_data.to_excel(new_file_path, index=False)

        process_data_json = process_data.to_dict(orient='records')

        result = {
            "datasetScore": dataset_score,
            "processData": process_data_json
        }


        return json.dumps(result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":

    file_path = sys.argv[1]
    file_name = sys.argv[2]
    user_name = sys.argv[3]

    result = calculate_dataset_score(file_path, file_name, user_name)
    print(result)