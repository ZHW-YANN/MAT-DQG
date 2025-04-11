# normalization_methods.py
import numpy as np
import pandas as pd
#import re
from sklearn.preprocessing import MinMaxScaler



def text_encoding(df):
    encoded_df = df.copy()
    encoded_columns = {}  # Store processed column names and feature information

    for column in df.columns:

        if df[column].apply(
                lambda x: not any(char.isdigit() for char in str(x)) and any(char.isalpha() for char in str(x))).any():
            num_features = len(df[column].unique())
            encoded_df[column], labels = pd.factorize(df[column])
            encoded_columns[column] = {'processed_column_name': column,
                                       'num_features': num_features,
                                       'labels': {original_value: encoded_value for original_value, encoded_value in zip(labels.tolist(), encoded_df[column].unique())}}
    return encoded_df, encoded_columns

def ch_unit(df):
    modified_df = df.copy()
    rows_with_non_h_units = pd.DataFrame(columns=['Row Number', 'Column Name', 'Original Data', 'Modified Data'])
    def convert_and_replace(value):
        if "min" in value:
            return str(float(value.replace("min", "")) / 60) + "h"
        elif "KPa" in value:
            return str(float(value.replace("KPa", "")) / 1000) + "MPa"
        elif "K" in value:
            return str(float(value.replace("K", "")) - 273) + "℃"
        else:
            return value

    for column in df.columns:
        if df[column].apply(
                lambda x: any(char.isdigit() for char in str(x)) and any(char.isalpha() for char in str(x))).any():
            unit = df[column].astype(str).str.extract(r'([a-zA-Z]+)')[0]
            if "h" in unit.unique():
                non_h_index = unit != "h"
                non_h_data = df.loc[non_h_index, column]
                modified_df.loc[non_h_index, column] = non_h_data.apply(convert_and_replace)
                modified_data = non_h_data.apply(convert_and_replace)
                modified_rows = df.loc[non_h_index].copy()
                modified_rows['Row Number'] = modified_rows.index + 1
                modified_rows['Original Data'] = non_h_data
                modified_rows['Modified Data'] = modified_data
                modified_rows['Column Name'] = column
                modified_rows = modified_rows[['Row Number', 'Column Name', 'Original Data', 'Modified Data']]
                rows_with_non_h_units = pd.concat([rows_with_non_h_units, modified_rows])
            elif "MPa" in unit.unique():
                non_h_index = unit != "MPa"
                non_h_data = df.loc[non_h_index, column]
                modified_df.loc[non_h_index, column] = non_h_data.apply(convert_and_replace)
                modified_data = non_h_data.apply(convert_and_replace)
                modified_rows = df.loc[non_h_index].copy()
                modified_rows['Row Number'] = modified_rows.index + 1
                modified_rows['Original Data'] = non_h_data
                modified_rows['Modified Data'] = modified_data
                modified_rows['Column Name'] = column
                modified_rows = modified_rows[['Row Number', 'Column Name', 'Original Data', 'Modified Data']]
                rows_with_non_h_units = pd.concat([rows_with_non_h_units, modified_rows])
            elif "K" in unit.unique():
                non_h_index = unit == "K"
                non_h_data = df.loc[non_h_index, column]
                modified_df.loc[non_h_index, column] = non_h_data.apply(convert_and_replace)
                modified_data = non_h_data.apply(convert_and_replace)
                modified_rows = df.loc[non_h_index].copy()
                modified_rows['Row Number'] = modified_rows.index + 1
                modified_rows['Original Data'] = non_h_data
                modified_rows['Modified Data'] = modified_data
                modified_rows['Column Name'] = column
                modified_rows = modified_rows[['Row Number', 'Column Name', 'Original Data', 'Modified Data']]
                rows_with_non_h_units = pd.concat([rows_with_non_h_units, modified_rows])
    return modified_df, rows_with_non_h_units

def range_process(df):
    data = df.copy()
    column_means = data[data.columns[1:]].mean()

    max_values = {}
    for column_name, mean_value in column_means.items():
        max_distance = data[column_name].sub(mean_value).abs().nlargest(10)
        max_values[column_name] = data.loc[max_distance.index, column_name].to_dict()

    charts_data = []
    table_data = []
    for column_name, column_data in data.iteritems():
        line_chart = {
            'name': column_name,
            'type': 'line',
            'data': column_data.values.tolist()
        }
        charts_data.append(line_chart)

        mean_diff = column_data.iloc[1:] - column_means[column_name]
        bar_chart = {
            'name': column_name,
            'type': 'bar',
            'data': mean_diff.abs().values.tolist()
        }
        charts_data.append(bar_chart)

        table_data.append({
            'Column Name': column_name,
            'Mean Value': column_means[column_name],
            'Top 10 Deviations': max_values[column_name]
        })
    return charts_data, table_data