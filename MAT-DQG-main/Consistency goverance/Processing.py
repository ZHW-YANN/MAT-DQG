import json
import os
import re
import sys
import pandas as pd
import numpy as np
import urllib.parse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from consistency.normalization_methods import text_encoding, range_process, ch_unit
from dataBase.add import insert_record, save_data_info

def convert_to_native_types(obj):
    if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(v) for v in obj]
    else:
        return obj

def process_excel3(file_path, file_name, method, userName):
    try:
        file_path = urllib.parse.unquote(file_path)
        file_path = os.path.normpath(file_path)
        dir_name, file_name = os.path.split(file_path)
        base_name, ext_name = os.path.splitext(file_name)
        new_file_consist = f"uconsist_{base_name}{ext_name}"
        consistency_dir = os.path.join(dir_name, 'consistency')
        if not os.path.exists(consistency_dir):
            os.makedirs(consistency_dir)
        new_file_path_ok = os.path.join(consistency_dir, new_file_consist)

        if not os.path.exists(new_file_path_ok):
            return {
                "message": "Please first perform dimensionality inconsistency detection!",
                "train": None,
                "test": None
            }

        df = pd.read_excel(new_file_path_ok)

        if method == 'Yes':
            encoded_df, encoded_columns_info = text_encoding(df)  # 使用text_encoding方法处理数据集
        elif method == 'No':
            encoded_df, encoded_columns_info = text_encoding(df)  # 使用text_encoding方法处理数据集
        else:
            raise ValueError("Unsupported method.")

        dir_name, file_name = os.path.split(file_path)
        base_name, ext_name = os.path.splitext(file_name)
        new_file_info = f"info_{base_name}{ext_name}"
        new_temp_file_consist = f"D40_{base_name}{ext_name}"
        new_file_consist = f"D4_{base_name}{ext_name}"
        over_consist = f"over_{base_name}{ext_name}"

        encoded_columnsinfo_df = pd.DataFrame(encoded_columns_info).T

        consistency_dir = os.path.join(dir_name, 'consistency')
        if not os.path.exists(consistency_dir):
            os.makedirs(consistency_dir)

        new_file_path1 = os.path.join(consistency_dir, new_file_consist)
        new_file_path10 = os.path.join(consistency_dir, new_temp_file_consist)
        new_file_path2 = os.path.join(consistency_dir, new_file_info)
        over_consist_path = os.path.join(consistency_dir, over_consist)

        encoded_df.to_excel(new_file_path10, index=False, sheet_name='data')
        if 'Property' in encoded_df.columns:
            encoded_df.drop('Property', axis=1, inplace=True)

        complete_df = encoded_df.copy()
        complete_df.insert(0, 'No', range(len(complete_df)))
        with pd.ExcelWriter(new_file_path1, engine='xlsxwriter') as writer:
            encoded_df.to_excel(writer, index=False, sheet_name='data')
            complete_df.to_excel(writer, index=False, sheet_name='complete information')

        encoded_columnsinfo_df.to_excel(new_file_path2, index=False, sheet_name='data')


        insert_record(userName, 'Consistency', method, new_file_path1, file_name, new_file_consist, 1)
        insert_record(userName, 'Consistency', method, new_file_path10, file_name, new_temp_file_consist, 1)

        train_data = encoded_df.astype(object).where(pd.notnull(encoded_df), None).to_dict(orient='records')
        test_data = encoded_columnsinfo_df.astype(object).where(pd.notnull(encoded_columnsinfo_df), None).to_dict(orient='records')

        train_data = convert_to_native_types(train_data)
        test_data = convert_to_native_types(test_data)

        if isinstance(test_data, list) and len(test_data) > 0:
            for item in test_data:
                if "labels" in item and isinstance(item["labels"], dict):
                    item["labels"] = str(item["labels"])  # 将字典转换为字符串

        return {
            "message": "File processed successfully",
            "train": train_data,
            "test": test_data
        }
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        raise

def process_excel4(file_path, method, file_name):
    try:
        file_path = urllib.parse.unquote(file_path)
        file_path = os.path.normpath(file_path)
        df = pd.read_excel(file_path, sheet_name='data')
        property_column = df['Property'] if 'Property' in df.columns else None
        if 'Property' in df.columns:
            df.drop('Property', axis=1, inplace=True)

        if method == 'yes':
            encoded_df, encoded_columns_info = ch_unit(df)
        elif method == 'no':
            encoded_df, encoded_columns_info = ch_unit(df)
        else:
            raise ValueError("Unsupported method.")

        dir_name, file_name = os.path.split(file_path)
        base_name, ext_name = os.path.splitext(file_name)
        new_file_consist = f"uconsist_{base_name}{ext_name}"
        temp_file_info = f"tempinfo_{base_name}{ext_name}"
        temp_file_consist = f"tempconsist_{base_name}{ext_name}"

        consistency_dir = os.path.join(dir_name, 'consistency')
        if not os.path.exists(consistency_dir):
            os.makedirs(consistency_dir)

        new_file_path1 = os.path.join(consistency_dir, temp_file_consist)
        new_file_path2 = os.path.join(consistency_dir, temp_file_info)
        new_file_path_ok = os.path.join(consistency_dir, new_file_consist)

        encoded_df.to_excel(new_file_path1, index=False, sheet_name='data')
        encoded_columns_info.to_excel(new_file_path2, index=False, sheet_name='data')

        def remove_units(value):
            if pd.isnull(value):
                return value
            return pd.to_numeric(re.sub(r'[^\d.]+', '', str(value)), errors='coerce')

        for column in encoded_df.columns:
            if encoded_df[column].dtype == object:
                encoded_df[column] = encoded_df[column].apply(remove_units)

        encoded_df = encoded_df.apply(pd.to_numeric, errors='coerce')
        if property_column is not None:
            encoded_df['Property'] = property_column

        encoded_df.to_excel(new_file_path_ok, index=False, sheet_name='data')

        encoded_df_pdata = pd.read_excel(new_file_path1)
        encoded_df_infodata = pd.read_excel(new_file_path2)

        # 将 NaN 替换为 None
        encoded_df_pdata = encoded_df_pdata.where(pd.notnull(encoded_df_pdata), None)
        encoded_df_infodata = encoded_df_infodata.where(pd.notnull(encoded_df_infodata), None)
        # 返回 JSON 数据
        return {
            "message": "File processed successfully",
            "pdata": encoded_df_pdata.to_dict(orient='records'),  # 返回字典而不是 JSON 字符串
            "info": encoded_df_infodata.to_dict(orient='records')  # 返回字典而不是 JSON 字符串
        }

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        raise



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_excel.py <function_name> [args]")
        sys.exit(1)

    function_name = sys.argv[1]

    if function_name == "process_excel3":
        if len(sys.argv) != 6:
            print("Usage: python process_excel.py process_excel3 <file_path> <file_name> <method> <userName>")
            sys.exit(1)

        # 获取参数
        file_path = sys.argv[2]
        file_name = sys.argv[3]
        method = sys.argv[4]
        userName = sys.argv[5]

        result = process_excel3(file_path, file_name, method, userName)
        print(json.dumps(result))

    elif function_name == "process_excel4":
        if len(sys.argv) != 5:
            print("Usage: python process_excel.py process_excel4 <file_path> <file_name> <method>")
            sys.exit(1)


        file_path = sys.argv[2]
        file_name = sys.argv[3]
        method = sys.argv[4]
        result = process_excel4(file_path, file_name, method)

        print(json.dumps(result))

    else:
        print(f"Unknown function: {function_name}")
        sys.exit(1)

