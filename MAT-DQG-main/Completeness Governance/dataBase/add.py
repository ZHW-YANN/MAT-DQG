import sys
import json
import pymysql
import pymysql.cursors
from datetime import datetime

def insert_record(userName, op_type, op_describe, filePath, fileName, NewfileName, fileType):
    # 数据库连接配置
    op_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='123456',
                                 database='xai',
                                 charset='utf8',
                                 cursorclass=pymysql.cursors.DictCursor)

    try:
        with connection.cursor() as cursor:
            # 创建插入SQL语句
            sql = """
            INSERT INTO user_op_info 
            (userName, op_time, op_type, op_describe, filePath, fileName, NewfileName, fileType) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            # 执行SQL语句
            cursor.execute(sql, (userName, op_time, op_type, op_describe, filePath, fileName, NewfileName, fileType))
            # 提交至数据库执行
            connection.commit()
    except Exception as e:
        print(f"插入记录失败: {e}")
    finally:
        # 关闭数据库连接
        connection.close()

def save_data_info(data):
    # 连接数据库
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='123456',
                                 database='xai',
                                 charset='utf8',
                                 cursorclass=pymysql.cursors.DictCursor)

    try:
        with connection.cursor() as cursor:
            # 创建保存数据的SQL语句
            sql = """INSERT INTO data_info (materialName, datasource, method, samples, features, targetPro, ML, intro, knowledge1, knowledge2, knowledge3, task, timefeature, time, fileName, userName) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

            # 执行SQL语句
            cursor.execute(sql, (
                data['materialName'], data['datasource'], data['method'], data['samples'],
                data['features'], data['targetPro'], data['ML'], data['intro'],
                data['knowledge1'], data['knowledge2'], data['knowledge3'],
                data['task'], data['timefeature'], data['time'],
                data['fileName'], data['userName']
            ))

            # 提交事务
            connection.commit()
    finally:
        # 关闭数据库连接
        connection.close()

# 解析命令行参数并调用 insert_record 函数
if __name__ == "__main__":
    if len(sys.argv) == 8:
        print("Usage: python add.py <userName> <op_type> <op_describe> <filePath> <fileName> <NewfileName> <fileType>")
        # 获取命令行参数
        userName = sys.argv[1]
        op_type = sys.argv[2]
        op_describe = sys.argv[3]
        filePath = sys.argv[4]
        fileName = sys.argv[5]
        NewfileName = sys.argv[6]
        fileType = int(sys.argv[7])

        # 调用 insert_record 函数
        insert_record(userName, op_type, op_describe, filePath, fileName, NewfileName, fileType)
    elif len(sys.argv) == 3:
        print("Usage: python add.py save_data_info <json_data>")
        # 解析 JSON 数据
        json_data = sys.argv[2]
        data = json.loads(json_data)

        # 调用 save_data_info 函数
        save_data_info(data)