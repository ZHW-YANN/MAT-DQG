import sys
import json
import pymysql
import pymysql.cursors
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def is_classification_task(userName):
    """
    检查给定的用户名是否对应于一个分类任务。

    :param userName: 根据此查表
    :return: 如果是分类任务返回True，否则返回False
    """
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='123456',
                                 database='xai',
                                 charset='utf8',
                                 cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            # 构造SQL查询语句
            sql = """
            SELECT task 
            FROM data_info 
            WHERE userName = %s
            ORDER BY id DESC
            LIMIT 1
            """
            cursor.execute(sql, (userName,))
            result = cursor.fetchone()
            if result:
                task = result['task']
                # print(f"task：{task}")
                # print(f"task == '分类'：{task == '分类'}")
                return task == '分类'
    finally:
        connection.close()
    # print("分类：false")
    return False

# 自定义 JSON 序列化器
def default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')  # 将 datetime 转换为字符串
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def get_user_ops_trace(userName):
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='123456',
                                 database='xai',
                                 charset='utf8',
                                 cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            # 编写你的SQL查询语句
            sql = """
                   SELECT * FROM user_op_info
                   WHERE userName = %s AND op_time >= (
                       SELECT op_time FROM user_op_info
                       WHERE userName = %s AND op_type = '数据挂号'
                       ORDER BY op_time DESC
                       LIMIT 1
                   )
                   ORDER BY op_time DESC
                   """
            cursor.execute(sql, (userName, userName))
            results = cursor.fetchall()
            return results
    finally:
        connection.close()

# 解析命令行参数并调用相应函数
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find.py <function_name> [args]")
        sys.exit(1)

    function_name = sys.argv[1]

    if function_name == "get_user_ops_trace":
        if len(sys.argv) != 3:
            print("Usage: python find.py get_user_ops_trace <userName>")
            sys.exit(1)

        # 获取用户名
        userName = sys.argv[2]

        # 调用 get_user_ops_trace 函数
        results = get_user_ops_trace(userName)

        # 将结果转换为 JSON 字符串并打印
        print(json.dumps(results, default=default_serializer, ensure_ascii=False))  # 禁用 ASCII 编码
    else:
        print(f"Unknown function: {function_name}")
        sys.exit(1)
