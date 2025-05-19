import json
import numpy as np
import copy
import random
import sqlite3
import pandas as pd
from pprint import pprint


def load_from_json(file_path):
    # 读取 JSON 文件并返回字典列表
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


# 加载 json 文件
def load_list_json(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_to_json(file_path, prompts):
    # 将字典列表逐行写入 JSON 文件
    with open(file_path, "w", encoding="utf-8") as f:
        for d in prompts:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")


def random_items_from_dict(input_dict, num_items):
    if num_items > len(input_dict):
        raise ValueError("指定的数量大于字典中的项目数量")

    keys = list(input_dict.keys())
    random_keys = random.sample(keys, num_items)

    random_items = {key: input_dict[key] for key in random_keys}

    return random_items


def calculate_median(arr):
    sorted_arr = np.sort(arr)
    length = len(arr)
    middle_index = length // 2

    if length % 2 == 1:
        # 数组长度为奇数，直接返回中间值
        median = sorted_arr[middle_index]
    else:
        # 数组长度为偶数，返回中间两个值的较小值
        median = sorted_arr[middle_index - 1]

    return median


# 读取某个表的某个字段数据
def read_sqlite_column_data(db_id, table_name, column_name, db_type):
    if db_type == "spider":
        db_path = f"sql_data/spider/database/{db_id}/{db_id}.sqlite"
    elif db_type == "cspider":
        db_path = f"sql_data/cspider/database/{db_id}/{db_id}.sqlite"

    conn = sqlite3.connect(db_path)
    # conn.text_factory = str
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cursor = conn.cursor()
    # query = f'SELECT {column_name} FROM {table_name}'
    query = f"SELECT [{column_name}] FROM {table_name}"
    cursor.execute(query)

    results = cursor.fetchall()
    column_data = [res[0] for res in results]

    cursor.close()
    conn.close()

    return column_data


def str_to_number(input_str):
    if isinstance(input_str, str):
        try:
            res = int(input_str)
        except:
            try:
                res = float(input_str)
            except:
                res = None
    else:
        res = input_str
    return res


op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!="}
agg_sql_dict = {0: "", 1: "mean", 2: "max", 3: "min", 4: "unique", 5: "sum"}
conn_sql_dict = {0: "", 1: "and", 2: "or"}


def generate_nl2sql_dataset_describes():
    # 转换NL2SQL数据集
    dataset_list = []
    nl2sql_table_info = load_from_json(
        "../SQL_DATA/nl2sql/TableQA/train/train.tables.json"
    )

    for table_info in nl2sql_table_info:
        categorical_cols = []
        numeric_cols = []

        for idx, col in enumerate(table_info["header"]):
            if table_info["types"][idx] == "real":
                numeric_cols.append(col)
            elif table_info["types"][idx] == "text":
                categorical_cols.append(col)
            else:
                print("不支持的 types")

        # info 信息
        rows = np.array(table_info["rows"])

        # 类别 info
        categorical_info = dict()
        for col in categorical_cols:
            idx = table_info["header"].index(col)
            cgl_rows = rows[:, idx]
            unique_values = np.unique(cgl_rows).tolist()

            # 判断离散类别存在字符大于30的把该类别去掉
            len_bool = [
                True if len(ca_value) > 30 else False for ca_value in unique_values
            ]
            if True in len_bool:
                continue

            if len(unique_values) == 0:
                continue

            if len(unique_values) > 10:
                unique_values = random.sample(unique_values, 10)

            categorical_info[col] = unique_values

        # 数值 info
        numeric_info = dict()
        for cl in numeric_cols:
            idx = table_info["header"].index(cl)
            cgl_rows = rows[:, idx]

            elements_to_remove = ["None", "-"]
            cgl_rows = cgl_rows[~np.isin(cgl_rows, elements_to_remove)]

            if len(cgl_rows) == 0:
                # numeric_info[cl] = []
                continue
            else:
                try:
                    median_value = calculate_median(cgl_rows)
                    max_value = np.max(cgl_rows)
                    min_value = np.min(cgl_rows)
                except:
                    try:
                        cgl_rows = cgl_rows.astype(int)
                        median_value = calculate_median(cgl_rows)
                        max_value = np.max(cgl_rows)
                        min_value = np.min(cgl_rows)
                    except:
                        # print(cgl_rows)
                        cgl_rows = cgl_rows.astype(float)
                        median_value = calculate_median(cgl_rows)
                        max_value = np.max(cgl_rows)
                        min_value = np.min(cgl_rows)

                numeric_info[cl] = [min_value, median_value, max_value]

        # 过滤
        if len(categorical_info) > 10:
            categorical_info = random_items_from_dict(categorical_info, 10)
            # random_items = random.sample(categorical_info.items(), 10)
            # categorical_info = dict(random_items)

        if len(numeric_info) > 10:
            numeric_info = random_items_from_dict(numeric_info, 10)
            # random_items = random.sample(numeric_info.items(), 10)
            # numeric_info = dict(random_items)

        if len(categorical_info) == 0 and len(numeric_info) == 0:
            continue

        # print("table_info", table_info)
        # exit()
        # 如果table_title 不为空
        table_title = table_info["title"]
        if table_title.strip() == "":
            continue

        # 表名处理
        # 找到逗号第一次出现的位置
        comma_index_1 = table_title.find("：")
        comma_index_2 = table_title.find("、")
        comma_index_3 = table_title.find(":")
        comma_index_4 = table_title.find(".")
        # 过滤掉-1
        comma_index = [comma_index_1, comma_index_2, comma_index_3, comma_index_4]
        filtered_list = [x for x in comma_index if x != -1]

        if len(filtered_list) == 0:
            continue
        else:
            minimum_index = min(filtered_list)
            result_string = table_title[minimum_index + 1 :]

        # 过滤掉空字符
        filtered_string = result_string.replace(" ", "")
        # print(filtered_string)

        # print("table_title", table_title)
        db_name_list = [ii["db_name"] for ii in dataset_list]
        if filtered_string not in db_name_list:
            dataset_list.append(
                {
                    "db_name": filtered_string,
                    "db_info": {
                        filtered_string: {
                            "numeric_info": numeric_info,
                            "categorical_info": categorical_info,
                            "date_cols_info": {},
                        }
                    },
                    "foreign_keys": [],
                }
            )

    print("nl2sql 数据量：", len(dataset_list))
    # 保存结果

    return dataset_list


def generate_wikisql_dataset_describes():
    # 转换NL2SQL数据集
    dataset_list = []
    wikisql_table_info = load_from_json("../SQL_DATA/wikisql/data/train.tables.jsonl")

    # from pprint import pprint
    # pprint(wikisql_table_info[1])
    # exit()

    for table_info in wikisql_table_info:
        categorical_cols = []
        numeric_cols = []

        for idx, col in enumerate(table_info["header"]):
            if table_info["types"][idx] == "real":
                numeric_cols.append(col)
            elif table_info["types"][idx] == "text":
                categorical_cols.append(col)
            else:
                print("不支持的 types")

        # info 信息
        rows = np.array(table_info["rows"])

        # 类别 info
        categorical_info = dict()
        for col in categorical_cols:
            idx = table_info["header"].index(col)
            cgl_rows = rows[:, idx]
            unique_values = np.unique(cgl_rows).tolist()

            # 判断离散类别存在字符大于30的把该类别去掉
            len_bool = [
                True if len(ca_value) > 30 else False for ca_value in unique_values
            ]
            if True in len_bool:
                continue

            if len(unique_values) == 0:
                continue

            if len(unique_values) > 10:
                unique_values = random.sample(unique_values, 10)

            categorical_info[col] = unique_values

        # 数值 info
        numeric_info = dict()
        for cl in numeric_cols:
            idx = table_info["header"].index(cl)
            cgl_rows = rows[:, idx]

            elements_to_remove = ["None", "-"]
            cgl_rows = cgl_rows[~np.isin(cgl_rows, elements_to_remove)]
            try:
                cgl_rows = np.char.replace(cgl_rows, ",", "")
            except:
                cgl_rows = cgl_rows

            # print(cgl_rows)

            if len(cgl_rows) == 0:
                # numeric_info[cl] = []
                continue
            else:
                try:
                    median_value = calculate_median(cgl_rows)
                    max_value = np.max(cgl_rows)
                    min_value = np.min(cgl_rows)
                except:
                    try:
                        cgl_rows = cgl_rows.astype(int)
                        median_value = int(calculate_median(cgl_rows))
                        max_value = np.max(cgl_rows)
                        min_value = np.min(cgl_rows)
                    except:
                        # print(cgl_rows)
                        cgl_rows = cgl_rows.astype(float)
                        median_value = calculate_median(cgl_rows)
                        max_value = np.max(cgl_rows)
                        min_value = np.min(cgl_rows)

                numeric_info[cl] = np.array(
                    [min_value, median_value, max_value]
                ).tolist()

        # 过滤
        if len(categorical_info) > 10:
            categorical_info = random_items_from_dict(categorical_info, 10)
            # random_items = random.sample(categorical_info.items(), 10)
            # categorical_info = dict(random_items)

        if len(numeric_info) > 10:
            numeric_info = random_items_from_dict(numeric_info, 10)
            # random_items = random.sample(numeric_info.items(), 10)
            # numeric_info = dict(random_items)

        if len(categorical_info) == 0 and len(numeric_info) == 0:
            continue

        if "caption" not in table_info.keys():
            continue

        table_title = table_info["caption"]
        if len(table_title) <= 3:
            continue

        # 过滤掉空字符
        replace_string = table_title.replace(" ", "_")

        db_name_list = [ii["db_name"] for ii in dataset_list]
        if replace_string in db_name_list:
            print(replace_string)
        if replace_string not in db_name_list:
            dataset_list.append(
                {
                    "db_name": replace_string,
                    "db_info": {
                        replace_string: {
                            "numeric_info": numeric_info,
                            "categorical_info": categorical_info,
                            "date_cols_info": {},
                        }
                    },
                    "foreign_keys": [],
                }
            )

    print("wikisql 数据量：", len(dataset_list))
    # 保存结果

    return dataset_list


def generate_spider_dataset_describes(table_path):
    # table_path = "sql_data/spider/tables.json"
    tables_data = load_list_json(table_path)

    db_data = []
    db_data_dict = {}

    for table in tables_data:
        table_names_original = table["table_names_original"]
        db_id = table["db_id"]
        column_names_original = table["column_names_original"]
        column_types = table["column_types"]
        foreign_keys = table["foreign_keys"]

        # 解析外键
        columns_foreign_keys = []
        cc_column_names_original = copy.deepcopy(column_names_original)
        cc_table_names_original = copy.deepcopy(table_names_original)
        for inxnum in foreign_keys:
            first_tc = inxnum[0]
            second_tc = inxnum[1]

            first_key = copy.deepcopy(cc_column_names_original[first_tc])
            second_key = copy.deepcopy(cc_column_names_original[second_tc])

            first_key[0] = cc_table_names_original[first_key[0]]
            second_key[0] = cc_table_names_original[second_key[0]]

            columns_foreign_keys.append([tuple(first_key), tuple(second_key)])

        # 赋值初始化
        db_info = dict()
        columns_info = dict()
        for single_table_name in table_names_original:
            columns_info[single_table_name] = {
                "head": {"categorical_cols": [], "numeric_cols": [], "date_cols": []},
                "df_info": {
                    "categorical_info": dict(),
                    "numeric_info": dict(),
                    "date_cols_info": dict(),
                },
            }

        for k, v in enumerate(column_names_original):
            # 跳过第一个
            if k == 0:
                continue

            single_table_name = table_names_original[v[0]]

            if column_types[k] == "number":
                column_data = read_sqlite_column_data(
                    db_id, single_table_name, v[1], "spider"
                )

                # 过滤 None 空值，特殊字符形式
                filter_column_data = []
                for dt in column_data:
                    if dt == "":
                        continue
                    if dt == None:
                        continue
                    if dt == "inf":
                        continue
                    if dt == "NULL":
                        continue
                    # 其他字符形式
                    if isinstance(dt, str):
                        dt = str_to_number(dt)
                        if dt is not None:
                            filter_column_data.append(dt)
                    else:
                        filter_column_data.append(dt)

                # 计算中位数最大最小值
                if len(filter_column_data) == 0:
                    # continue
                    columns_info[single_table_name]["head"]["numeric_cols"].append(v[1])
                    columns_info[single_table_name]["df_info"]["numeric_info"][v[1]] = []
                else:
                    median_value = calculate_median(filter_column_data)
                    max_value = np.max(filter_column_data)
                    min_value = np.min(filter_column_data)
                    num_list = np.array([min_value, median_value, max_value]).tolist()
                    columns_info[single_table_name]["head"]["numeric_cols"].append(v[1])
                    columns_info[single_table_name]["df_info"]["numeric_info"][
                        v[1]
                    ] = num_list

            elif column_types[k] == "time":
                column_data = read_sqlite_column_data(
                    db_id, single_table_name, v[1], "spider"
                )
                # 含有 None 处理
                if None in column_data:
                    column_data = [col for col in column_data if col is not None]
                try:
                    datetime_series = pd.to_datetime(column_data)
                    date_range_start = str(datetime_series.min())
                    date_range_end = str(datetime_series.max())

                    columns_info[single_table_name]["head"]["date_cols"].append(v[1])
                    columns_info[single_table_name]["df_info"]["date_cols_info"][
                        v[1]
                    ] = [date_range_start, date_range_end]
                except:
                    columns_info[single_table_name]["head"]["date_cols"].append(v[1])
                    columns_info[single_table_name]["df_info"]["date_cols_info"][v[1]] = []

            elif (
                column_types[k] == "text"
                or column_types[k] == "boolean"
                or column_types[k] == "others"
            ):
                column_data = read_sqlite_column_data(
                    db_id, single_table_name, v[1], "spider"
                )
                # 含有 None 处理
                if None in column_data:
                    column_data = [col for col in column_data if col is not None]
                unique_values = np.unique(column_data).tolist()

                if len(unique_values) > 5:
                    unique_values = unique_values[0:5]

                # 判断离散类别存在字符大于30的把该类别去掉
                len_bool = [
                    True if len(str(ca_value)) > 30 else False
                    for ca_value in unique_values
                ]
                if True in len_bool:
                    unique_values = unique_values[0:2]
                    # continue

                # if len(unique_values) == 0:
                #     continue

                columns_info[single_table_name]["head"]["categorical_cols"].append(v[1])
                columns_info[single_table_name]["df_info"]["categorical_info"][
                    v[1]
                ] = unique_values
            else:
                raise ValueError("no column type: ", column_types[k])
                # column_data = read_sqlite_column_data(db_id, single_table_name, v[1])

        columns_info_copy = copy.deepcopy(columns_info)


        # 重新拼装格式，过滤掉没有列的表

        format_db_info = {"db_name": db_id, "db_info": dict()}

        all_table_columns = []
        continue_flag = False

        for ks, vs in columns_info_copy.items():
            s_table_info = copy.deepcopy(vs["df_info"])
            s_head = vs["head"]

            ff_all_table_columns = (
                s_head["categorical_cols"]
                + s_head["date_cols"]
                + s_head["numeric_cols"]
            )
            cc_all_table_columns = copy.deepcopy(ff_all_table_columns)

            if len(cc_all_table_columns) == 0:
                raise ValueError("len(cc_all_table_columns)==0")

            chose_ca = copy.deepcopy(s_table_info["categorical_info"])
            categorical_info = chose_ca
            # if len(s_head["categorical_cols"]) > 10:
            #     categorical_info = random_items_from_dict(chose_ca, 10)
            # else:
            #     categorical_info = chose_ca

            chose_nu = copy.deepcopy(s_table_info["numeric_info"])
            numeric_info = chose_nu
            # if len(s_head["numeric_cols"]) > 10:
            #     numeric_info = random_items_from_dict(chose_nu, 10)
            # else:
            #     numeric_info = chose_nu

            zz_all_table_columns = [
                copy.deepcopy((ks, g)) for g in cc_all_table_columns
            ]
            all_table_columns.extend(zz_all_table_columns)


            format_db_info["db_info"][ks] = {
                "categorical_info": copy.deepcopy(categorical_info),
                "numeric_info": copy.deepcopy(numeric_info),
                "date_cols_info": copy.deepcopy(s_table_info["date_cols_info"]),
            }

        filter_columns_foreign_keys = []
        for tup in columns_foreign_keys:
            if tup[0] in all_table_columns and tup[1] in all_table_columns:
                filter_columns_foreign_keys.append(tup)

        format_db_info["foreign_keys"] = copy.deepcopy(filter_columns_foreign_keys)

        if len(format_db_info["db_info"]) != 0:
            db_data.append(copy.deepcopy(format_db_info))
            db_data_dict[db_id] = format_db_info

    print("spider 数据量", len(db_data))
    # pprint(db_data)
    return db_data,db_data_dict


def generate_dusql_dataset_describes():
    db_schema_path = "sql_data/DuSQL/db_schema.json"
    db_content_path = "sql_data/DuSQL/db_content.json"

    tables_data = load_list_json(db_schema_path)
    db_content_data = load_list_json(db_content_path)

    db_data = []
    db_data_dict = {}

    for table in tables_data:
        # print(table)
        table_names_original = table["table_names"]
        db_id = table["db_id"]
        column_names_original = table["column_names"]
        column_types = table["column_types"]
        foreign_keys = table["foreign_keys"]

        # 解析外键
        columns_foreign_keys = []
        cc_column_names_original = copy.deepcopy(column_names_original)
        cc_table_names_original = copy.deepcopy(table_names_original)
        for inxnum in foreign_keys:
            first_tc = inxnum[0]
            second_tc = inxnum[1]

            first_key = copy.deepcopy(cc_column_names_original[first_tc])
            second_key = copy.deepcopy(cc_column_names_original[second_tc])

            first_key[0] = cc_table_names_original[first_key[0]]
            second_key[0] = cc_table_names_original[second_key[0]]

            columns_foreign_keys.append([tuple(first_key), tuple(second_key)])

        # 赋值初始化
        db_info = dict()
        columns_info = dict()
        for single_table_name in table_names_original:
            columns_info[single_table_name] = {
                "head": {"categorical_cols": [], "numeric_cols": [], "date_cols": []},
                "df_info": {
                    "categorical_info": dict(),
                    "numeric_info": dict(),
                    "date_cols_info": dict(),
                },
            }

        for k, v in enumerate(column_names_original):
            # 跳过第一个
            if k == 0:
                continue

            single_table_name = table_names_original[v[0]]

            # tables_df
            tables_df = [
                contt["tables"][single_table_name]
                for contt in db_content_data
                if contt["db_id"] == db_id][0]
            

            if column_types[k] == "number":
                rows = np.array(tables_df["cell"])
                idx = tables_df["header"].index(v[1])
                cgl_rows = rows[:, idx]
                # 含有 None 处理
                if None in cgl_rows:
                    cgl_rows = [col for col in cgl_rows if col is not None]

                # column_data = read_sqlite_column_data(db_id, single_table_name, v[1], "spider")

                # 过滤 None 空值，特殊字符形式
                filter_column_data = []
                for dt in cgl_rows:
                    if dt == "":
                        continue
                    if dt == None:
                        continue
                    if dt == "inf":
                        continue
                    if dt == "NULL":
                        continue
                    # 其他字符形式
                    if isinstance(dt, str):
                        dt = str_to_number(dt)
                        if dt is not None:
                            filter_column_data.append(dt)
                    else:
                        filter_column_data.append(dt)

                # 计算中位数最大最小值
                if len(filter_column_data) == 0:
                    # 按类别解析
                    unique_values = np.unique(cgl_rows).tolist()
                    if len(unique_values) > 10:
                        unique_values = unique_values[0:10]

                    # 判断离散类别存在字符大于30的把该类别去掉
                    len_bool = [
                        True if len(str(ca_value)) > 30 else False
                        for ca_value in unique_values
                    ]
                    if True in len_bool:
                        # print(unique_values)
                        continue

                    columns_info[single_table_name]["head"]["categorical_cols"].append(
                        v[1]
                    )
                    columns_info[single_table_name]["df_info"]["categorical_info"][
                        v[1]
                    ] = unique_values
                    # columns_info[single_table_name]["head"]["numeric_cols"].append(v[1])
                    # columns_info[single_table_name]["df_info"]["numeric_info"][v[1]] = []
                else:
                    median_value = calculate_median(filter_column_data)
                    max_value = np.max(filter_column_data)
                    min_value = np.min(filter_column_data)
                    num_list = np.array([min_value, median_value, max_value]).tolist()

                    columns_info[single_table_name]["head"]["numeric_cols"].append(v[1])
                    columns_info[single_table_name]["df_info"]["numeric_info"][
                        v[1]
                    ] = num_list

            elif column_types[k] == "time":
                rows = np.array(tables_df["cell"])
                idx = tables_df["header"].index(v[1])
                cgl_rows = rows[:, idx]
                # 含有 None 处理
                if None in cgl_rows:
                    cgl_rows = [col for col in cgl_rows if col is not None]

                try:
                    datetime_series = pd.to_datetime(cgl_rows)
                    date_range_start = str(datetime_series.min())
                    date_range_end = str(datetime_series.max())

                    columns_info[single_table_name]["head"]["date_cols"].append(v[1])
                    columns_info[single_table_name]["df_info"]["date_cols_info"][
                        v[1]
                    ] = [date_range_start, date_range_end]
                except:
                    columns_info[single_table_name]["head"]["date_cols"].append(v[1])
                    columns_info[single_table_name]["df_info"]["date_cols_info"][
                        v[1]
                    ] = []


            elif (
                column_types[k] == "text"
                or column_types[k] == "boolean"
                or column_types[k] == "binary"
                or column_types[k] == "others"
            ):
                rows = np.array(tables_df["cell"])
                idx = tables_df["header"].index(v[1])
                cgl_rows = rows[:, idx]
                # 含有 None 处理
                if None in cgl_rows:
                    cgl_rows = [col for col in cgl_rows if col is not None]
                unique_values = np.unique(cgl_rows).tolist()
                if len(unique_values) > 5:
                    unique_values = unique_values[0:5]

                # 判断离散类别存在字符大于30的把该类别去掉
                len_bool = [
                    True if len(str(ca_value)) > 30 else False
                    for ca_value in unique_values
                ]
                if True in len_bool:
                    # continue
                    unique_values = unique_values[0:2]

                columns_info[single_table_name]["head"]["categorical_cols"].append(v[1])
                columns_info[single_table_name]["df_info"]["categorical_info"][
                    v[1]
                ] = unique_values
            else:
                # column_data = read_sqlite_column_data(db_id, single_table_name, v[1])
                print("no column type: ", column_types[k])
                # print(column_data)

        # print("columns_info", columns_info)
        # pprint(columns_info)
        columns_info_copy = copy.deepcopy(columns_info)

        # from pprint import pprint
        # pprint(columns_info_copy)

        # 重新拼装格式，过滤掉没有列的表

        format_db_info = {"db_name": db_id, "db_info": dict()}

        all_table_columns = []
        continue_flag = False

        for ks, vs in columns_info_copy.items():
            s_table_info = copy.deepcopy(vs["df_info"])
            s_head = vs["head"]

            ff_all_table_columns = (
                s_head["categorical_cols"]
                + s_head["date_cols"]
                + s_head["numeric_cols"]
            )
            cc_all_table_columns = copy.deepcopy(ff_all_table_columns)

            if len(cc_all_table_columns) == 0:
                raise ValueError("len(cc_all_table_columns)==0")

            chose_ca = copy.deepcopy(s_table_info["categorical_info"])
            categorical_info = chose_ca
            # if len(s_head["categorical_cols"]) > 10:
            #     categorical_info = random_items_from_dict(chose_ca, 10)
            # else:
            #     categorical_info = chose_ca

            chose_nu = copy.deepcopy(s_table_info["numeric_info"])
            numeric_info = chose_nu
            # if len(s_head["numeric_cols"]) > 10:
            #     numeric_info = random_items_from_dict(chose_nu, 10)
            # else:
            #     numeric_info = chose_nu

            zz_all_table_columns = [
                copy.deepcopy((ks, g)) for g in cc_all_table_columns
            ]
            all_table_columns.extend(zz_all_table_columns)

            format_db_info["db_info"][ks] = {
                "categorical_info": copy.deepcopy(categorical_info),
                "numeric_info": copy.deepcopy(numeric_info),
                "date_cols_info": copy.deepcopy(s_table_info["date_cols_info"]),
            }

        filter_columns_foreign_keys = []
        for tup in columns_foreign_keys:
            if tup[0] in all_table_columns and tup[1] in all_table_columns:
                filter_columns_foreign_keys.append(tup)

        format_db_info["foreign_keys"] = copy.deepcopy(filter_columns_foreign_keys)

        if len(format_db_info["db_info"]) != 0:
            db_data.append(copy.deepcopy(format_db_info))
            db_data_dict[db_id] = format_db_info

        # db_data.append(copy.deepcopy(format_db_info))

    print("dusql 数据量", len(db_data))

    return db_data,db_data_dict


if __name__ == "__main__":
    # spider_data = generate_spider_dataset_describes()
    # dusql_data = generate_dusql_dataset_describes()

    nl2sql_data = generate_nl2sql_dataset_describes()
