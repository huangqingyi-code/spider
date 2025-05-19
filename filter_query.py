import datetime
from numbers import Number
from typing import Optional
from itertools import chain
import pandas as pd
from pandas.api.types import is_numeric_dtype
import json


#TODO pd.query的column不能中'()','.','[]','/'特殊字符

def __map_dtypes(df_col: pd.Series) -> str:
    """Convert pandas dtype to our definition of data type."""
    if len(df_col) > 10000:
        threshold = 0.01
    else:
        threshold = 0.05

    dtype = df_col.dtype
    if is_numeric_dtype(dtype):
        if df_col.nunique() / len(df_col) >= threshold:
            return "numerical"
        else:
            return "categorical"
    else:
        try:
            datetime = pd.to_datetime(df_col)
            return "datetime"
        except:
            try:
                timedelta = pd.to_timedelta(df_col)
                return "timedelta"
            except:
                return "categorical"


def equal_and_unequal(col, condition, value, dsl_name: str = ""):
    assert condition in ["==", "!="], f"equal_and_unequal,condition:{condition}"
    if value == dsl_name:
        query_func = f"`{col}` {condition} @{value}"
    elif value == None or value == [None]:
        if condition == "==":
            query_func = f"`{col}`.isnull()"
        else:
            query_func = f"`{col}`.notnull()"
    elif isinstance(value, list):
        query_func = f"`{col}` {condition} {value}"
    else:
        raise ValueError(f"equal_and_unequal,value:{value}")

    return query_func


def parse_numerical(col, condition, not_op, value, cols, inputs):
    """
    value为str时,value为column或者dsl_name;
    value是None时,(选出某列不为空的数据);
    """
    assert condition in [
        ">",
        "<",
        ">=",
        "<=",
        "==",
        "!=",
        "in",
        "between",
    ], f"numerical不支持condition:{condition}！"

    if len(value) > 1:
        # between是2个，==和!=可以多个值。
        if condition == "between" and len(value) == 2:
            # condition:between, value:[10,20]
            query_func = f"{value[0]} < `{col}` < {value[1]}"
        elif condition == "in":
            query_func = f"`{col}` {condition} {value}"
        else:
            # condition:==,value:[1,2,3,4,5]
            assert condition in [
                "==",
                "!=",
            ], f"numerical时,只有=与!=支持多个值,condition:{condition},value:{value}"
            query_func = equal_and_unequal(col, condition, value)
    else:
        assert condition != "between", "value只有1个值时,condition不能为between！"
        value = value[0]
        if value=="null":
            if condition=="==":
                query_func = f'`{col}`.isnull()"'
            elif condition=="!=":
                query_func = f'`{col}`.notnull()"'
            else:
                raise ValueError(f"当value为null,condition值error。condition:{condition}")
        else:
            if isinstance(value, Number):
                query_func = f"`{col}` {condition} {value}"
            elif value in cols:
                # value是column
                query_func = f"`{col}` {condition} {value}"
            # elif value.startswith("df_"):
            elif value == inputs[-1]:
                # value是df_dsl
                query_func = f"`{col}` {condition} @{value}"
            
            elif value == None:
                # value是空值(选出某列不为空的数据)
                assert condition in [
                    "==",
                    "!=",
                ], f"value为None时,condition不支持:{condition}"
                query_func = equal_and_unequal(col, condition, value)
            else:
                raise ValueError(
                    f"当column为numerical,value值error。value:{value},{type(value)}"
                )
    if not_op:
        query_func = "(not " + query_func + ")"
    return query_func


def parse_categorical(col, condition, not_op, value, cols, inputs):
    """
    condition为==,!=时value可以为任意类型的数据，其他condition时只能为str
    """
    if len(value) > 1:
        assert condition in [
            "==",
            "!=",
            "in",
        ], f"categorical时,len(value)>1不支持condition:{condition}！"
        query_func = f"`{col}` {condition} {value}"

    else:
        value = value[0]
        if value=="null":
            if condition=="==":
                query_func = f'`{col}`.isnull()"'
            elif condition=="!=":
                query_func = f'`{col}`.notnull()"'
            else:
                raise ValueError(f"当value为null,condition值error。condition:{condition}")
        else:
            if condition in [">", "<", ">=", "<=", "==", "!=", "in"]:
                # if value.startswith("df_"):
                if value == inputs[-1]:
                    query_func = f"`{col}` {condition} @{value}"
                elif value in cols:
                    query_func = f"`{col}` {condition} {value}"
                elif value == None:
                    # value是空值(选出某列不为空的数据)
                    assert condition in [
                        "==",
                        "!=",
                    ], f"value为None时,condition不支持:{condition}"
                    query_func = equal_and_unequal(col, condition, value)
                else:
                    query_func = f'`{col}` {condition} "{value}"'
            elif condition in ["contains", "startswith", "endswith"]:
                assert isinstance(
                    value, str
                ), f"当column为categorical,value值error。value:{value},{type(value)}"
                query_func = f'`{col}`.str.{condition}("{value}")'
            else:
                raise ValueError(f"condition error,condition:{condition}")
    if not_op:
        query_func = "(not " + query_func + ")"
    return query_func


def parse_date(col, condition, not_op, value, inputs):
    """
    在两个时间段之间value:[time_stamp1,time_stamp2],condition:between
    某个时段value:[time_stamp],2023.11.16
    时间差：value:[time_diff],18.0-days
    """
    units_map = {
        "years": 60 * 60 * 24 * 365,
        "quarters": 60 * 60 * 24 * 90,
        "months": 60 * 60 * 24 * 30,
        "weeks": 60 * 60 * 24 * 7,
        "days": 60 * 60 * 24,
        "hours": 60 * 60,
        "minutes": 60,
        "seconds": 1,
    }
    # assert condition in [
    #     ">",
    #     "<",
    #     ">=",
    #     "<=",
    #     "==",
    #     "!=",
    #     "between",
    # ], f"date不支持condition:{condition}！"
    if len(value) == 1:
        value = value[0]
        # if isinstance(value, Number):
        #     value = str(value)
        assert isinstance(value, str), f"当column为date,value只能是str或num,value:{value}"
        try:
            time_stamp = pd.to_datetime(value)
            query_func = f'`{col}` {condition} "{value}"'
        except:
            if value == "time_now":
                time_stamp = datetime.datetime.now().strftime("%Y-%m-%d")
                query_func = f'`{col}` {condition} "{value}"'
            elif "-" in value:
                value_split = value.split("-")
                unit = value_split[-1]
                num = value_split[0]
                try:
                    num = float(num)
                    if unit in units_map and len(value_split) == 2:
                        time_num = num * units_map[unit]
                        query_func = f'`{col}` {condition} "{num} {unit}"'
                    else:
                        raise ValueError(f"当column为date,value值error。value:{value}")
                except:
                    raise ValueError(f"当column为date,value值error。value:{value}")
            elif condition in ["contains", "startswith", "endswith"]:
                assert isinstance(
                    value, str
                ), f"当column为categorical,value值error。value:{value},{type(value)}"
                query_func = f'`{col}`.str.{condition}("{value}")'
            else:
                # TODO
                if value == inputs[-1]:
                    # 结束时间_sub_起始时间 <= @df_statics_column_1
                    query_func = f"`{col}` {condition} @{value}"
                else:
                    try:
                        # 起始时间 <= "1456"
                        float(value)
                        query_func = f'`{col}` {condition} "{value}"'
                    except:
                        if value=="null":
                            if condition=="==":
                                query_func = f'`{col}`.isnull()"'
                            elif condition=="!=":
                                query_func = f'`{col}`.notnull()"'
                            else:
                                raise ValueError(f"当value为null,condition值error。condition:{condition}")
                        else:
                            raise ValueError(f"当column为date,value值error。value:{value}")

    elif len(value) == 2:
        # 在两个时段之间,condition:between,value:[2012.2,2012.8]
        time_stamp1 = str(value[0])
        time_stamp2 = str(value[1])
        try:
            time_stamp1_ = pd.to_datetime(time_stamp1)
            time_stamp2_ = pd.to_datetime(time_stamp2)
            query_func = f'"{time_stamp1}" < `{col}` < "{time_stamp2}"'
        except:
            raise ValueError(f"当column为date,value值error。value:{value}")

    else:
        raise ValueError(f"当column为date,value值error。value:{value}")
    if not_op:
        query_func = "(not " + query_func + ")"
    return query_func


def parse_single_bool_arg(arg, cols, date_cols, inputs):
    """
    {"columns": ["City"], "condition": "==", "not":False,"value": "Asheville"}
    """
    assert set(list(arg.keys())) == set(
        ["bool_columns", "condition", "not", "value"]
    ), f"bool_args key error,key:{arg.keys()}"

    col = arg["bool_columns"][0]
    # if (not col in cols) or len(arg["bool_columns"]) != 1:
    #     raise ValueError(f"column错误,dataframe的columns:{cols},输入的column:`{col}`")

    condition = arg["condition"]
    not_op = arg["not"]
    value = arg["value"]

    if condition == "is":
        # in is to ==
        condition = "=="
    elif condition == "in" and len(value) == 1:
        if not (value[0] == inputs[-1] or value[0] in cols):
            # value不在input中（不是@）或者value不是column
            condition = "=="

    if inputs[-1] in value or None in value:
        assert len(value) == 1, f"value为DSL输出或None时，里面只能有一个值。value:{value}"

    if col[0] == "(" and col[-1] == ")" and ("sub" in col or "add" in "col"):
        # "(结束时间_sub_起始时间,mean)",(all,count)不能去掉括号
        col_ = col[1:-1]
    else:
        col_ = col
    if "." in col_:
        c_s = col_.split(".")
    else :
        #"_" in col_
        c_s = col_.split("_")
    if (
        col in date_cols
        or "time_now" in col
        or (c_s[0] in date_cols or c_s[-1] in date_cols)
    ):
        query_func = parse_date(col, condition, not_op, value, inputs)
    else:
        if isinstance(value[0], Number):
            query_func = parse_numerical(col, condition, not_op, value, cols, inputs)
        else:
            query_func = parse_categorical(col, condition, not_op, value, cols, inputs)

    return query_func


def parse_bool_args(param, cols, date_cols, inputs):
    if len(param) == 1:
        if "or" in param:
            or_args = param["or"]
            parsed_args = [parse_bool_args(arg, cols, date_cols, inputs) for arg in or_args]
            return f"({' or '.join(parsed_args)})"
        elif "and" in param:
            and_args = param["and"]
            parsed_args = [parse_bool_args(arg, cols, date_cols, inputs) for arg in and_args]
            return f"({' and '.join(parsed_args)})"
        else:
            raise ValueError(f"bool关系只支持or和and,传入的为{list(param.keys())[0]}")
    else:
        return parse_single_bool_arg(param, cols, date_cols, inputs)


def bool_arg_2_query(data):
    table_infos = data["table_infos"]
    date_cols = list(
        chain(
            *[
                list(value["date_cols_info"].keys())
                for key, value in table_infos["db_info"].items()
            ]
        )
    )
    categorical_cols = list(
        chain(
            *[
                list(value["categorical_info"].keys())
                for key, value in table_infos["db_info"].items()
            ]
        )
    )
    numeric_cols = list(
        chain(
            *[
                list(value["numeric_info"].keys())
                for key, value in table_infos["db_info"].items()
            ]
        )
    )
    cols = date_cols + categorical_cols + numeric_cols
    conv = data["conversations"]
    value = conv[-1]["value"]
    for v in value:
        inputs = v["input"]
        if v["command"] == "Filter":
            bool_args = v["command_args"]["bool_args"]
            if isinstance(bool_args,dict):
                query_func = parse_bool_args(bool_args, cols, date_cols, inputs)
                bool_args = query_func
                v["command_args"]["bool_args"] = bool_args

    return data


if __name__ == "__main__":
    # with open("/home/qyhuang/project/spider/dsl_data/nl2sql_train_dsl.json") as f:
    with open("val_sql.json") as f:
        datas = json.load(f)
    new_datas = []
    for data in datas:
        new_data = bool_arg_2_query(data)
        new_datas.append(new_data)
    with open("val_sql_query.json", "w") as f:
        json.dump(new_datas, f, ensure_ascii=False)
