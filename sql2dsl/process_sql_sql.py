################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#   4. table_name以及column_name中没有"."," ","/","[]","%","#","!"
#   5. 日期和类别要加引号
#   6. 操作符要空格隔开
#
# val: number(float)/string(str)/sql(dict)/list
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, (agg_id, val_unit), val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'having': condition
#   'limit': "none"/limit value
#   'intersect': "none"/sql
#   'except': "none"/sql
#   'union': "none"/sql
# }
################################

"""
跳过
FROM (SELECT....)  parse_from
WHERE_OPS (SELECT....)  parse_value
1.where 只能是column;value 如果是column只能是简单的嵌套(select col from table);是个数值可以agg
2.groupby select中有没有agg的column
3.select 后面有 distinct,agg(column)不需要filter
"""


import json
import sqlite3
from nltk import word_tokenize
import pandas as pd
import numbers
import re
from copy import deepcopy
from numbers import Number

CLAUSE_KEYWORDS = (
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
)
JOIN_KEYWORDS = ("join", "on", "as")

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}
COND_OPS = ("and", "or")
SQL_OPS = ("intersect", "union", "except")
ORDER_OPS = ("desc", "asc")


OPTIMIZES = [
    "select",
    "from",
    "where",
    "group",
    "order",
    "limit",
    "intersect",
    "union",
    "except",
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
    "by",
    "join",
    "on",
    "as",
    "max",
    "min",
    "count",
    "sum",
    "avg",
    "and",
    "or",
    "-",
    "+",
    "*",
    "/",
    "distinct",
    ",",
]
OP_MAP = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
WHERE_OPSS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "like",
    "contains",
    "startswith",
    "endswith",
    "is",
    "in",
)


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {"*": "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = (
                    "__" + key.lower() + "." + val.lower() + "__"
                )
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry["table"].lower())
        cols = [str(col["column_name"].lower()) for col in entry["col_data"]]
        schema[table] = cols

    return schema


def tokenize(string):
    string = str(string)
    string = string.replace(
        "'", '"'
    )  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    # assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        val = string[qidx1 : qidx2 + 1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2 + 1 :]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ("!", ">", "<")
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx - 1]
        if pre_tok in prefix:
            toks = toks[: eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1 :]

    return toks


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == "as"]
    alias = {}
    for idx in as_idxs:
        alias[toks[idx + 1]] = toks[idx - 1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(
    toks, start_idx, tables_with_alias, schema, table_info, default_tables="none"
):
    """
    :returns next idx, column id
    """
    table_names_original = table_info["table_names_original"]
    column_names_original = table_info["column_names_original"]

    tok = toks[start_idx]
    if tok == "time_now":
        return start_idx + 1, "time_now"
    if tok == "null":
        return start_idx + 1, "null"

    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if "." in tok:  # if token is a composite
        alias, col = tok.split(".")
        key = tables_with_alias[alias] + "." + col
        # return start_idx+1, schema.idMap[key]

        col_id = schema.idMap[key]
        table_name_original = table_names_original[column_names_original[col_id][0]]
        column_name_original = column_names_original[col_id][1]
        column = table_name_original + "." + column_name_original
        return start_idx + 1, column

    assert (
        default_tables != "none" and len(default_tables) > 0
    ), "Default tables should not be none or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            # return start_idx+1, schema.idMap[key]

            col_id = schema.idMap[key]
            table_name_original = table_names_original[column_names_original[col_id][0]]
            column_name_original = column_names_original[col_id][1]
            column = table_name_original + "." + column_name_original
            return start_idx + 1, column

    assert False, "Error col: {}".format(tok)


def parse_col_unit(
    toks, start_idx, tables_with_alias, schema, table_info, default_tables="none"
):
    """
    :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        # agg_id = AGG_OPS.index(toks[idx])
        agg_id = toks[idx]
        idx += 1
        assert idx < len_ and toks[idx] == "(", "toks[idx] error1"
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(
            toks, idx, tables_with_alias, schema, table_info, default_tables
        )
        assert idx < len_ and toks[idx] == ")", "toks[idx] error2"
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    # agg_id = AGG_OPS.index("none")
    agg_id = "none"
    idx, col_id = parse_col(
        toks, idx, tables_with_alias, schema, table_info, default_tables
    )

    if isBlock:
        assert toks[idx] == ")", "toks[idx] error3"
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(
    toks, start_idx, tables_with_alias, schema, table_info, default_tables="none"
):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    col_unit1 = "none"
    col_unit2 = "none"
    # unit_op = UNIT_OPS.index('none')
    unit_op = "none"

    idx, col_unit1 = parse_col_unit(
        toks, idx, tables_with_alias, schema, table_info, default_tables
    )
    if idx < len_ and toks[idx] in UNIT_OPS:
        # unit_op = UNIT_OPS.index(toks[idx])
        unit_op = toks[idx]
        idx += 1
        idx, col_unit2 = parse_col_unit(
            toks, idx, tables_with_alias, schema, table_info, default_tables
        )

    if isBlock:
        assert toks[idx] == ")", "toks[idx] error4"
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema, table_info):
    """
    :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]  # assert error

    if idx + 1 < len_ and toks[idx + 1] == "as":
        idx += 3
    else:
        idx += 1
    # return idx, schema.idMap[key], key
    table_idx = schema.idMap[key]
    table_name_original = table_info["table_names_original"][table_idx]

    return idx, table_name_original, key


def parse_value(
    toks, start_idx, tables_with_alias, schema, table_info, default_tables="none"
):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    if toks[idx] == "select":
        # WHERE City_ID NOT IN (SELECT Host_city_ID FROM farm_competition) PASS(HQY)
        idx, val = parse_sql(toks, idx, tables_with_alias, schema, table_info)
        # val = "skip_sql"
    elif '"' in toks[idx]:  # token is a string value
        if isBlock:
            # multi string value,such as ('"A"','"B"')
            val = []
            while idx < len_ and toks[idx] != ")":
                if '"' in toks[idx]:
                    val.append(toks[idx])
                    idx += 1
                else:
                    idx += 1
        else:
            # singe string value,such as '"A"'
            val = toks[idx]
            idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            try:
                date = pd.to_datetime(toks[idx])
                val = toks[idx]
                idx += 1
            except:
                end_idx = idx
                while (
                    end_idx < len_
                    and toks[end_idx] != ","
                    and toks[end_idx] != ")"
                    and toks[end_idx] != "and"
                    and toks[end_idx] not in CLAUSE_KEYWORDS
                    and toks[end_idx] not in JOIN_KEYWORDS
                ):
                    end_idx += 1

                idx, val = parse_col_unit(
                    toks[start_idx:end_idx],
                    0,
                    tables_with_alias,
                    schema,
                    table_info,
                    default_tables,
                )
                idx = end_idx

    if isBlock:
        assert toks[idx] == ")", "toks[idx] error5"
        idx += 1

    return idx, val


def parse_condition(
    toks, start_idx, tables_with_alias, schema, table_info, default_tables="none"
):
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        agg_id = "none"
        if toks[idx] in AGG_OPS:
            # agg_id = AGG_OPS.index(toks[idx])
            agg_id = toks[idx]
            idx += 1
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, table_info, default_tables
        )
        not_op = False
        if toks[idx] == "not":
            not_op = True
            idx += 1

        assert (
            idx < len_ and toks[idx] in WHERE_OPS
        ), "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        op_id_ = toks[idx]
        idx += 1
        val1 = val2 = "none"
        if op_id == WHERE_OPS.index(
            "between"
        ):  # between..and... special case: dual values
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, table_info, default_tables
            )
            assert toks[idx] == "and", "and error"
            idx += 1
            idx, val2 = parse_value(
                toks, idx, tables_with_alias, schema, table_info, default_tables
            )
        else:  # normal case: single value
            idx, val1 = parse_value(
                toks, idx, tables_with_alias, schema, table_info, default_tables
            )
            val2 = "none"

        # conds.append((not_op, op_id, (agg_id, val_unit), val1, val2))
        conds.append((not_op, op_id_, (agg_id, val_unit), val1, val2))

        if idx < len_ and (
            toks[idx] in CLAUSE_KEYWORDS
            or toks[idx] in (")", ";")
            or toks[idx] in JOIN_KEYWORDS
        ):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(
    toks, start_idx, tables_with_alias, schema, table_info, default_tables="none"
):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == "select", "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        # agg_id = AGG_OPS.index("none")
        agg_id = "none"
        if toks[idx] in AGG_OPS:
            # agg_id = AGG_OPS.index(toks[idx])
            agg_id = toks[idx]
            idx += 1
        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, table_info, default_tables
        )
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema, table_info):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert "from" in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index("from", start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == "(":
            isBlock = True
            idx += 1

        if toks[idx] == "select":
            # PASS (HQY)
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema, table_info)
            # sql = "skip_sql"  #"intersect", "union", "except"
            table_units.append((TABLE_TYPE["sql"], sql))
        else:
            if idx < len_ and toks[idx] == "join":
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(
                toks, idx, tables_with_alias, schema, table_info
            )
            table_units.append((TABLE_TYPE["table_unit"], table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(
                toks, idx, tables_with_alias, schema, table_info, default_tables
            )
            if len(conds) > 0:
                conds.append("and")
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ")", ") error"
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, table_info, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != "where":
        return idx, []

    idx += 1
    idx, conds = parse_condition(
        toks, idx, tables_with_alias, schema, table_info, default_tables
    )
    return idx, conds


def parse_group_by(
    toks, start_idx, tables_with_alias, schema, table_info, default_tables
):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != "group":
        return idx, col_units

    idx += 1
    assert toks[idx] == "by", "by error"
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(
            toks, idx, tables_with_alias, schema, table_info, default_tables
        )
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(
    toks, start_idx, tables_with_alias, schema, table_info, default_tables
):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = "asc"  # default type is 'asc'

    if idx >= len_ or toks[idx] != "order":
        return idx, val_units

    idx += 1
    assert toks[idx] == "by", "by error"
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        agg_id = "none"
        if toks[idx] in AGG_OPS:
            # agg_id = AGG_OPS.index(toks[idx])
            agg_id = toks[idx]
            idx += 1

        idx, val_unit = parse_val_unit(
            toks, idx, tables_with_alias, schema, table_info, default_tables
        )
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ",":
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(
    toks, start_idx, tables_with_alias, schema, table_info, default_tables
):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != "having":
        return idx, []

    idx += 1
    idx, conds = parse_condition(
        toks, idx, tables_with_alias, schema, table_info, default_tables
    )
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == "limit":
        idx += 2
        return idx, int(toks[idx - 1])

    return idx, "none"


def parse_sql(toks, start_idx, tables_with_alias, schema, table_info):
    isBlock = False  # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == "(":
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(
        toks, start_idx, tables_with_alias, schema, table_info
    )
    sql["from"] = {"table_units": table_units, "conds": conds}
    # select clause
    _, select_col_units = parse_select(
        toks, idx, tables_with_alias, schema, table_info, default_tables
    )
    idx = from_end_idx
    sql["select"] = select_col_units
    # where clause
    idx, where_conds = parse_where(
        toks, idx, tables_with_alias, schema, table_info, default_tables
    )
    sql["where"] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(
        toks, idx, tables_with_alias, schema, table_info, default_tables
    )
    sql["groupBy"] = group_col_units
    # having clause
    idx, having_conds = parse_having(
        toks, idx, tables_with_alias, schema, table_info, default_tables
    )
    sql["having"] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(
        toks, idx, tables_with_alias, schema, table_info, default_tables
    )
    sql["orderBy"] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql["limit"] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ")", ") error"
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = "none"
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema, table_info)
        # skip (HQY)
        IUE_sql = "skip_sql"
        sql[sql_op] = IUE_sql

    return idx, sql


def parse_sql_dsl(sql, date_cols, dsl):
    """sql嵌套sql情况：1.from；2.condition中的value1；3.("intersect", "union", "except")"""

    if "skip_sql" in str(sql) or "skip_dsl" in str(dsl):
        return dsl

    # 1.sql嵌套:from，sql["from"]["table_units"] = [("sql",{...})]
    if "sql" in sql["from"]["table_units"][0]:
        dsl = parse_sql_dsl(sql["from"]["table_units"][0][1], date_cols, dsl)
    else:
        dsl = sql2dsl_from(sql, dsl)
    from_length = len(dsl)

    # 2.sql嵌套:condition中的val1,sql["where"][-2] = {...}, condition=in
    dsl_in = []
    flag_val1 = False
    if len(sql["where"]):
        conds_convert = []
        for i, conds in enumerate(sql["where"]):
            if i % 2 == 0:
                conds = list(conds)
                if isinstance(conds[-2], dict):
                    if flag_val1:
                        # 多个条件嵌套(a>sql and b>sql),目前只支持嵌套一个
                        raise ValueError("val1有多个嵌套")
                    if conds[1] in ["in", ">", ">=", "=", "<", "<=", "!="]:
                        dsl_in = parse_sql_dsl(conds[-2], date_cols, [])
                        # bool_args中的previous_output替换成dsl output name
                        output_name = dsl_in[-1]["output"][0]
                        index = output_name.split("_")[-1]
                        output_name = output_name.replace(
                            index, str(int(index) + from_length)
                        )
                        conds[-2] = f'"{output_name}"'
                        flag_val1 = True
                    else:
                        raise ValueError("val1嵌套，condition为{conds[1]}不支持")

                # 备注：dsl_in此时放进去会影响后续dsl的命名，最后再放进去
            conds_convert.append(conds)

        sql["where"] = conds_convert
        dsl = sql2dsl_where(sql, dsl, date_cols, dsl_in)

    if len(sql["groupBy"]):
        dsl = sql2dsl_groupby(sql, dsl, date_cols)
        return dsl

    dsl = sql2dsl_orderby(sql, dsl)

    dsl = sql2dsl_select(sql, dsl)

    # if "skip_sql" in str(sql) or "skip_dsl" in str(dsl):
    #     return dsl

    # dsl顺序1.from;2.嵌套dsl_in；3.后续的dsl
    dsl = dsl[0:from_length] + dsl_in + dsl[from_length:]

    # dsl的output后缀索引重置
    new_dsl = []
    for i, d in enumerate(dsl):
        # 重置output
        output = d["output"][0].split("_")
        output = output[0:-1]
        output.append(str(i))
        output = "_".join(output)
        d["output"] = [output]

        # 重置input
        if i == 0:
            # dsl中第一个input不用改
            pass
        elif len(dsl_in) and i == from_length:
            # dsl_in中的第一个input不用改
            pass
        elif i == from_length + len(dsl_in):
            # dls_in后第一个dsl的input为join之后的output，不是dsl_in后的output
            if from_length > 0:
                d["input"] = dsl[from_length - 1]["output"]
            else:
                # 没有join不用改
                pass
            # dsl_in后第一个dsl的input加上dls_in的最后一个output
            if len(dsl_in):
                # TODO BUG
                d["input"].append(dsl[i - 1]["output"][0])
        else:
            d["input"] = dsl[i - 1]["output"]
        new_dsl.append(deepcopy(d))
    return new_dsl


def sql2dsl_from(sql, dsl):
    if "skip_sql" in str(sql) or "skip_dsl" in str(dsl):
        return dsl
    froms = sql["from"]
    conds = froms["conds"]
    table_units = froms["table_units"]
    if len(conds):
        if len(conds[::2]) + 1 > len(table_units):
            # left_on 和 right_on两个键，12条数据
            return ["skip_dsl"]

        ons = []
        hows = []
        for i, cond in enumerate(conds[::2]):
            left_on = cond[2][1][1][1]
            right_on = cond[3][1]
            # ons.append([left_on, right_on])
            t1 = left_on.split(".")[0]
            key1 = left_on.split(".")[1]
            t2 = right_on.split(".")[0]
            key2 = right_on.split(".")[1]
            ons.append({t1: [key1], t2: [key2]})
            hows.append("inner")
        input = [table_unit[1] for table_unit in table_units]
        command = "Join"
        output = [f"df_join_{len(dsl)}"]
        dsl.append(
            {
                "input": input,
                "output": output,
                "command": command,
                "command_args": {
                    "ons": ons,
                    "how": hows,
                },
            }
        )
    return dsl


def single_where_args(cond, date_cols, col=None):
    assert cond[1] in WHERE_OPSS, "cond error"
    if col != None:
        column = col
    else:
        val_unit = cond[2][1]
        column = val_unit[1][1]
    # not op
    not_op = False
    if cond[0] == True:
        not_op = True

    if cond[1] == "between":
        val1 = cond[-2]
        val2 = cond[-1]
        # column是时间列value需要是str
        if column.split(".")[-1] in date_cols:
            if isinstance(val1, Number):
                val1_split = str(val1).split(".")
                if val1_split[-1] == "0":
                    val1 = str(int(val1))
                else:
                    val1 = str(val1)
            if isinstance(val2, Number):
                val2_split = str(val2).split(".")
                if val2_split[-1] == "0":
                    val2 = str(int(val2))
                else:
                    val2 = str(val2)
        args = {
            "bool_columns": [column],
            "condition": "between",
            "not": not_op,
            "value": [val1, val2],
        }
    else:
        condition = cond[1]
        if condition in ["=", "is"]:
            # = -> ==
            condition = "=="
        if isinstance(cond[-2], list):
            val = cond[-2]
        else:
            val = [cond[-2]]

        # column是时间列value需要是str
        if column.split(".")[-1] in date_cols:
            val_new = []
            for v in val:
                if isinstance(v, Number):
                    v_split = str(v).split(".")
                    if v_split[-1] == "0":
                        v = str(int(v))
                    else:
                        v = str(v)
                val_new.append(v)
            val = val_new

        args = {
            "bool_columns": [column],
            "condition": condition,
            "not": not_op,
            "value": val,
        }
    return args


def get_input(sql, dsl):
    if len(dsl):
        input = dsl[-1]["output"]
    else:
        input = [sql["from"]["table_units"][0][1]]
    return input


def sql2dsl_where(sql, dsl: list, date_cols, dsl_in):
    if "skip_sql" in str(sql) or "skip_dsl" in str(dsl):
        return dsl
    # 1.where case
    # 预处理解析value_unit的四则运算，以及把column重命名
    conds = sql["where"]
    new_conds = []
    for i, cond in enumerate(conds):
        if i % 2 == 0:
            # cond=(not_op, op_id, (agg,val_unit), val1, val2)
            new_cond = []
            new_cond.extend(cond[0:2])
            # 1.解析value_unit
            value_unit_agg = cond[2]
            val_unit = value_unit_agg[1]
            agg_id = value_unit_agg[0]
            assert agg_id == "none", f"where中agg_id不为none,agg_id:{agg_id}"
            unit_op = val_unit[0]
            col_unit1 = val_unit[1]
            if unit_op != "none":
                # unit_op != "none"进行FourOperation操作
                dsl, new_val_unit = four_operation(dsl, sql, val_unit, dsl_in=dsl_in)
            else:
                # 有join的列名为table.column;没有join的为column
                new_col = rename_col(sql, col_unit1[1])
                new_val_unit = ("none", ("none", new_col, False), "none")
            new_cond.append((agg_id, new_val_unit))

            # 2.解析val=col_unit,str,float TODO T1.Adults + T1.Kids
            if isinstance(cond[-2], tuple) and cond[1] == "like":
                raise ValueError(f"like 只能跟str,{cond}")
            if cond[-1] != "none" and cond[1] != "between":
                raise ValueError(f"val2不为none,op只能是between,{cond}")

            # 解析val1
            if isinstance(cond[-2], tuple):
                # val1是column，col_unit转成column
                val1 = cond[-2][1]
                if len(sql["from"]["table_units"]) == 1:
                    val1 = val1.split(".")[-1]
            elif isinstance(cond[-2], list):
                # in("a","b")
                val1 = cond[-2]
            elif isinstance(cond[-2], str):
                # val1为str类型删除最外层'',有可能是日期
                try:
                    date = pd.to_datetime(cond[-2])
                    val1 = cond[-2]
                except:
                    val1 = cond[-2][1:-1]
            elif isinstance(cond[-2], Number):
                val1 = cond[-2]
            elif isinstance(cond[-2], dict):
                # val1为嵌套sql
                # val1 = cond[-2]
                raise ValueError(f"cond中的val1类型未知,{type(cond[-2])}")
            else:
                raise ValueError(f"cond中的val1类型未知,{type(cond[-2])}")

            # 解析val2
            if isinstance(cond[-1], str) and cond[1] == "between":
                # val2为日期('"2007-11-05"'),去除最外层''
                val2 = cond[-1][1:-1]
            else:
                # val2为数值或者是默认值none
                val2 = cond[-1]

            if cond[1] == "like":
                # "%NY%"
                if "[" in val1 or "_" in val1 or "*" in val1 or "^" in val1:
                    raise ValueError(f"op为like时val1中含有转义字符,{val1}")
                if val1[0] == "%" and val1[-1] == "%":
                    val1 = val1[1:-1]
                    new_cond[1] = "contains"
                elif val1[0] == "%" and val1[-1] != "%":
                    new_cond[1] = "endswith"
                    val1 = val1[1:]
                elif val1[0] != "%" and val1[-1] == "%":
                    new_cond[1] = "startswith"
                    val1 = val1[0:-1]
                else:
                    # 没有%
                    val1 = val1
                    new_cond[1] = "contains"

            new_cond.extend([val1, val2])
            new_conds.append(tuple(new_cond))
        else:
            new_conds.append(cond)

    for i, cond in enumerate(new_conds):
        if i % 2 == 0:
            if i != 0:
                args_ = single_where_args(cond, date_cols)
                if len(args) == 1:
                    key = list(args.keys())[0]
                    if key == new_conds[i - 1]:
                        value = list(args.values())[0]
                        value.append(args_)
                        args = {key: value}
                    else:
                        args = {new_conds[i - 1]: [args, args_]}
                else:
                    args = {new_conds[i - 1]: [args, args_]}
            else:
                args = single_where_args(cond, date_cols)

    # Bool DSL
    if len(conds):
        output = [f"df_filter_{len(dsl)}"]
        input = get_input(sql, dsl)
        filer_dsl = {
            "input": input,
            "output": output,
            "command": "Filter",
            "command_args": {
                "bool_args": args,
                "columns": ["all"],
                "index": "null",
                "axis": "null",
                "slice": "null",
                "type": "select",
            },
        }
        dsl.append(filer_dsl)
    return dsl


def sql2dsl_groupby(sql, dsl, date_cols):
    if "skip_sql" in str(sql) or "skip_dsl" in str(dsl):
        return dsl
    groupby = sql["groupBy"]
    groupby_dsl = []
    if len(groupby):
        bys = []
        for gb in groupby:
            bys.append(gb[1])

        having = sql["having"]
        if len(having):
            for hv in having[::2]:
                val_unit_agg = hv[2]
                if val_unit_agg[0] == "none":
                    # groupby后having没有agg的数据，3条
                    return ["skip_dsl"]
            # 1.先解析having中四则运算
            new_having = []
            for i, cond in enumerate(having):
                if i % 2 == 0:
                    # cond=(not_op, op_id, (agg,val_unit), val1, val2)
                    assert cond[0] == False or (
                        not cond[1] in ["like", "in", "exists", "is"]
                    ), f"not_p为False,op_id不能为like,cond:{cond}"
                    new_cond = []
                    new_cond.extend(cond[0:2])
                    # 1.解析value_unit
                    value_unit_agg = cond[2]
                    val_unit = value_unit_agg[1]
                    agg_id = value_unit_agg[0]
                    unit_op = val_unit[0]
                    col_unit1 = val_unit[1]
                    if unit_op != "none":
                        # unit_op != "none"进行FourOperation操作
                        four_operation_args, new_val_unit = four_operation(
                            dsl, sql, val_unit, groupby=True
                        )
                        if four_operation_args == None:
                            # 重复的四则运算
                            pass
                        else:
                            input = get_input(sql, dsl)
                            output = [f"df_four_operation_{len(dsl)}"]
                            dsl.append(
                                {
                                    "input": input,
                                    "output": output,
                                    "command": "FourOperation",
                                    "command_args": four_operation_args,
                                }
                            )
                    else:
                        # 有join的列名为table.column;没有join的为column
                        new_col = rename_col(sql, col_unit1[1])
                        new_val_unit = ("none", ("none", new_col, False), "none")
                    new_cond.append((agg_id, new_val_unit))
                    # val1不能为"none""
                    val1 = cond[-2]
                    val2 = cond[-1]
                    assert val1 != "none" and isinstance(
                        val1, Number
                    ), f"val1不为数值,cond:{val1}"
                    new_cond.extend([val1, val2])
                    new_having.append(tuple(new_cond))
                else:
                    new_having.append(cond)
            sql["having"] = new_having

        orderby = sql["orderBy"]
        if len(orderby):
            if len(orderby[1]) > 1:
                # orderby多列,无
                return ["skip_dsl"]
            val_unit_agg = orderby[1][0]
            if val_unit_agg[0] == "none":
                # groupby后orderby没有agg的列且不是by列，10条
                return ["skip_dsl"]
            # 1.orderby1列，先进行四则运算
            new_val_unit_aggs = []
            for agg_id, val_unit in orderby[1]:
                if val_unit[0] != "none":
                    four_operation_args, new_val_unit = four_operation(
                        dsl, sql, val_unit, groupby=True
                    )
                    if four_operation_args == None:
                        # 重复的四则运算
                        pass
                    else:
                        input = get_input(sql, dsl)
                        output = [f"df_four_operation_{len(dsl)}"]
                        dsl.append(
                            {
                                "input": input,
                                "output": output,
                                "command": "FourOperation",
                                "command_args": four_operation_args,
                            }
                        )
                else:
                    col = val_unit[1][1]
                    new_col = rename_col(sql, col)
                    new_val_unit = ("none", ("none", new_col, False), "none")
                new_val_unit_aggs.append((agg_id, new_val_unit))
            new_orderby = (orderby[0], new_val_unit_aggs)
            sql["orderBy"] = new_orderby

        select = sql["select"]
        # (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
        # (True, [('none', ('none', ('none', 'employees.department_id', False), 'none'))])
        if len(select):
            if select[0] == True:
                # groupby后select distinct,6条数据（by多列distinct某列）
                return ["skip_dsl"]
            for s in select[1]:
                # val_unit中的unit_op!=none 和 col_unit agg_id!=none,无
                if s[1][0] != "none" or s[1][1][0] != "none":
                    return ["skip_dsl"]
                if s[0] == "none" and (not s[1][1][1] in bys) and s[1][1][1] != 0:
                    # TODO agg_id为none，且column不在bys中，且column不为*(select *)，550条数据
                    return ["skip_dsl"]

    if len(groupby):
        bys = []
        for gb in groupby:
            bys.append(gb[1])
        input = get_input(sql, dsl)
        output = [f"df_groupby_agg_{len(dsl)}"]

        bys_col = []
        for col in bys:
            col = rename_col(sql, col)
            bys_col.append(col)
        dsl.append(
            {
                "input": input,
                "output": output,
                "command": "GroupbyAgg",
                "command_args": {"by": bys_col, "agg_args": {}},
            }
        )

        having = sql["having"]
        if len(having):
            # 再解析having的条件
            for i, cond in enumerate(new_having):
                if i % 2 == 0:
                    val_unit_agg = cond[2]
                    val_unit = val_unit_agg[1]
                    col = val_unit[1][1]
                    col = rename_col(sql, col)
                    agg = val_unit_agg[0]
                    distinct_flag = val_unit[1][2]
                    agg = convert_agg(agg, distinct_flag)
                    dsl[-1]["command_args"]["agg_args"][col] = [agg]
                    bool_column = f"({col},{agg})"
                    if i != 0:
                        args_ = single_where_args(cond, date_cols, bool_column)
                        if len(args) == 1:
                            key = list(args.keys())[0]
                            if key == having[i - 1]:
                                value = list(args.values())[0]
                                value.append(args_)
                                args = {key: value}
                            else:
                                args = {having[i - 1]: [args, args_]}
                        else:
                            args = {having[i - 1]: [args, args_]}
                    else:
                        args = single_where_args(cond, date_cols, bool_column)

            output = [f"df_filter_{len(dsl)}"]
            input = get_input(sql, dsl)
            filer_dsl = {
                "input": input,
                "output": output,
                "command": "Filter",
                "command_args": {
                    "bool_args": args,
                    "columns": ["all"],
                    "index": "null",
                    "axis": "null",
                    "slice": "null",
                    "type": "select",
                },
            }
            groupby_dsl.append(filer_dsl)

        orderby = sql["orderBy"]
        if len(orderby):
            # 2.orderby1列，解析orderby
            asc = orderby[0]
            if asc == "asc":
                asc = True
            else:
                asc = False
            val_unit_agg = orderby[1][0]
            val_unit = val_unit_agg[1]
            col = val_unit[1][1]
            new_col = rename_col(sql, col)
            agg = val_unit_agg[0]
            distinct_flag = val_unit[1][2]
            agg = convert_agg(agg, distinct_flag)
            bool_column = f"({new_col},{agg})"
            dsl[-1]["command_args"]["agg_args"][new_col] = [agg]
            if len(groupby_dsl):
                input = groupby_dsl[-1]["output"]
            else:
                input = dsl[-1]["output"]
            orderby_dsl = {
                "input": input,
                "output": [f"df_sort_values_{len(dsl)+len(groupby_dsl)}"],
                "command": "SortValues",
                "command_args": {"by": [bool_column], "ascending": asc},
            }
            groupby_dsl.append(orderby_dsl)

        select = sql["select"]
        if len(select):
            select_cols = []
            for sel in select[1]:
                agg = sel[0]
                val_unit = sel[1]
                col = val_unit[1][1]
                col = rename_col(sql, col)
                if agg != "none":
                    # select嵌套groupby agg
                    distinct_flag = val_unit[1][2]
                    agg = convert_agg(agg, distinct_flag)
                    agg_args = dsl[-1]["command_args"]["agg_args"]
                    if col in agg_args:
                        # col已经在agg的dict中
                        if not agg in agg_args[col]:
                            agg_args[col].append(agg)
                    else:
                        agg_args[col] = [agg]
                    dsl[-1]["command_args"]["agg_args"] = agg_args

                    col = f"({col},{agg})"
                select_cols.append(col)

            limit = sql["limit"]
            if limit != "none":
                if limit == 1:
                    index = [0]
                    axis = 0
                    slice = False
                else:
                    index = [0, limit]
                    axis = 0
                    slice = True
            else:
                index = "null"
                axis = "null"
                slice = "null"
            if len(select_cols):
                columns = select_cols
            else:
                columns = "null"

            # if limit != "none" and agg_flag:
            #     # TODO groupby后select中的agg怎么执行
            #     pass

            # 有having的话select可以合并到having中
            if len(having):
                if groupby_dsl[-1]["command"] == "Filter":
                    groupby_dsl[-1]["command_args"]["index"] = index
                    groupby_dsl[-1]["command_args"]["axis"] = axis
                    groupby_dsl[-1]["command_args"]["slice"] = slice
                    groupby_dsl[-1]["command_args"]["columns"] = columns
                else:
                    groupby_dsl[-2]["command_args"]["index"] = index
                    groupby_dsl[-2]["command_args"]["axis"] = axis
                    groupby_dsl[-2]["command_args"]["slice"] = slice
                    groupby_dsl[-2]["command_args"]["columns"] = columns
            else:
                if len(groupby_dsl):
                    input = groupby_dsl[-1]["output"]
                else:
                    input = dsl[-1]["output"]
                filter_dsl = {
                    "input": input,
                    "output": [f"df_filter_{len(dsl)+len(groupby_dsl)}"],
                    "command": "Filter",
                    "command_args": {
                        "bool_args": "null",
                        "columns": columns,
                        "index": index,
                        "axis": axis,
                        "slice": slice,
                        "type": "select",
                    },
                }
                groupby_dsl.append(filter_dsl)
    dsl.extend(groupby_dsl)
    return dsl


def sql2dsl_orderby(sql, dsl):
    if "skip_sql" in str(sql) or "skip_dsl" in str(dsl):
        return dsl
    orderby = sql["orderBy"]
    if len(orderby):
        val_unit_agg = orderby[1][0]
        agg_id = val_unit_agg[0]
        val_unit = val_unit_agg[1]
        if agg_id != "none":
            # orderby之前做agg
            return ["skip_dsl"]
        if val_unit[1][2] == True:
            # orderby之前做distinct，无
            return ["skip_dsl"]

        new_val_units = []
        for agg_id, val_unit in orderby[1]:
            assert agg_id == "none", f"orderby中agg_id不为none,agg_id:{agg_id}"
            if val_unit[0] != "none":
                # orderby之前做四则运算
                dsl, new_val_unit = four_operation(dsl, sql, val_unit)
                new_val_units.append((agg_id, new_val_unit))
            else:
                col = val_unit[1][1]
                new_col = rename_col(sql, col)
                new_val_unit = (agg_id, ("none", ("none", new_col, False), "none"))
                new_val_units.append(new_val_unit)
        new_orderby = (orderby[0], new_val_units)

        # 生成orderby的dsl
        asc = new_orderby[0]
        if asc == "asc":
            asc = True
        else:
            asc = False
        columns = []
        for _, val_unit in new_orderby[1]:
            col = val_unit[1][1]
            columns.append(col)
        input = get_input(sql, dsl)
        orderby_dsl = {
            "input": input,
            "output": [f"df_sort_values_{len(dsl)}"],
            "command": "SortValues",
            "command_args": {"by": columns, "ascending": asc},
        }
        dsl.append(orderby_dsl)

    return dsl


def four_operation(dsl, sql, val_unit, groupby=False, dsl_in=[]):
    unit_op = val_unit[0]
    col_unit1 = val_unit[1]
    col_unit2 = val_unit[2]
    # unit_op和col_unit2同时有或同时没有
    assert (unit_op != "none" and col_unit2 != "none") or (
        unit_op == "none" and col_unit2 == "none"
    ), "four_operation error"
    # col_unit不能嵌套agg和distinct
    agg_id1 = col_unit2[0]
    distinct1 = col_unit2[2]
    agg_id2 = col_unit2[0]
    distinct2 = col_unit2[2]
    if (
        (agg_id1 != "none")
        or (distinct1 != False)
        or (agg_id2 != "none")
        or (distinct2 != False)
    ):
        raise ValueError(f'(agg_id != "none") or (distinct != False),{val_unit}')

    if len(sql["from"]["table_units"]) == 1:
        # 单表
        input = get_input(sql, dsl)
        if col_unit1[1] == "time_now" or col_unit1[1] == "null":
            col1 = col_unit1[1]
        else:
            col1 = col_unit1[1].split(".")[1]
        input = get_input(sql, dsl)
        if col_unit2[1] == "time_now" or col_unit2[1] == "null":
            col2 = col_unit2[1]
        else:
            col2 = col_unit2[1].split(".")[1]
    else:
        # 多表
        input = [dsl[-1]["output"][0]]
        col1 = col_unit1[1]
        col2 = col_unit2[1]

    new_col = col1 + "_" + OP_MAP[unit_op] + "_" + col2
    output = [f"df_four_operation_{len(dsl)}"]

    if "time_now" in col1:
        assert OP_MAP[unit_op] == "sub", f"col1为time_now，operation_type不为sub"
        command_args = {
            "value": [col1],
            "columns": [col2],
            "index": "null",
            "operation_type": "rsub",
            "new_cols": [new_col],
        }
    elif "time_now" in col2:
        assert OP_MAP[unit_op] == "sub", f"col2为time_now，operation_type不为sub"
        command_args = {
            "value": [col2],
            "columns": [col1],
            "index": "null",
            "operation_type": OP_MAP[unit_op],
            "new_cols": [new_col],
        }
    else:
        command_args = {
            "value": "null",
            "columns": [col1, col2],
            "index": "null",
            "operation_type": OP_MAP[unit_op],
            "new_cols": [new_col],
        }
    four_operation = {
        "input": input,
        "output": output,
        "command": "FourOperation",
        "command_args": command_args,
    }
    new_val_unit = ("none", ("none", new_col, False), "none")
    if groupby:
        if new_col in str(dsl) or new_col in str(dsl_in):
            print("----------->", None)
            return None, new_val_unit

        return four_operation["command_args"], new_val_unit
    else:
        if new_col in str(dsl) or new_col in str(dsl_in):
            # "select 名称 from 电视台 where 上线时间 - 开播时间 < ( select avg ( 上线时间 - 开播时间 ) from 电视台 )"
            pass
        else:
            dsl.append(four_operation)
        return dsl, new_val_unit


def rename_col(sql, col):
    # 有join的列名为table.column;没有join的为column
    if col == 0:
        col = "all"
    if len(sql["from"]["table_units"]) == 1:
        # 单表
        if "." in col:
            col = col.split(".")[1]
    return col


def convert_agg(agg, distinct_flag):
    if agg == "count" and distinct_flag == True:
        # count(distinct column)->nunique
        agg = "nunique"
    if agg == "avg":
        agg = "mean"
    return agg


def select_limit(dsl, sql, select):
    """
    (False, [('count', ('none', ('none', 'video_games.gtype', True), 'none'))])
    1.select
    2.agg
    3.COUNT(DISTINCT city)情况
    """
    input = get_input(sql, dsl)

    limit = sql["limit"]
    if limit != "none":
        if limit == 1:
            index = [0]
            axis = 0
            slice = False
        else:
            index = [0, limit]
            axis = 0
            slice = True
    else:
        index = "null"
        axis = "null"
        slice = "null"

    agg_id = select[1][0][0]
    if agg_id == "none":
        # select
        cols = []
        for s in select[1]:
            val_unit = s[1]
            col = val_unit[1][1]
            col = rename_col(sql, col)
            cols.append(col)

        filter_dsl = {
            "input": input,
            "output": [f"df_filter_{len(dsl)}"],
            "command": "Filter",
            "command_args": {
                "bool_args": "null",
                "columns": cols,
                "index": index,
                "axis": axis,
                "slice": slice,
                "type": "select",
            },
        }
        dsl = merge_filter(dsl, filter_dsl)
    else:
        # agg
        limit = sql["limit"]
        if limit != "none":
            # limit不为none，先进行limit再agg
            if limit != 1:
                filter_dsl = {
                    "input": input,
                    "output": [f"df_filter_{len(dsl)}"],
                    "command": "Filter",
                    "command_args": {
                        "bool_args": "null",
                        "columns": "null",
                        "index": [0, limit],
                        "axis": 0,
                        "slice": True,
                        "type": "select",
                    },
                }
                dsl = merge_filter(dsl, filter_dsl)
            else:
                return ["skip_dsl"]
        aggs = {}
        for s in select[1]:
            val_unit = s[1]
            col = val_unit[1][1]
            distinct_flag = val_unit[1][2]
            agg_id = s[0]
            agg_id = convert_agg(agg_id, distinct_flag)
            col = rename_col(sql, col)
            # agg_id为count，用{all:count}表示
            if agg_id == "count":
                col = "all"
            # 同一column多个agg
            if col in aggs:
                aggs[col].append(agg_id)
            else:
                aggs[col] = [agg_id]
        dsl.append(
            {
                "input": input,
                "output": [f"df_statics_column_{len(dsl)}"],
                "command": "StaticsColumn",
                "command_args": {"aggregate": aggs, "is_index": "null"},
            }
        )

    return dsl


def merge_filter(dsl, filter_dsl):
    if len(dsl):
        if dsl[-1]["command"] == "Filter":
            command_args = dsl[-1]["command_args"]
            if command_args["index"] == "null" and command_args["columns"] == ["all"]:
                index = filter_dsl["command_args"]["index"]
                columns = filter_dsl["command_args"]["columns"]
                axis = filter_dsl["command_args"]["axis"]
                slice = filter_dsl["command_args"]["slice"]
                dsl[-1]["command_args"]["index"] = index
                dsl[-1]["command_args"]["columns"] = columns
                dsl[-1]["command_args"]["axis"] = axis
                dsl[-1]["command_args"]["slice"] = slice

                return dsl

    dsl.append(filter_dsl)
    return dsl


def sql2dsl_select(sql, dsl):
    if "skip_sql" in str(sql) or "skip_dsl" in str(dsl):
        return dsl
    select = sql["select"]
    if len(select):
        # 1.unit_op不为none，先进行四则运算；2.agg和none不能同时出现;3agg和distinct，COUNT(DISTINCT city)
        #'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
        for s in select[1]:
            if s[1][1][0] != "none":
                # col_unit agg_id!=none，无
                return ["skip_dsl"]
            if s[1][1][2] == True and s[0] != "count":
                # col_unit distinct=True,但是agg_id不为count（不是COUNT(DISTINCT city)情况），无
                return ["skip_dsl"]
            if select[0] == True and s[0] != "none":
                # distinct=True,agg_id不为none，无
                return ["skip_dsl"]

        aggs = [s[0] for s in select[1]]
        if "none" in aggs:
            # none 和其他agg不能共存，3条；TODO（删除none的val_unit）
            for agg in aggs:
                if agg != "none":
                    return ["skip_dsl"]

        """
        解析：1.COUNT(DISTINCT city)；2.unip_op不为none；3.agg不为none
        """
        # 1.select之前做四则运算
        new_val_units = []
        for s in select[1]:
            val_unit = s[1]
            unit_op = val_unit[0]
            if unit_op != "none":
                dsl, new_val_unit = four_operation(dsl, sql, val_unit)
            else:
                new_val_unit = s[1]
            new_val_units.append((s[0], new_val_unit))
        select = (select[0], new_val_units)

        # 2.Distinct去重
        if select[0] == True:
            new_val_units = []
            subset = []
            for s in select[1]:
                val_unit = s[1]
                col_unit1 = val_unit[1]
                col = col_unit1[1]
                col = rename_col(sql, col)
                subset.append(col)
                new_val_unit = ("none", ("none", col, False), "none")
                new_val_units.append((s[0], new_val_unit))

            # add DropDuplicates dsl
            input = get_input(sql, dsl)
            dsl.append(
                {
                    "input": input,
                    # "output": [f"df_distinct_{len(dsl)}"],
                    # "command": "Distinct",
                    # "command_args": {
                    #     "columns": subset,
                    # },
                    "output": [f"df_drop_duplicates_{len(dsl)}"],
                    "command": "DropDuplicates",
                    "command_args": {
                        "subset": subset,
                        "keep": "null",
                        "subset_only": True,
                    },
                }
            )
            # set Distinct False
            select = (False, new_val_units)
            return dsl

        # 3.agg and select
        dsl = select_limit(dsl, sql, select)

    return dsl


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def toks_optimize(toks):
    # 财政预算收入', '(', '亿', ')'->'财政预算收入(亿)'
    i = 0
    while i < len(toks) - 1:
        break_flag = False
        start_index = None
        end_index = None
        for i, tok in enumerate(toks):
            if (tok == "(" or tok == "（") and i != 0:
                if not (toks[i - 1] in OPTIMIZES):
                    start_index = i - 1
                    for j, tok in enumerate(toks[start_index:]):
                        if tok == ")" or tok == "）":
                            end_index = start_index + j + 1
                            new_tok = "".join(toks[start_index:end_index])
                            break_flag = True
                            break
            if break_flag:
                break

        if start_index != None and end_index != None:
            # print(toks)
            new_tok = "".join(toks[start_index:end_index])
            toks.insert(end_index, new_tok)
            del toks[start_index:end_index]
    return toks


def datesub_to_time_now(toks):
    # 交易日 > DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR) -->TIME_NOW- 交易日 < 1year
    time_char = ""
    toks_str = "".join(toks)
    if "date_add" in toks_str:
        raise ValueError("不支持dateadd！")
    if "date_sub" in toks_str or "datesub" in toks_str:
        for i, tok in enumerate(toks):
            if tok == "date_sub(current_date()" or tok == "datesub(current_date()":
                datesub_index = i
            if tok == "interval":
                interval_index = i
        start_index = datesub_index - 2
        col = toks[start_index]
        comparison_op = toks[datesub_index - 1]
        if comparison_op == ">=":
            comparison_op = "<="
        elif comparison_op == ">":
            comparison_op = "<"
        elif comparison_op == "<":
            comparison_op = ">"
        elif comparison_op == "<=":
            comparison_op = ">="
        num = toks[interval_index + 1]
        time_char = toks[interval_index + 2]
        for i in range(interval_index, len(toks)):
            if toks[i] == ")":
                end_index = i
                break
        time_now = ["time_now", "-", col, comparison_op, num]

        new_toks = []
        for i, tok in enumerate(toks):
            if i < start_index:
                new_toks.append(tok)
            elif i == start_index:
                new_toks.extend(time_now)
            elif i > end_index:
                new_toks.append(tok)
        toks = new_toks

    return toks, time_char


def add_time_unit(bool_args, time_char, question, date_cols):
    if len(bool_args) == 1:
        if "or" in bool_args:
            parsed_args = [
                add_time_unit(arg, time_char, question, date_cols)
                for arg in bool_args["or"]
            ]
            return {"or": parsed_args}
        else:
            parsed_args = [
                add_time_unit(arg, time_char, question, date_cols)
                for arg in bool_args["and"]
            ]
            return {"and": parsed_args}
    else:
        bool_col = bool_args["bool_columns"][0]
        # "(结束时间_sub_起始时间,mean)"
        if bool_col[0] == "(" and bool_col[-1] == ")":
            bool_col_ = bool_col[1:-1]
        else:
            bool_col_ = bool_col
        c_s = bool_col_.split("_")
        if (
            bool_col in date_cols
            or "time_now" in bool_col
            or (c_s[0] in date_cols or c_s[-1] in date_cols)
        ):
            v = bool_args["value"]
            if isinstance(v[0], Number):
                time_unit = get_time_unit(time_char, question)
                bool_args["value"] = [f"{v[0]}-{time_unit}"]

        return bool_args


def get_time_unit(time_char, question):
    units_map = {
        "years": ["years", "year", "年"],
        "quarters": ["quarters", "quarter", "季度", "个季度"],
        "months": ["months", "month", "月", "个月"],
        "weeks": ["weeks", "week", "周"],
        "days": ["days", "day", "天"],
        "hours": ["hours", "hour", "小时", "个小时"],
        "minutes": ["minutes", "minute", "分钟"],
        "seconds": ["seconds", "second", "秒", "秒钟"],
    }
    # 正则匹配模板
    pattern = r"(\d+|[一二三四五六七八九十两]{1,5})\s*(小时|个小时|分钟|秒钟|秒|天|年|季度|个季度|周|月|个月|year|month|quarter|week|day|hour|minute|second)"
    if not len(time_char):
        # 通过正则匹配query中的单位
        matches = re.findall(pattern, question, re.IGNORECASE)
        if len(matches) != 1:
            time_chars = [match[1] for match in matches]
            if len(set(time_chars)) == 1:
                time_char = matches[0][1]
            else:
                raise ValueError(f"{question}中有多个时间单位，无法确定哪个！")
        else:
            time_char = matches[0][1]
    for key, value in units_map.items():
        if time_char in value:
            time_unit = key
    return time_unit


def post_timeinterval(dsl, date_cols, time_char, question):
    if "skip_dsl" in str(dsl):
        return dsl

    # 加上单位years,quarters,months,weeks,days,hours,minutes,seconds
    """
    input:
    {"columns": ["time_now_sub_成立时间"],"condition": "!=","not": false,"value": [12.0]}
    return:
    {"columns": ["time_now_sub_成立时间"],"condition": "!=","not": false,"value": ["12.0-years"]}
    """

    for old_dsl in dsl:
        if old_dsl["command"] == "Filter":
            if old_dsl["command_args"]["bool_args"] == "null":
                return dsl

            bool_args = deepcopy(old_dsl["command_args"]["bool_args"])
            new_bool_args = add_time_unit(bool_args, time_char, question, date_cols)
            old_dsl["command_args"]["bool_args"] = new_bool_args

    return dsl


def is_2_equal(toks):
    # is、is not、<> 在where and or 后面两个字符
    indexs = [
        index
        for index, value in enumerate(toks)
        if value in ["where", "and", "or"] and index + 2 < len(toks)
    ]
    for index in indexs:
        if toks[index + 2] == "is" and toks[index + 3] == "not":
            toks[index + 2] = "!="
            toks[index + 3] = ""
        elif toks[index + 2] == "is":
            toks[index + 2] = "="
        elif toks[index + 2] == "<" and toks[index + 3] == ">":
            toks[index + 2] = "!="
            toks[index + 3] = ""
    toks = [tok for tok in toks if len(tok)]

    return toks


def remove_agg_op_as(toks):
    select_indexs = []
    from_indexs = []
    for i, tok in enumerate(toks):
        if tok == "select":
            select_indexs.append(i)
        elif tok == "from":
            from_indexs.append(i)
    assert len(select_indexs) == len(from_indexs), f"select和from的数量不同"
    new_toks = []
    for i, select_index in enumerate(select_indexs):
        from_index = from_indexs[i]
        if i != 0:
            new_toks += toks[from_indexs[i - 1] : select_indexs[i]]

        for j, tok in enumerate(toks[select_index:from_index]):
            if j == 0:
                new_toks.append(tok)
            elif (tok != "as") and (toks[select_index:from_index][j - 1] != "as"):
                # 删除as和as下一个token
                new_toks.append(tok)

    # 加上最后一个from到结尾的token
    new_toks = toks[0 : select_indexs[0]] + new_toks
    new_toks += toks[from_indexs[-1] :]
    # if toks != new_toks:
    #     print(toks)
    #     print(new_toks)
    #     print("*" * 100)

    return new_toks


def get_sql(schema, query, table_info, date_cols, question=""):
    # 切分query
    toks = tokenize(query)
    toks = toks_optimize(toks)

    # is转换成=；is not转成!=；<>转成!=
    toks = is_2_equal(toks)

    # remove agg_op as
    toks = remove_agg_op_as(toks)

    # datesub转成time_now
    toks, time_char = datesub_to_time_now(toks)

    # table起别名
    tables_with_alias = get_tables_with_alias(schema.schema, toks)

    # query转sql语法树
    _, sql = parse_sql(toks, 0, tables_with_alias, schema, table_info)

    # sql语法树转dsl
    dsl = parse_sql_dsl(sql, date_cols, dsl=[])

    # DSL中时的时间数值加单位
    dsl = post_timeinterval(dsl, date_cols, time_char, question)
    return sql, dsl
