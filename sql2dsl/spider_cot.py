import os, sys
import json
import sqlite3
import traceback
import argparse
from process_sql_cot import get_sql
from generate_describe import generate_spider_dataset_describes
from nltk import word_tokenize
from itertools import chain


def tokenize(string):
    string = str(string)
    string = string.replace(
        "'", '"'
    )  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

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


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """

    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table["column_names_original"]
        table_names_original = table["table_names_original"]
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {"*": i}
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i

        return idMap


def get_schemas_from_json(fpath):
    """
    schemas = {"db_id":{"table_name":[column_names_original],...}}
    db_names = ["db_id",...]
    tables = {"db_id":{'column_names_original': column_names_original, 'table_names_original': table_names_original}}
    """
    with open(fpath) as f:
        data = json.load(f)
    db_names = [db["db_id"] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db["db_id"]
        schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db["column_names_original"]
        table_names_original = db["table_names_original"]
        tables[db_id] = {
            "column_names_original": column_names_original,
            "table_names_original": table_names_original,
        }
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables


def sql2dsl(gold_file, sql_path, table_file, output_file):
    with open(gold_file, "r") as f:
        golds = f.readlines()
    with open(sql_path) as f:
        sql_data = json.load(f)
    schemas, db_names, tables = get_schemas_from_json(table_file)
    # _, db_data_dict = generate_spider_dataset_describes(table_file)
    with open("sql_data/spider/spider_info.json")as f:
        db_data_dict = json.load(f)

    dsl_datas = []
    gold_datas = []
    error_num = 0
    correct_num = 0
    for i, data in enumerate(sql_data):
        db_id = data["db_id"]
        table_infos = db_data_dict[db_id]
        date_cols = list(
            chain(
                *[
                    list(value["date_cols_info"].keys())
                    for key, value in table_infos["db_info"].items()
                ]
            )
        )
        schema = schemas[db_id]
        table = tables[db_id]
        schema = Schema(schema, table)
        sql = data["query"]
        question = data["question"]
        # sql_label, dsl = get_sql(schema, sql, table, date_cols, question)
        try:
            sql_label, dsl = get_sql(schema, sql, table, date_cols, question)
        except Exception as e:
            # print(sql)
            # print(tokenize(sql))
            # print(e)
            # print("*"*100)
            error_num += 1
            continue
        if "skip_sql" in str(sql_label) or "skip_dsl" in str(dsl):
            error_num += 1
            continue
        else:
            correct_num += 1
            if "cspider" in sql_path:
                id = f"cspider_{correct_num}"
            else:
                id = f"spider_{correct_num}"
            dsl_data = {
                "id": id,
                "table_infos": table_infos,
                "sql": sql,
                "question":data["question"],
                "dsl":dsl
            }

        dsl_datas.append(dsl_data)
        print("error_num:", error_num, "correct_num:", correct_num)

    with open(output_file, "w") as f:
        json.dump(dsl_datas, f, ensure_ascii=False)
    # with open("gold_dev.txt","w")as f:
    #     for g in gold_datas:
    #         f.write(g)


if __name__ == "__main__":
    gold_file = "sql_data/spider/dev_gold.sql"

    table_file = "sql_data/spider/tables.json"
    sql_path = "sql_data/spider/train.json"
    output_file = "cot_data_ori/spider_train_cot.json"

    # table_file = "sql_data/spider/tables.json"
    # sql_path = "sql_data/spider/sample_sql.json"
    # output_file = "dsl_data/sample_sql.json"

    sql2dsl(gold_file, sql_path, table_file, output_file)
