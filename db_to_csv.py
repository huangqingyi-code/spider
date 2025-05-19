import pandas as pd
import json
import os
import shutil

sub_dir = "/home/qyhuang/project/spider/dusql_csv"

with open("/home/qyhuang/project/spider/sql_data/DuSQL/db_content.json")as f:
    contents = json.load(f)
with open("/home/qyhuang/project/spider/sql_data/DuSQL/db_schema.json")as f:
    schemas = json.load(f)
for i,content in enumerate(contents):
    db_id = content["db_id"]
    print(db_id)
    db_path = os.path.join(sub_dir,db_id)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.mkdir(db_path)
    schema = schemas[i]
    if schema["db_id"]!=db_id:
        raise ValueError("db_id不一样！")
    for table_name in content["tables"].keys():
        table = content["tables"][table_name]
        header = table["header"]
        cells = table["cell"]
        cols = [[]for _ in range(len(header))]
        for cell in cells:
            if len(cell)!=len(cols):
                raise ValueError("cell长度和header不一样！")
            for index,c in enumerate(cell):
                cols[index].append(c)
        data = {}
        for j,col in enumerate(header):
            data[col] = cols[j]
        table_path = f"{db_path}/{table_name}.csv"
        df = pd.DataFrame(data)
        df.to_csv(table_path,index=False)