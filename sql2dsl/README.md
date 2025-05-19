# SQL2DSL

### Assumptions:

- 1. sql is correct
- 2. only table name has alias
- 3. no intersect/union/except
- 4. table_name以及column_name中没有特殊字符"."," ","/","[]","%","#","!"
- 5. no time_now
### preparation

1.table_file里面必须包含"table_names"和"column_names",参考spider数据集中tables.json
2.sql_file里面必须包含"query"和"db_id",参考spider数据集中dev.json

### run

传入table_file,sql_file以及output_file,运行spider.py