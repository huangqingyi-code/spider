# import pandas as pd
# from datetime import timedelta
# import datetime
# import re

# # # 创建一个包含时间字符串的DataFrame
# data = {
#     "timestamp1": ["2012-11-05 10:00:00", "2023-11-02 11:15:00"],
#     "timestamp2": ["2023-11-02 11:15:00", "2023-11-08 14:45:00"],
# }

# df = pd.DataFrame(data)

# # # 将字符串转换为时间戳
# df["timestamp1"] = pd.to_datetime(df["timestamp1"])
# df["timestamp2"] = pd.to_datetime(df["timestamp2"])

# # # 计算两个时间戳的差异
# df["time_difference"] = df["timestamp2"] - df["timestamp1"]
# cols = ["Fdate","FRelateBrID","Fnote","Fauxqty"]
# df = pd.read_csv("/home/qyhuang/project/spider/vwICBill_28.csv")
# time_stamp = datetime.datetime.now().strftime("%Y-%m-%d")
# df["time_difference"] =pd.to_datetime(time_stamp)- pd.to_datetime(df["Fdate"])
# # print(df)

# # 打印结果

# query_ret = df.query("`time_difference` > '800.0 days'",engine='python')
# print(query_ret)


# def replace_time_difference(match):
#     number = match.group(1)
#     unit = match.group(2)
#     print(number)
#     print(unit)
#     # 根据单位进行替换
#     if unit.lower() == "hours":
#         # number = float(number) * 60
#         # return f"{number} minutes"
#         return f"{number} {unit}"
#     elif unit.lower() == "days":
#         number = float(number) * 3600
#         return f"{number} minutes"


# query_func = "'1 Hours' < time_difference < '2 Days'"
# pattern = re.compile(
#     r"(\d+)\s*(years|quarters|months|weeks|days|hours|minutes|seconds)", re.IGNORECASE
# )
# result = pattern.sub(replace_time_difference, query_func)
# print(result)


# time_stamp = datetime.datetime.now().strftime("%Y-%m-%d")
# query_func = "time_difference > 'Time_Now'"
# pattern = re.compile('time_now', re.IGNORECASE)
# query_func = pattern.sub(time_stamp, query_func)
# print(query_func)


# import pandas as pd
# import datetime

# # 创建一个包含时间字符串的DataFrame
# data = {'timestamp': ['2023-11-05', '2023-11-06', '2022-10-07']}
# df = pd.DataFrame(data)

# # 将字符串转换为时间戳
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# # 指定一个特定的时间点
# target_time = pd.to_datetime("2023.10.12")
# # target_time = datetime.datetime.now()
# print(target_time)

# # 筛选时间列大于特定时间点的行
# filtered_df = df[df['timestamp'] < target_time]
# query_ret = df.query("2020 < timestamp",engine='python')

# print(filtered_df)
# print(query_ret)

# import re

# # # 定义一个文本句子
# sentence = "在过去的三十天里，我们完成了很多工作，花了大约2HOURS阅读，300小时写作和30分钟锻炼。"
# sentence = "哪些经纪公司成立两年了"

# # 定义时间单位的正则表达式模式
# pattern = r'(\d+|[一二三四五六七八九十两]{1,4})\s*(小时|分钟|秒钟|天|年|季度|周|月|year|month|quarter|week|day|hour|minute|second)'

# # 使用正则表达式查找匹配的时间单位
# matches = re.findall(pattern, sentence,re.IGNORECASE)

# # 遍历匹配结果
# # print(len(matches))
# print(matches[0][1])
# for match in matches:
#     value, unit = match
#     print(f'找到了{value} {unit}')


# import pandas as pd

# df = pd.DataFrame({"a":[1,2,1,2,3,4],"b":["杭州","hangzhou","杭州","宁波","绍兴",pd.np.nan]})
# df_copy = pd.DataFrame({"a":[1,2],"b":["杭州","hangzhou"]})
# df_ret = df[~(~(df["b"].isin(df_copy["b"])))]
# df_ret = df[df["b"].isnull()]
# print(df_ret)

# import json
# x = "df_join_0=Join([\"Dogs\",\"Treatments\"],{\"ons\":[{\"Dogs\":[\"dog_id\"],\"Treatments\":[\"dog_id\"]}],\"how\":[\"inner\"]})\ndf_groupby_agg_1=GroupbyAgg([\"df_join_0\"],{\"by\":[\"Dogs.name\"],\"agg_args\":{\"all\":[\"count\"]}})\ndf_sort_values_2=SortValues([\"df_groupby_agg_1\"],{\"by\":[\"(all,count)\"],\"ascending\":true})\ndf_filter_3=Filter([\"df_sort_values_2\"],{\"bool_args\":\"null\",\"columns\":[\"Dogs.name\",\"Treatments.date_of_treatment\"],\"index\":[1],\"axis\":0,\"slice\":false,\"type\":\"select\"})"
# x = " df_filter_0=Filter([\"参考书\"],{\"bool_args\":{\"bool_columns\":[\"适用年级\"],\"condition\":\"!=\",\"not\":false,\"value\":[1.0-3.0]},\"columns\":[\"all\"],\"index\":\"null\",\"axis\":\"null\",\"slice\":\"null\",\"type\":\"select\"})\ndf_groupby_agg_1=GroupbyAgg([\"df_filter_0\"],{\"by\":[\"类型\"],\"agg_args\":{\"all\":[\"count\"],\"价格\":[\"max\"]}})\ndf_filter_2=Filter([\"df_groupby_agg_1\"],{\"bool_args\":{\"bool_columns\":[\"(all,count)\"],\"condition\":\">=\",\"not\":false,\"value\":[5.0]},\"columns\":[\"类型\",\"(价格,max)\"],\"index\":\"null\",\"axis\":\"null\",\"slice\":\"null\",\"type\":\"select\"})"
# def python_code_format(preds):
#     def parse_function_call(s):
#         # 移除变量赋值部分
#         s = s.split('=', 1)
#         right = s[1]

#         # output
#         output = s[0]

#         # 查找第一个左括号和最后一个右括号的位置
#         start = right.find('(')
#         end = right.rfind(')')

#         # 提取函数名和参数部分
#         command = right[:start].strip()
#         params_str = right[start+1:end].strip()

#         # 切分input和command_args
#         bracket_index = params_str.index("{")
#         input_str = params_str[0:bracket_index-1]
#         command_args_str = params_str[bracket_index:]
#         print("---->",command_args_str)
#         input = json.loads(input_str)
#         command_args = json.loads(command_args_str)

#         return input,output,command,command_args
#     values = []
#     preds = preds.split("\n")
#     for pred in preds:
#         input,output,command,command_args = parse_function_call(pred)
#         values.append({"input":input,"command":command,"command_args":command_args,"output":output})
#     return values
# y = python_code_format(x)
# print(y)


import pandas as pd
import datetime
import numpy as np
# query nan,(.isnull()  .notnull())
df = pd.DataFrame({'a': [1,2,3,4],'b':[1,3,1,2],'c':[np.nan,np.nan,1,1],'d':['aa','bb','ab','cd'],'e':['ff','bb','ff','gg']})
# df_d = df["d"]
# ret = df.query("not d.str.contains('a')", engine='python')
# ret = df.query("d_ddd.str.startswith('a')", engine='python')
# ret = df.query("not `d`.str.endswith('a')", engine='python')
# res = pd.Series([1,2,1,2]).tolist()
# dsl_name = "df_dsl"
# dsl_name = res
# ret = df.query("`bsum,mean` > @dsl_name", engine='python')
# ret = df.query("a in [2,3]", engine='python')
# ret = df.query("`a` in `b`", engine='python')
# ret = df.query("`c`.isnull()", engine='python')
# ret = df.query("`e` in ['bb']")
ret = df.query("not(not a>2 or b>1)")
print(ret)
