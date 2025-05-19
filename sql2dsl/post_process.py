import json
import os

def merge_dataset(data_dir):
    data_lst = os.listdir(data_dir)
    data_train = []
    data_dev = []
    for data_name in data_lst:
        if "train" in data_name:
            data_train.append(data_name)
        else:
            data_dev.append(data_name)
    def merge(data_lst:list,output_path):
        datas = []
        for name in data_lst:
            path = os.path.join(data_dir,name)
            with open(path,"r")as f:
                data = json.load(f)
            datas.extend(data)
        print(f"共有{len(datas)}条数据")
        merge_filter(datas,output_path)

    merge(data_train,"train_sql.json")
    merge(data_dev,"val_sql.json")



def merge_filter(datas,output_path):
    # for data in datas:
    #     conv = data["conversations"]
    #     value = conv[1]["value"]
    #     new_value = []
    #     for i,v in enumerate(value):
    #         if v["command"]=="Filter":
    #             command_args = v["command_args"]
    #             if command_args["bool_args"]=="null" and command_args["columns"]=="null" and command_args["index"]=="null":
    #                 print(conv)
    #                 print(data["sql"])
    #                 print("*"*100)
    #             else:
    #                 new_value.append(v)
    #         else:
    #             new_value.append(v)
            
    #     data["conversations"][1]["value"] = new_value
    with open(output_path,"w")as f:
        json.dump(datas,f,ensure_ascii=False)

if __name__=="__main__":
    # merge_filter("dsl_data/spider_train_dsl.json")
    merge_dataset("dsl_data")