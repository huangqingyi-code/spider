from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import os, copy, json
import random

from get_system import generate_prompt,generate_prompt_cot


def python_code_format_output(values):
    for value in values:
        command_args = value["command_args"]
        args = {}
        for k, v in command_args.items():
            if v != "null":
                args[k] = v
        value["command_args"] = args

    funcs = ""
    for value in values:
        input = json.dumps(value["input"], separators=(",", ":"), ensure_ascii=False)
        output = value["output"]
        command = value["command"]
        command_args = json.dumps(
            value["command_args"], separators=(",", ":"), ensure_ascii=False
        )
        func = f"{output[0]}={command}({input},{command_args})" + "\n"
        funcs += func
    return funcs.strip()


def list_format_output(values):
    for value in values:
        command_args = value["command_args"]
        args = {}
        for k, v in command_args.items():
            if v != "null":
                args[k] = v
        value["command_args"] = args

    ls = []
    for value in values:
        input = value["input"]
        output = value["output"]
        command = value["command"]
        command_args = value["command_args"]
        ls.append([input, command, command_args, output])
    ls = json.dumps(ls, separators=(",", ":"), ensure_ascii=False)
    return ls


def convert_llama_efficient(
    input_data_path,
    output_data_path,
    data_type="schema",
    output_format=None,
):
    i = 0
    model_to_load = "/home/baize/weights/WizardLM-13B-V1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)
    # 数据转换成llama-efficient格式
    with open(input_data_path) as f:
        datas = json.load(f)
    data_news = []
    token_num = []
    dsl_length = []
    for data in datas:
        data_new = {}
        table_infos = data["table_infos"]
        system = generate_prompt(table_infos, data_type, wrap=False)
        data_new["system"] = system
        convs = data["conversations"]
        # query,response
        if convs[-2]["from"] == "human":
            query = convs[-2]["value"]
        else:
            raise ValueError("-2不是human")

        if convs[-1]["from"] == "gpt":
            # output放在最后，明天跟command相关
            values = convs[-1]["value"]
            for value in values:
                output = value["output"]
                del value["output"]
                value["output"] = output
            if output_format == None:
                response = json.dumps(values, ensure_ascii=False)
            elif output_format == "python_code":
                response = python_code_format_output(values)
            elif output_format == "list":
                response = list_format_output(values)
            else:
                raise ValueError(f"output_format error,{output_format}")
            token_num.append(len(tokenizer.encode(response)))
            dsl_length.append(len(values))
        else:
            raise ValueError("-1不是gpt")

        # history
        history = []
        data_new["prompt"] = query
        data_new["query"] = ""
        data_new["response"] = response
        data_new["history"] = history
        if "val" in input_data_path:
            data_new["id"] = data["id"]
            data_news.append(data_new)
        else:
            length = len(tokenizer.encode(system + query + response))
            if length > 4080:
                i += 1
                print(i)
                print("length:", length)
            else:
                data_news.append(data_new)
    print("平均token长度：", sum(token_num) / len(token_num))
    print("平均dsl长度：", sum(dsl_length) / len(dsl_length))
    with open(output_data_path, "w") as f:
        json.dump(data_news, f, ensure_ascii=False)


def convert_llama_efficient_cot(
    input_data_path, output_data_path, prompt_type="prompt_cot", data_type="schema"
):
    i = 0
    model_to_load = "/alg_vepfs/public/hqy/pretrain_weights/WizardLM-13B-V1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)
    # 数据转换成llama-efficient格式
    with open(input_data_path) as f:
        datas = json.load(f)
    data_news = []
    for data in datas:
        data_new = {}
        table_infos = data["table_infos"]
        # cot = json.dumps(data["cot"], separators=(",", ":"), ensure_ascii=False)
        cot = data["cot"]
        system = generate_prompt_cot(
            table_infos, prompt_type, cot, data_type, wrap=False
        )
        query = data["question"]
        if prompt_type == "prompt_cot":
            response = json.dumps(
                data["cot"], separators=(",", ":"), ensure_ascii=False
            )
        else:
            response = json.dumps(
                data["dsl"], separators=(",", ":"), ensure_ascii=False
            )
        data_new["system"] = system
        history = []
        data_new["prompt"] = query
        data_new["query"] = ""
        data_new["response"] = response
        data_new["history"] = history

        length = len(tokenizer.encode(system + query + response))
        if length > 4080:
            i += 1
            print(i)
            print("length:", length)
        else:
            data_news.append(data_new)

    with open(output_data_path, "w") as f:
        json.dump(data_news, f, ensure_ascii=False)


def merge_data():
    with open("dsl_data/sql/train.json") as f:
        train_sql = json.load(f)
    with open("dsl_data/zjuici/train.json") as f:
        train_manual = json.load(f)
        random.shuffle(train_manual)
        train_manual = random.sample(train_manual, 12000)
    print("train_sql:", len(train_sql), "train_manual:", len(train_manual))

    with open("dsl_data/sql/val.json") as f:
        val_sql = json.load(f)
        # random.shuffle(val_sql)
        # val_sql = random.sample(val_sql, 1600)
    with open("dsl_data/zjuici/val.json") as f:
        val_manual = json.load(f)
        random.shuffle(val_manual)
        val_manual = random.sample(val_manual, 1600)
    print("val_sql:", len(val_sql), "val_manual:", len(val_manual))

    train = train_sql + train_manual
    val = val_sql + val_manual

    with open("dsl_data/train_all.json", "w") as f:
        json.dump(train, f, ensure_ascii=False)
    with open("dsl_data/val_all.json", "w") as f:
        json.dump(val, f, ensure_ascii=False)


if __name__ == "__main__":
    # merge_data()
    convert_llama_efficient_cot(
        "cot_datas.json",
        "cot_datas_dsl_trans.json",
        prompt_type="prompt_cot_dsl",
        data_type="schema",
    )
