import json
import re
from tqdm import tqdm
import requests
import time
import random
import concurrent.futures
import threading
from tqdm import tqdm


fname='/data/yuchen_llm_eval/data/用于人工标注数据/label_Llama3.3-70B_data_raw_0202.json'
with open(fname, 'r', encoding='utf-8') as file:
    prompt_for_artificial_data=json.load(file)

def citation_generation(prompt):
    url = "http://101.132.252.74:20012/proxy_generate"

    headers = {
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Connection': 'keep-alive'
        }
    max_retries=3
    count = 0
    response = None  # 初始化 response 变量
    while True:
        try:
            #model_name可以输入如下多个选择
            # Qwen2.5_7B，Qwen2.5_14B，Qwen2.5_32B，Qwen2.5_72B，Qwen2_7B，Qwen2_57B，Qwen2_72B
            # Llama3.3_70B
            payload = json.dumps({
                'text': prompt,
                "model_name":"Llama3.3-70B",
                "temperature":0,
                "max_tokens":2048
                })
            # 发送请求并获取响应
            response = requests.request("POST", url, headers=headers, data=payload,timeout=300)
            #print(response.text)
            # 检查响应状态
            response.raise_for_status()  # 如果响应错误，抛出异常
            if response.status_code == 200:
                #print('response text:\n',response.text)
                response_json = json.loads(response.text)
                if "text" in response_json:
                    result = response_json["text"][0].partition(prompt)[-1]
                res = {
                    'prompt': prompt,
                    'response': result
                }
                break
            else:
                count=count+1
                print("response.status_code:{}, response.text:{}".format(response.status_code, response.text))
                if count>=max_retries:
                    res = {
                        'prompt': prompt,
                        'response':  "RunTimeError Message\n\n" + response.text,
                    }
                    return res
                time.sleep(60)
        except Exception as e:
            count = count + 1
            print(f"请求失败: {e}, 正在重试... ({count}/{max_retries})")
            if count >= max_retries:
                if response:
                    result = "RunTimeError Message\n\n" + response.text
                    res = {
                        'prompt':prompt,
                        'response':result
                    }
                else:
                    result = "RunTimeError Message\n\nFailed to get a response from the server."
                    res = {
                        'prompt':prompt,
                        'response':result
                    }
                return res
    return res

#处理每条原始的引证数据，每条原始的引证数据是一个字典dict，有category,label_prompt,output三个字段
#返回每条原始引证数据在一个字典中，包含原有的category,prompt,output，并且加上模型的回答。
def item_processing(dic:dict):
    prompt = dic['label_prompt']
    #Llama3.3-70B
    # new_before_string='<s><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    # old_before_string='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    # prompt=prompt.replace(old_before_string,new_before_string)
    # new_div_string='<|im_end|>\n'
    # old_div_string='<|im_end|>\n<|im_start|>assistant\n'
    # prompt=prompt.replace(old_div_string,new_div_string)

    #

    category = dic['category']
    output= dic['output']
    try:
        ans_raw = citation_generation(prompt)['response']
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "prompt": prompt,
            "response": "{}: {}".format(type(e).__name__, e)
        }
    dic_new={
        'category': category,
        'prompt': prompt,
        'response': ans_raw
    }
    return dic_new

lock = threading.Lock()
def parallel_processing(items):
    #重要步骤，创建文件时写入开方括号
    filename_='/data/yuchen_llm_eval/data/用于人工标注数据/artifical_Llama3.3-70B_0204-01.json'
    with open(filename_, "a", encoding="utf-8") as f:
        f.write("[")
        f.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for result in tqdm(executor.map(item_processing, items), total=len(items)):
            with lock:
                with open(filename_, 'a', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                    f.write(',')

    with open(filename_, 'a', encoding='utf-8') as f:
        f.write(']')

parallel_processing(prompt_for_artificial_data[3:4])
