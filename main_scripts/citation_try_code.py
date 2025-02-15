#2025 1月27日大规模引证准确率试验

# -*- coding: utf-8 -*-
#测评大语言模型生成正确的引证，也就是citation的能力的脚本
#这个脚本目前只针对开源模型，而且会用一个函数来实现测评大语言模型是否
#生成正确的引证的功能的脚本

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import requests
import random
import time
import concurrent.futures
import threading
from tqdm import tqdm
#设置随机数种子
random.seed(30)

print('检查gpu： ',torch.cuda.is_available())

#加载参考数据
fname='/data/yuchen_llm_eval/data/参考资料倒序/1000_sample_ref_reverse_order.json'
with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)


prompts=[item['prompt'] for item in data_citation_combo]
print('prompts数量',len(prompts))
print('独特prompts数量',len(set(prompts)))

max_retries = 3

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
            # model_name可以输入如下多个选择
            # Qwen2.5_7B，Qwen2.5_14B，Qwen2.5_32B，Qwen2.5_72B，Qwen2_7B，Qwen2_57B，Qwen2_72B
            # Llama3.3_70B，glm_4_9B_chat，deepseek
            payload = json.dumps({
                'text': prompt,
                "model_name":"Qwen2_7B",
                "temperature":0,
                "max_tokens":20
                })
            # 发送请求并获取响应
            response = requests.request("POST", url, headers=headers, data=payload,timeout=5)
            # 检查响应状态
            response.raise_for_status()  # 如果响应错误，抛出异常
        
            if response.status_code == 200:
                # print('response text:\n',response.text)
                # result = (json.loads(response.text)['text'][0]).partition(prompt)[-1]
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
                print('try number:',count)
                print("response.status_code:{}, response.text:{}".format(response.status_code, response.text))
                if count>= max_retries:
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
            

print("\n模型调用开始\n")
cnt_irr=0

#处理每条原始的引证数据，每条原始的引证数据是一个字典dict，有category,prompt,output三个字段
#返回每条原始引证数据在一个字典中，包含原有的category,prompt,output，并且加上模型的回答。
def item_processing(dic:dict):
    prompt = dic['prompt']
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
    end_index = ans_raw.find(']')
    ans_final = ans_raw[0:end_index]
    dic_new={
        'category': category,
        'output': output,
        'prompt': prompt,
        'response': ans_final
    }
    return dic_new
    

lock = threading.Lock()

def parallel_processing(items):
    #重要步骤，创建文件时写入开方括号
    filename_='/data/yuchen_llm_eval/data/参考资料倒序/Qwen2_7B_1000_reverse_order_0208__00.json'
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

# 调试用
# def parallel_processing_test(items):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         for result in tqdm(executor.map(item_processing, items), total=len(items)):
#             with lock:
#                 print(result)

parallel_processing(data_citation_combo)
#parallel_processing_test(data_citation_combo[1921:1922])