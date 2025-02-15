# 开源模型处理Error

import json
import requests
import random
import time
from tqdm import tqdm

#设置随机数种子
random.seed(30)

#加载参考数据
fname='/data/yuchen_llm_eval/tangbo/full_res_gpt_4o_tb_our_0128_03.json'
with open(fname, 'r', encoding='utf-8') as file:
    content=file.read()
if content.endswith(',]'):
    content=content[:-2]+']'
if content.endswith(','):
    print('end with comma')
    content=content[:-1]+']'
data_citation_combo= json.loads(content)

output_file='/data/yuchen_llm_eval/tangbo/full_res_gpt_4o_tb_our_0128_03_new.json'

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
            #model_name可以输入如下多个选择，Qwen2.5_7B，Qwen2.5_14B，Qwen2.5_32B，Qwen2.5_72B，Qwen2_7B，Qwen2_57B，Qwen2_72B，Llama3.3_70B，glm_4_9B_chat
            payload = json.dumps({
                'text': prompt,
                "model_name":"Qwen2_57B",
                "temperature":0,
                "max_tokens":20
                })
            # 发送请求并获取响应
            response = requests.request("POST", url, headers=headers, data=payload,timeout=30)
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
            

print("\n模型调用开始\n")
cnt_irr=0

#处理每条原始的引证数据，每条原始的引证数据是一个字典dict，有category,prompt,output三个字段
#返回每条原始引证数据在一个字典中，包含原有的category,prompt,output，并且加上模型的回答。
def item_processing(dic:dict):
    prompt = dic['prompt']
    try:
        # 调用 citation_generation 函数获取新的 response
        new_response = citation_generation(prompt)['response']
        end_index = new_response.find(']')
        new_response = new_response[0:end_index]
        # 修改原字典中的 'response' 键的值
        dic['response'] = new_response
    except (TypeError, KeyError) as e:
        dic['response'] = "{}: {}".format(type(e).__name__, e)

    return dic
       
    
def process_list_and_write_to_file(data_list: list, output_file: str):
    # 创建一个新列表用于存储最终结果
    num = 0 
    handling_error_num=0
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("[")
        f.close()
    for item in tqdm(data_list, desc="Processing items"):
        if "RunTimeError Message\n\n" not in item.get("response"):
        #if item.get("response") != "RunTimeError Message\n\nFailed to get a response from the server":
            # 如果响应不是错误信息，直接添加到结果列表
            num+=1
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(item, f, indent=4, ensure_ascii=False)
                f.write(',')
        else:
             # 如果是错误信息，调用 item_processing 函数处理，直到响应不再是错误信息
            attempt_count = 0  # 用于记录当前元素的处理尝试次数
            last_result = item  # 用于存储最后一次调用的结果
            while attempt_count < 3:
                last_result = item_processing(last_result)  # 调用 item_processing 函数
                attempt_count += 1  # 尝试次数加一
                if "RunTimeError Message\n\n" not in last_result.get("response"):
                #if last_result.get("response") != "RunTimeError Message\n\nFailed to get a response from the server":
                    with open(output_file, 'a', encoding='utf-8') as f:
                        json.dump(last_result, f, indent=4, ensure_ascii=False)
                        f.write(',')
                    num += 1
                    handling_error_num += 1
                    break
            else:
                # 如果尝试了 3 次仍然没有成功，将最后一次调用的结果加入结果列表
                with open(output_file, 'a', encoding='utf-8') as f:
                    json.dump(last_result, f, indent=4, ensure_ascii=False)
                    f.write(',')
                num += 1
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(']')

    print(f"处理完成，成功处理了 {num} 个元素")
    print(f"处理完成，成功处理了 {handling_error_num} 个RunTimeError")

process_list_and_write_to_file(data_citation_combo, output_file)


