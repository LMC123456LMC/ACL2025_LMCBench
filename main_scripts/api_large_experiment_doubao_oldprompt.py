# coding:utf-8
import json
import sys
import requests
from tqdm import tqdm
import concurrent.futures
import threading
import random
import time


key_dict={
    "gpt": "",
    "baichuan":"",
    "moonshot": "",
    "doubao": "",
    "deepseek_v3":"",
    "glm":""

}
url_dict={
    "gpt": "https://apigateway.offline.xinyunews.cn/llm/v1/chat/completions",
    "baichuan":"https://api.baichuan-ai.com/v1/chat/completions",
    "moonshot": "https://api.moonshot.cn/v1/chat/completions",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "deepseek_v3":"https://api.deepseek.com/chat/completions",
    "glm":"https://open.bigmodel.cn/api/paas/v4/chat/completions"
}

# gpt-4-turbo    gpt-4o    gpt-4o-mini 
model_name_dict={
    "gpt":"gpt-4o",
    "moonshot":"moonshot-v1-32k",
    "doubao":"ep-20250123113406-7vzr5",
    "deepseek_v3":"deepseek-chat",
    "glm":"glm-4-plus",
    "baichuan":"baichuan4-turbo"

}
#非常重要，本次实验模型的名称。
model_name_here='doubao'
fname='/data/yuchen_llm_eval/data/是否给出引证前所有内容/1000_sample_ref_notall_02.json'

with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)

prompts=[item['prompt'] for item in data_citation_combo]
print('prompts数量',len(prompts))
print('独特prompts数量',len(set(prompts)))

def chat_with_api(user_msg: str,
                  assistant_msg: str,
                  key: str ,
                  url: str ,
                  model: str ,
                  system_message: str = None,
                  temperature: float = 0,
                  retry_time: int = 6,
                  json_mode: bool = False
                  ):
    #url = "http://47.88.65.188:8405/v1/chat/completions"
    if system_message:
        query = "<im_user>{}<user_end><im_assistant>{}".format(user_msg, assistant_msg)
        message = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": query
            }
        ]
    else:
        #system_message = "我将给你提供某AI助手与其用户的一段对话，其中AI助手的发言被截断了一部分，请根据上下文语境进行补充。注意：用户的发言以'<im_user>'开始，以'<user_end>'结束，AI助手的发言以'<im_assistant>'开始。"
        query = "<im_user>{}<user_end><im_assistant>{}".format(user_msg, assistant_msg)
        prompt = "{}{}".format(user_msg, assistant_msg)
        message = [   
            # {
            #     "role": "system",
            #     "content": system_message
            # },
            {
                "role": "user",
                "content": prompt
            },
        ]
    payload = {
        "model": model,
        "messages": message,
        "temperature": temperature,
        "max_tokens":20
    }
    if json_mode:
        payload.update(response_format={"type": "json_object"})
    payload = json.dumps(payload)
    headers = {
        'Authorization': 'Bearer {}'.format(key),
        'Content-Type': 'application/json',
    }
    count = 0
    response = None  # 初始化 response 变量
    while True:
        try:
            response = requests.request("POST", url, headers=headers, data=payload, timeout=300)
            if response.status_code == 200:
                # print('response text:\n',response.text)
                result = json.loads(response.text)["choices"][0]["message"]["content"]
                res = {
                    'prompt': query,
                    'response': result
                }
                break
            else:
                count=count+1
                print('try number:',count)
                print("response.status_code:{}, response.text:{}".format(response.status_code, response.text))
                if count>= retry_time:
                    res = {
                        'prompt': query,
                        'response':  "RunTimeError Message\n\n" + response.text,
                    }
                    return res
                time.sleep(60)
        except Exception as e:
            count = count + 1
            print('try number:',count)
            print("response.status_code:{}, response.text:{}".format(response.status_code, response.text))
            print("Full error is {}, full response is {}".format(e, response))
            if count >= retry_time:
                if response:
                    result = "RunTimeError Message\n\n" + response.text
                    res = {
                        'prompt':query,
                        'response':result
                    }
                else:
                    result = "RunTimeError Message\n\nFailed to get a response from the server."
                    res = {
                        'prompt':query,
                        'response':result,
                    }
                return res
    return res

#处理每条原始的引证数据，每条原始的引证数据是一个字典dict，有category,label_prompt,output三个字段
#返回每条原始引证数据在一个字典中，包含原有的category,prompt,output，并且加上模型的回答。
def item_processing(dic:dict):
    # Add a random sleep between 0 and 2 seconds
    time.sleep(random.uniform(0, 2))
    prompt = dic['prompt']

    #切分出需要的user_message和assistant message
    user_message_here = prompt.partition("<|im_end|>\n<|im_start|>user\n")[-1].partition("<|im_end|>\n<|im_start|>assistant\n")[0] 
    
    str = "\n***********\n请只关注'***********'之前内容的参考资料部分。输出下面这段话中的最后一句话挂载的引证，不输出这段话，形式如'[abcd1234]'：\n"
    assistant_message_here = str + prompt.partition("<|im_end|>\n<|im_start|>assistant\n")[-1]
    
    category = dic['category']
    output= dic['output']
    try:
        if "你是一个中文大语言模型。你在做一个百科问答任务，" in prompt:
            ans_raw = chat_with_api(user_msg=user_message_here,
                              assistant_msg=assistant_message_here,
                              key=key_dict[model_name_here],
                                url=url_dict[model_name_here],
                                model=model_name_dict[model_name_here],
                                json_mode=False)
        else:
            ans_raw = chat_with_api(user_msg=user_message_here,
                              assistant_msg=assistant_message_here,
                              key=key_dict[model_name_here],
                                url=url_dict[model_name_here],
                                model=model_name_dict[model_name_here],
                                json_mode=False)
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "prompt": "<im_user>{}<user_end><im_assistant>{}".format(user_message_here, assistant_message_here),
            "response": "{}: {}".format(type(e).__name__, e)
        }
    dic_new={
        'category': category,
        "output": output,
        "prompt": ans_raw['prompt'],
        "response": ans_raw['response']
    }
    return dic_new

lock = threading.Lock()
def parallel_processing(items):
    #重要步骤，创建文件时写入开方括号
    filename_='/data/yuchen_llm_eval/data/是否给出引证前所有内容/doubao_1000_notall_0207__00.json'
    with open(filename_, "a", encoding="utf-8") as f:
        f.write("[")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for result in tqdm(executor.map(item_processing, items), total=len(items)):
            with lock:
                #print(result)
                with open(filename_, 'a', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                    f.write(',')

    with open(filename_, 'a', encoding='utf-8') as f:
        f.write(']')


# 调试用
# def parallel_processing_test(items):
#     #重要步骤，创建文件时写入开方括号
#     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#         for result in tqdm(executor.map(item_processing, items), total=len(items)):
#             with lock:
#                 print(result)



print('现在运行的模型名称:',model_name_dict[model_name_here])  
# random_combo=random.sample(data_citation_combo,100)      
parallel_processing(data_citation_combo)
#parallel_processing_test(data_citation_combo[8:10])
