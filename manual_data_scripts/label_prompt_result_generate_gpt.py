import json
from tqdm import tqdm
import requests
import concurrent.futures
import threading
from tqdm import tqdm


key = ""
# gpt-4-turbo    gpt-4o    gpt-4o-mini 
model = "gpt-4o"

fname='/root/yuchen_llm_eval/data/用于人工标注数据/label_qwen_data_raw_0115.json'
with open(fname, 'r', encoding='utf-8') as file:
    prompt_for_artificial_data=json.load(file)
print(len(prompt_for_artificial_data))

def chat_with_gpt(query: str,
                  key: str = key,
                  model: str = model,
                  system_message: str = None,
                  temperature: float = 0,
                  retry_time: int = 5,
                  json_mode: bool = False
                  ):
    url = "https://apigateway.offline.xinyunews.cn/llm/v1/chat/completions"
    if system_message:
        message = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": query
            },
        ]
    else:
        message = [
            {
                "role": "user",
                "content": query
            }
        ]
    payload = {
        "model": model,
        "messages": message,
        "temperature": temperature
    }
    if json_mode:
        payload.update(response_format={"type": "json_object"})
    payload = json.dumps(payload)
    headers = {
        'Authorization': 'Bearer {}'.format(key),
        'Content-Type': 'application/json',
    }
    count = 0
    while True:
        try:
            response = requests.request("POST", url, headers=headers, data=payload, timeout=300)
            #print(response.text)
            result = json.loads(response.text)["choices"][0]["message"]["content"]
            break
        except Exception as e:
            count = count + 1
            print(e)
            if count > retry_time:
                raise Exception('ReturnCode.LLM_ERROR')
    return result

#处理每条原始的引证数据，每条原始的引证数据是一个字典dict，有category,label_prompt,output三个字段
#返回每条原始引证数据在一个字典中，包含原有的category,prompt,output，并且加上模型的回答。
def item_processing(dic:dict):
    prompt = dic['label_prompt']
    # 删掉千问模型的模板
    new_prompt = prompt.partition("<|im_end|>\n<|im_start|>user\n")[-1].partition("<|im_end|>\n<|im_start|>assistant\n")[0]

    category = dic['category']
    output= dic['output']
    try:
        ans_raw = chat_with_gpt(new_prompt)
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "gpt prompt": new_prompt,
            "response": "{}: {}".format(type(e).__name__, e)
        }
    dic_new={
        'category': category,
        'gpt prompt': new_prompt,
        'response': ans_raw
    }
    return dic_new

lock = threading.Lock()
def parallel_processing(items):
    #重要步骤，创建文件时写入开方括号
    filename_='/root/yuchen_llm_eval/data/用于人工标注数据/artifical_gpt4turbo_0120——00.json'
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
def parallel_processing_try(items):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for result in tqdm(executor.map(item_processing, items), total=len(items)):
            with lock:
                print(result)

parallel_processing(prompt_for_artificial_data)