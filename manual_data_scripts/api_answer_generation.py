# coding:utf-8
import json
import requests
from tqdm import tqdm
import re
import concurrent.futures
import threading

key = "sk-b12UmEtrvVSyfp64B47fC451643341A382603a10A490B221"
model = "gpt-4o"

key_dict={
    "gpt-4o": "sk-b12UmEtrvVSyfp64B47fC451643341A382603a10A490B221",
    "baichuan":"sk-84105483f548e0855736af7cbdd79150",
    "moonshot": "sk-RSwjF2Nb3vAC5MyI5jgsWZLCLJwoF2Ro7EyCcUpSwXtJcPg2",
    "doubao": "0608ec35-fca7-4552-b8ef-7434a8ad3652",
    "deepseek_v3":"sk-a68c6274279e41e89ef4c9d62fc2ea35",
    "glm":"003316014c2c4f7c932127a6486e2180.hscLE45CoqpiRaBE"

}
url_dict={
    "gpt-4o": "http://47.88.65.188:8405/v1/chat/completions",
    "baichuan":"https://api.baichuan-ai.com/v1/chat/completions",
    "moonshot": "https://api.moonshot.cn/v1/chat/completions",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "deepseek_v3":"https://api.deepseek.com/chat/completions",
    "glm":"https://open.bigmodel.cn/api/paas/v4/chat/completions"
}
model_name_dict={
    "moonshot":"moonshot-v1-32k",
    "doubao":"ep-20250123113406-7vzr5",
    "deepseek_v3":"deepseek-chat",
    "glm":"glm-4-plus",
    "baichuan":"baichuan4-turbo"

}

# file_output='/root/yuchen_llm_eval/data/try_result/gpt_try_res_0120_00.json'
# fname='/root/yuchen_llm_eval/data/新的验证结果/new_50%_combined_data_results2.5_7B_0110_01_stats2_cut.json'
# with open(fname, 'r', encoding='utf-8') as file:
#     data_citation_combo = json.load(file)

# data_citation_combo = data_citation_combo[1:2]

# prompts=[item['prompt'] for item in data_citation_combo]
# print('prompts数量',len(prompts))
# print('独特prompts数量',len(set(prompts)))

fname='/root/yuchen_llm_eval/data/用于人工标注数据/label_qwen_data_raw_0115.json'
with open(fname, 'r', encoding='utf-8') as file:
    prompt_for_artificial_data=json.load(file)

def chat_with_api(query: str,
                  key: str ,
                  url: str ,
                  model: str ,
                  system_message: str = None,
                  temperature: float = 0,
                  retry_time: int = 5,
                  json_mode: bool = False
                  ):
    #url = "http://47.88.65.188:8405/v1/chat/completions"
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
            print('response text:\n',response.text)
            result = json.loads(response.text)["choices"][0]["message"]["content"]
            break
        except Exception as e:
            count = count + 1
            print(e)
            if count > retry_time:
                return "RunTimeError Message"
    return result

#处理每条原始的引证数据，每条原始的引证数据是一个字典dict，有category,label_prompt,output三个字段
#返回每条原始引证数据在一个字典中，包含原有的category,prompt,output，并且加上模型的回答。
def item_processing(dic:dict):
    prompt = dic['label_prompt']
    #去掉问答数据中相关问题或者提示思路的部分
    # chunk_ref=prompt.partition("\n\n参考资料：\n")[-1].partition("相关问答：")[0].partition("提示思路：")[0].partition("\n\n\n结构化模版：\n")[0]
    # lis_here=prompt.partition(chunk_ref)
    # pre_ref=lis_here[0]
    # new_prompt=pre_ref
    #new_before_string='You are a helpful assistant.<|im_start|>user\n'
    #old_before_string='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    #prompt=prompt.replace(old_before_string,new_before_string)

    # 删掉千问模型的模板
    new_prompt = prompt.partition("<|im_end|>\n<|im_start|>user\n")[-1].partition("<|im_end|>\n<|im_start|>assistant\n")[0]

    category = dic['category']
    output= dic['output']
    try:
        ans_raw = chat_with_api(query=new_prompt,
                        key=key_dict['baichuan'],
                        url=url_dict['baichuan'],
                        model=model_name_dict['baichuan'])
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "gpt prompt": new_prompt,
            "response": "{}: {}".format(type(e).__name__, e)
        }
    dic_new={
        'category': category,
        'new prompt': new_prompt,
        'response': ans_raw
    }
    return dic_new

lock = threading.Lock()
def parallel_processing(items):
    #重要步骤，创建文件时写入开方括号
    filename_='/root/yuchen_llm_eval/data/用于人工标注数据/artifical_baichuan4_turbo_api_0124__01.json'
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
if __name__ == "__main__":
    print(chat_with_api(query='请进行简单自我介绍.',
                        key=key_dict['baichuan'],
                        url=url_dict['baichuan'],
                        model=model_name_dict['baichuan']))
    
    print('现在运行的模型名称:',model_name_dict['baichuan'])
    parallel_processing(prompt_for_artificial_data)
    # print(chat_with_api(query=prompt_for_artificial_data[48]['label_prompt'],
    #                     key=key_dict['moonshot'],
    #                     url=url_dict['moonshot'],
    #                     model=model_name_dict['moonshot']))
    # cnt_irr=0
    # cnt_wrong=0
    # #res_final = []

    # with open(file_output, "a", encoding="utf-8") as f:
    #     f.write("[")

    # for ind, item in enumerate(tqdm(data_citation_combo)):
    #     dic={}
    #     prompt = item['prompt']
    #     category = item['category']
    #     output= item['output']

    #     str = "\n***********\n请只关注'***********'之前内容的参考资料部分。输出下面这段话中的最后一句话挂载的引证，不输出这段话，形式如'[abcd1234]'：\n"
    #     new_prompt = prompt.partition("<|im_end|>\n<|im_start|>user\n")[-1].partition("<|im_end|>\n<|im_start|>assistant\n")[0] + str + prompt.partition("<|im_end|>\n<|im_start|>assistant\n")[-1]

    #     a = chat_with_gpt(
    #         query=new_prompt,
    #         key=key,
    #         model=model,
    #         json_mode=False)
    #     # print(a)

    #     # # 形如"ycJNEcVh]"
    #     # if len(a) == 9 and a[-1] == "]":
    #     #     cnt_wrong += 1
    #     #     print(a)
    #     #     a = "[" + a
    #     # # 形如"s7Y0cEcf"
    #     # if len(a) == 8 and re.match(r'[a-zA-Z0-9]{8}', a):
    #     #     cnt_wrong += 1
    #     #     print(a)
    #     #     a = "[" + a + "]"

    #     # 输出多个引证截取第一个
    #     if len(a) > 10:
    #         end_index = a.find(']')
    #         if end_index == 9:
    #             a = a[0:end_index+1]
    #         # cnt_wrong += 1
    #         # print(a)
    #         # match = re.search(r'\[[a-zA-Z0-9]{8}\]', a)
    #         # if match:
    #         #     a = match.group(0)

    #     if len(a) != 10:
    #         cnt_irr += 1
    #         print(a)

    #     dic = {
    #         'category': category,
    #         'output': output,
    #         'prompt': new_prompt,
    #         'response': a
    #     }
    #     #res_final.append(dic)

    #     with open(file_output, 'a', encoding='utf-8') as f:
    #         json.dump(dic, f, indent=4, ensure_ascii=False)
    #         f.write(',')

    # with open(file_output, 'a', encoding='utf-8') as f:
    #     f.write(']')

    # #print('可处理的不规则数据数',cnt_irr)
    # print('不规则数据数',cnt_irr)