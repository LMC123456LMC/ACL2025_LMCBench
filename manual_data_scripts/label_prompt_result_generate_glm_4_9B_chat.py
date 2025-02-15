import json
import re
from tqdm import tqdm
import requests
import time
import random
import concurrent.futures
import threading
from tqdm import tqdm

#加载参考数据
fname='/root/yuchen_llm_eval/data/新的验证结果/new_50%_combined_data_results2.5_7B_0110_01_stats2_cut.json'
with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)
prompts=[item['prompt'] for item in data_citation_combo]
print('prompts数量',len(prompts))
print('独特prompts数量',len(set(prompts)))
print('/....../....../....../....../....../')

# 输入拼接版本的prompt，输出供人工标注的prompt,并且修改母版:
def generate_label_prompt(prompt_ori):
    #从prompt中删掉assistant后面的提示语
    new_before_string='[gMASK]<sop><|system|>\nYou are a helpful assistant<|user|>\n'
    old_before_string='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    prompt=prompt_ori.replace(old_before_string,new_before_string)
    new_div_string='<|assistant|>\n'
    old_div_string='<|im_end|>\n<|im_start|>assistant\n'
    prompt=prompt.replace(old_div_string,new_div_string)
    div_string='<|assistant|>\n'
    #command = "\n\n你需要按照以下格式回复(记得带上“#######\n[回答]:”和“########”)：\n\n#######\n[回答]: ...\n########。另外遵循以下要求：\n（一）使用简洁明了的语言，回答结果尽可能的信息完整，对问题中的关键信息直接对应。\n（二）由于参考资料是从各种文章素材中摘取的片段，不必参考其样板格式和表述不当的地方。\n（三）确保连贯性和逻辑性流畅。以问题为导向，非直接回答问题的部分不要放在答案中。\n（四）每一句回答都要有明确的参考资料佐证，并在对应回答的句子结尾处添加引用证据编号，格式为[XXXXXXXX]，可以标记一个或者多个但同一句话不要重复添加相同编号，每个资料单独列出。例子：习近平总书记在2023年全国两会期间的行程安排紧凑，涉及多个重要活动和会议[11kk254v][2hdj4OHk]。"

    #label_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n问题:" + prompt_ori.partition('\n\n补充信息：')[0].partition('\n\n问题:')[-1] + '\n\n参考资料：\n' + prompt_ori.partition('\n提示思路：\n')[0].partition("相关问答：")[0].partition("\n\n\n结构化模版:\n")[0].partition('\n\n参考资料：\n')[-1] + command + '<|im_end|>\n<|im_start|>assistant\n'
    label_prompt = prompt.partition(div_string)[0] + div_string

    return label_prompt


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
            #model_name可以输入如下多个选择，glm_4_9B_chat
            payload = json.dumps({
                'text': prompt,
                "model_name":"glm_4_9B_chat",
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

#检查回答中是否有引证
def check_citation(response):
    """
    检查输入字符串中是否包含类似 [12345678] 这种格式的引证。

    参数：
    - response: 输入的字符串

    返回：
    - 如果存在这种格式的引证，返回 True；否则返回 False
    """
    # 定义正则表达式模式
    # 匹配形如 [12345678] 的模式，其中数字部分可以是任意长度的数字
    pattern = r'\[([a-zA-Z0-9]{8})\]'
    # 使用 re.search 检查是否存在匹配
    if re.search(pattern, response):
        return True
    else:
        return False
    
#删除字符串中子串第二次出现之后的内容
def remove_from_second_occurrence(text, substring):
    # 找到第一次出现的位置
    #[回答]:、[综述]:
    first_occurrence = text.find(substring)
    if first_occurrence == -1:
        return text  # 如果没有找到，直接返回原字符串
    
    # 从第一次出现的位置之后继续查找第二次出现的位置
    second_occurrence = text.find(substring, first_occurrence + len(substring))
    if second_occurrence == -1:
        return text  # 如果没有找到第二次出现，直接返回原字符串
    
    # 返回从开始到第二次出现之前的内容
    return text[:second_occurrence]


#处理每条原始的引证数据，每条原始的引证数据是一个字典dict，有category,label_prompt,output三个字段
#返回每条原始引证数据在一个字典中，包含原有的category,prompt,output，并且加上模型的回答。
def item_processing(dic:dict):
    prompt = dic['label_prompt']
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
    if check_citation(ans_raw):
        if ans_raw.find("[回答]:")!=-1:
            ans_raw=remove_from_second_occurrence(ans_raw, "[回答]:")
        elif ans_raw.find("[综述]:")!=-1:
            ans_raw=remove_from_second_occurrence(ans_raw, "[综述]:")
        else:
            ans_raw="No citation"
        dic_new={
            'category': category,
            'prompt': prompt,
            'response': ans_raw
        }
    else:
        dic_new={
            'category': category,
            'prompt': prompt,
            'response':"No citation"
        }
    return dic_new

filename_='/root/yuchen_llm_eval/data/用于人工标注数据/artifical_glm_4_9B_chat_0204——00.json'
with open(filename_, "a", encoding="utf-8") as f:
    f.write("[")
    f.close()

# 去重并创建映射
dic_mapping = {}
for item in data_citation_combo:
    output = item['output']
    if output not in dic_mapping:
        dic_mapping[output] = [item]
    else:
        dic_mapping[output].append(item)

print('处理过后篇章数目:', len(dic_mapping.keys()))

# 将所有output转换为列表
outputs = list(dic_mapping.keys())

# 初始化计数器和结果列表
non_no_citation_count = 0
selected_outputs = []  # 记录已选择的篇章

pbar = tqdm(total=100, desc="Processing citations", unit="citation")
# 处理直到得到100个非"No citation"的response
while non_no_citation_count < 100:
    # 从outputs中随机选择一个output，但排除已经标记为"No citation"的和已选择的
    available_outputs = [o for o in outputs if o not in selected_outputs]
    if not available_outputs:
        break  # 如果没有可用的outputs，则退出循环
    
    output_key = random.choice(available_outputs)
    selected_outputs.append(output_key)  # 标记为已选择
    ite = dic_mapping[output_key]
    prompt = ite[0]['prompt']
    label_prompt = generate_label_prompt(prompt)
    category = ite[0]['category']
    dic = {
        "category": category,
        "output": output_key,
        "label_prompt": label_prompt
    }
    
    # 调用item_processing函数
    result = item_processing(dic)
    
    # 检查返回的'response'是否是"No citation"
    if result.get('response') != "No citation":
        with open(filename_, 'a', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            f.write(',')
        non_no_citation_count += 1
        pbar.update(1)  # 更新进度条

pbar.close()   
with open(filename_, 'a', encoding='utf-8') as f:
    f.write(']')
print("结束*******************")

#parallel_processing(prompt_for_artificial_data[:10])
