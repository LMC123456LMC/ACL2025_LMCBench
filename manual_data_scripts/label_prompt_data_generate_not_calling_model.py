import json
import re
from tqdm import tqdm
import requests
import time
import random
from collections import defaultdict
random.seed(30)

#加载参考数据
fname='/root/yuchen_llm_eval/data/新的验证结果/new_50%_combined_data_results2.5_7B_0110_01_stats2_cut.json'
with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)
prompts=[item['prompt'] for item in data_citation_combo]
print('prompts数量',len(prompts))
print('独特prompts数量',len(set(prompts)))
print('/....../....../....../....../....../')

# 输入拼接版本的prompt，输出供人工标注的prompt:
def generate_label_prompt(prompt_ori):
    #从prompt中删掉assistant后面的提示语
    div_string='<|im_end|>\n<|im_start|>assistant\n'
    #command = "\n\n你需要按照以下格式回复(记得带上“#######\n[回答]:”和“########”)：\n\n#######\n[回答]: ...\n########。另外遵循以下要求：\n（一）使用简洁明了的语言，回答结果尽可能的信息完整，对问题中的关键信息直接对应。\n（二）由于参考资料是从各种文章素材中摘取的片段，不必参考其样板格式和表述不当的地方。\n（三）确保连贯性和逻辑性流畅。以问题为导向，非直接回答问题的部分不要放在答案中。\n（四）每一句回答都要有明确的参考资料佐证，并在对应回答的句子结尾处添加引用证据编号，格式为[XXXXXXXX]，可以标记一个或者多个但同一句话不要重复添加相同编号，每个资料单独列出。例子：习近平总书记在2023年全国两会期间的行程安排紧凑，涉及多个重要活动和会议[11kk254v][2hdj4OHk]。"

    #label_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n问题:" + prompt_ori.partition('\n\n补充信息：')[0].partition('\n\n问题:')[-1] + '\n\n参考资料：\n' + prompt_ori.partition('\n提示思路：\n')[0].partition("相关问答：")[0].partition("\n\n\n结构化模版:\n")[0].partition('\n\n参考资料：\n')[-1] + command + '<|im_end|>\n<|im_start|>assistant\n'
    label_prompt = prompt_ori.partition(div_string)[0] + div_string

    return label_prompt

# 去重, 随机选取100篇output
dic_mapping={}
for item in data_citation_combo:
    output=item['output']
    if output not in dic_mapping.keys():
        dic_mapping[output]=[item]
    else:
        dic_mapping[output].append(item)

print('处理过后篇章数目:',len(dic_mapping.keys()))
# 随机选取100篇文章
outputs=list(dic_mapping.keys())
sampled_outputs = random.sample(outputs,100)
print('独特选中篇章数:',len(set(sampled_outputs)))
#生成100篇unique篇章
lis_raw_label_data=[]#原始用于标注数据
for output_key in sampled_outputs:
    ite = dic_mapping[output_key]
    prompt=ite[0]['prompt']
    label_prompt=generate_label_prompt(prompt)
    category=ite[0]['category']
    dic = {
        "category":category,
        "output":output_key,
        "label_prompt":label_prompt
    }
    lis_raw_label_data.append(dic)

fname='/root/yuchen_llm_eval/data/用于人工标注数据/label_qwen_data_raw_0115.json'
with open(fname, 'w', encoding='utf-8') as file:
    json.dump(lis_raw_label_data,file,indent=4,ensure_ascii=False)
    
# 注释掉模型运行部分，准备产生人工标注数据prompt的代码和运行这些prompt
# 产生人工数据的代码要区分开来。
# def citation_generation(prompt):
#     url = "http://101.132.252.74:5004/proxy_generate"

#     headers = {
#             'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
#             'Content-Type': 'application/json',
#             'Accept': '*/*',
#             'Connection': 'keep-alive'
#         }
#     max_retries=3
#     for attempt in range(max_retries):
#         try:
#             #model_name可以输入如下四个选择，Qwen2.5_7B，Qwen2.5_14B，Qwen2.5_32B，Qwen2.5_72B。
#             payload = json.dumps({
#                 'text': prompt,
#                 "model_name":"Qwen2.5_72B",
#                 "temperature":0,
#                 "max_tokens":2048
#                 })
#             # 发送请求并获取响应
#             response = requests.request("POST", url, headers=headers, data=payload)
#             #print(response.text)
#             # 检查响应状态
#             response.raise_for_status()  # 如果响应错误，抛出异常

#             #response_filtered=(response.text).partition(prompt)[-1] #去除prompt
#             response_to_dic = json.loads(response.text)
#             response_filtered=(response_to_dic['text'][0]).partition(prompt)[-1]
#             #print(prompt[-20:])
#             #print(response_filtered)
#             result_dic={
#                 "prompt": prompt,
#                 "response": response_filtered
#             }
#             return result_dic
#         except requests.exceptions.RequestException as e:
#             print(f"请求失败: {e}, 正在重试... ({attempt + 1}/{max_retries})")
#             time.sleep(1)  # 等待1秒再重试
#         except Exception as e:
#             print(f"处理响应时发生错误: {e}")
#             return {"prompt": prompt,"response":'error message'}
# cnt_irr=0

# filename_ = '/root/yuchen_llm_eval/data/label_results2.5_72B_0103_trial_new_07.json'
# res = []

# for ind, item in enumerate(tqdm(unique_data_citation_combo)):
#     prompt = item['prompt']
#     category = item['category']
#     label_prompt = generate_label_prompt(prompt)
#     # print(label_prompt[-150:])
#     ans = citation_generation(label_prompt)['response']

#     dic = {
#         'category': category,
#         'prompt': label_prompt,
#         'answer': ans
#     }
#     res.append(dic)

# # Write to file after the loop
# with open(filename_, 'w', encoding='utf-8') as f:
#     json.dump(res, f, indent=4, ensure_ascii=False)
