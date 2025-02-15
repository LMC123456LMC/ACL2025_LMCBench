import json
import re
from tqdm import tqdm
import requests
import time
import random

import concurrent.futures
import threading


#加载参考数据
# fname='/root/yuchen_llm_eval/data/qwen2.5_14B_1226_stats2.json'
fname='/root/yuchen_llm_eval/data/3000_sample.json'
with open(fname, 'r', encoding='utf-8') as file:
    data_citation_combo = json.load(file)


prompts=[item['prompt'] for item in data_citation_combo]
print('prompts数量',len(prompts))
print('独特prompts数量',len(set(prompts)))

#处理从开头到带匹配引证的句子的答案，修改成xml格式。需要和别的格式保持一致。
def process_post_answer(answer_here):
    # 定义正则表达式匹配引证ID
    pattern = r'(\[[A-Za-z0-9]{8}\])' #用于切分答案，匹配切分出来的引证ID
    pattern_2 = r'\[([A-Za-z0-9]{8})\]' #用于按照新格式替换切分出来的引证ID
    # 使用 re.split 分割字符串，保留引证ID
    split_text = re.split(pattern, answer_here[:-1])

    # 过滤掉空字符串
    split_text = [part for part in split_text if part.strip()]

    if ''.join(split_text) != answer_here[:-1]:
        print('字符串比对测试error！')
        print(answer_here)

    #xml格式修改
    #xml格式修改第一句话
    replaced_texts=[]

    txt=split_text[0]
    replaced_text_first = """
    <cited_answer>
        <answer>{}</answer>
        <citations>
""".format(txt)
    replaced_texts.append(replaced_text_first)

    cnt_ids=0
    for splited_str in split_text[1:]:
        #替换后续切分出的引证id，换成xml格式
        if re.search(pattern, splited_str):
            replaced_text = re.sub(pattern_2,r'            <source_id>\1</source_id>\n',splited_str)
            replaced_texts.append(replaced_text)
            cnt_ids=cnt_ids+1
        #替换后续切分出的句子，换成xml格式
        else:
            replaced_text="        </citations>\n        <answer>"+splited_str+"</answer>\n        <citations>\n"
            replaced_texts.append(replaced_text)

    new_post_answer=''.join(replaced_texts)+"            <source_id>"
    ori_ids = len(re.findall(pattern,answer_here))
    if cnt_ids!=ori_ids:
        print('id number mismatch error!')
    return new_post_answer

#输入拼接版本的prompt，输出xml格式prompt:
def generate_xml_prompt(prompt_ori):
    #提取参考资料部分，引证id和资料一一对应，存储起来
    if "你是一个中文大语言模型。你在做一个百科问答任务，" in prompt_ori:
        raw_ref_chunk=prompt_ori.partition("\n\n参考资料：\n")[-1].partition("相关问答：")[0].partition("提示思路：")[0].partition("\n\n\n结构化模版：\n")[0]
    else:
        raw_ref_chunk = prompt_ori.partition("\n\n参考资料：\n")[-1].partition("注意遵守以下事项：\n1. 你需要在回答结果中插入引用证据的来源编号，格式为[编号]")[0]
    pattern_match=re.compile(r'\[([a-zA-Z0-9]{8})\]')#pattern找出所有引证id
    pattern_sep = re.compile(r'(\[([a-zA-Z0-9]{8})\])(.*?)(?=\[([a-zA-Z0-9]{8})\]|\Z)', re.DOTALL)#pattern找出所有引证id和对应引证资料
    all_refs = pattern_sep.findall(raw_ref_chunk)#所有引证id
    all_marks=pattern_match.findall(raw_ref_chunk)#所有引证id和对应参考资料
    if len(all_marks)!=len(all_refs):
        print('参考资料数量匹配错误')
    #参考资料转化为xml格式
    new_ref_chunk='<references>\n'
    for ite in all_refs:
        ref_id=ite[1]
        ref_content=ite[2][1:].rstrip('\n')
        formated_single_cite=f"    <reference><source_id>{ref_id}</source_id><content>{ref_content}</content></reference>\n"
        new_ref_chunk=new_ref_chunk+formated_single_cite
    new_ref_chunk=new_ref_chunk+'</references>\n\n'
    #提取问题，生成简化版本的参考资料之前的insturction
    if "你是一个中文大语言模型。你在做一个百科问答任务，" in prompt_ori:
        pre_ref=prompt_ori.partition(raw_ref_chunk)[0]
        start_index = pre_ref.find("问题: ") + len("问题: ")
        end_index = pre_ref.find("\n\n补充信息：")
        question=pre_ref[start_index:end_index]
        pre_ref_new='你在做一个百科问答任务，请基于参考资料来回答问题。\n\n'+"问题: "+question+"\n\n参考资料：\n\n"
    else:
        pre_ref=prompt_ori.partition(raw_ref_chunk)[0]
        start_index = pre_ref.find("\n你需要撰写的章节的分标题为：") + len("\n你需要撰写的章节的分标题为：")
        end_index = pre_ref.find("\n\n我将给你一些参考资料")
        question=pre_ref[start_index:end_index]
        pre_ref_new='你在创作一篇综述。请基于参考资料和给定的标题来创作这篇综述。\n\n'+"标题: "+question+"\n\n参考资料：\n\n"
    #生成参考资料之后的instruction
    div_string='<|im_end|>\n<|im_start|>assistant\n'
    #匹配包括带引证句子的答案或者综述。从答案或者综述开头到待匹配的句子
    matched_answer=prompt_ori.partition(div_string)[-1] #不用匹配后去除左括号，下面函数里面做了
    #生成xml格式answer
    xml_answer=process_post_answer(answer_here=matched_answer)

    post_ref="""请按照如下格式输出:
    <cited_answer>
        <answer></answer>
        <citations>
            <source_id></source_id>
            <source_id></source_id>
            ...
        </citations>
    </cited_answer>\n\n"""+"""回答示例:
    <cited_answer>
        <answer>习近平总书记在2023年全国两会期间的行程安排紧凑，涉及多个重要活动和会议</answer>
        <citations>
            <source_id>11kk254v</source_id>
            <source_id>2hdj4OHk</source_id>
        </citations>
    </cited_answer>\n\n"""+div_string+xml_answer
    #拼接最后xml格式prompt
    xml_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+pre_ref_new+new_ref_chunk+post_ref
    return xml_prompt
print("---------","\n\n\n\n\n\n")

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
            #model_name可以输入如下多个选择，Qwen2.5_7B，Qwen2.5_14B，Qwen2.5_32B，Qwen2.5_72B，Qwen2_7B, Qwen1.5_7B
            payload = json.dumps({
                'text': prompt,
                "model_name":"Qwen2.5_72B",
                "temperature":0,
                "max_tokens":50
                })
            # 发送请求并获取响应
            response = requests.request("POST", url, headers=headers, data=payload,timeout=60)
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
    category = dic['category']
    output= dic['output']
    xml_prompt = generate_xml_prompt(prompt)
    try:
        ans_raw = citation_generation(xml_prompt)['response']
    except (TypeError, KeyError) as e:
        return {
            "category": category,
            "output": output,
            "prompt": xml_prompt,
            "response": "{}: {}".format(type(e).__name__, e)
        }
    #end_index = ans_raw.find(']')
    end_index = ans_raw.find('</source_id>')
    ans_final = ans_raw[:end_index]
    dic_new={
        'category': category,
        'output': output,
        'prompt': xml_prompt,
        'response': ans_final
    }
    return dic_new

lock = threading.Lock()

# parallel_processing(data_citation_combo)
f_res="/root/yuchen_llm_eval/data/新的验证结果/xml_results2.5_72B_0205_3000_sample_new_01.json"
with open(f_res, 'r', encoding='utf-8') as file:
    content=file.read()
if content.endswith(',]'):
    content=content[:-2]+']'
if content.endswith(','):
    print('end with comma')
    content=content[:-1]+']'
qwen25_res = json.loads(content)
print('千问xml格式结果数据长度',len(qwen25_res))
output_file = "/root/yuchen_llm_eval/data/新的验证结果/xml_results2.5_72B_0205_3000_sample_new_02.json"
def process_list_and_write_to_file(data_list: list, output_file: str):
    # 创建一个新列表用于存储最终结果
    num = 0 
    handling_error_num=0
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("[")
        f.close()
    for item in tqdm(data_list, desc="Processing items"):
        if item.get("response") != "RunTimeError Message\n\nFailed to get a response from the server":
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
                if last_result.get("response") != "RunTimeError Message\n\nFailed to get a response from the server":
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


process_list_and_write_to_file(qwen25_res, output_file)