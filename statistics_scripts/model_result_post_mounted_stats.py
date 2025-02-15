# 新闻数据统计

import json
import re
from tqdm import tqdm
import sys

model = "Qwen2.5_72B"
f_res='/data/yuchen_llm_eval/data/新的验证结果/post_mounted_qwen2.5_72B_0205_3000_sample_02.jsonl'
#f_res='/root/yuchen_llm_eval/data/参考资料是否乱序/Llama3.3-70B_1000_disorder_0206__00.json'
#f_stats='/root/yuchen_llm_eval/data/try_result/gpt_try_stats_0125_01.json'
log_file_path = "/data/yuchen_llm_eval/data/新的验证结果/post_mounted_results_qwen2.5_72B_3000_log_0207.txt"
# model = "千问2-72B"
# f_res='/root/yuchen_llm_eval/data/新的验证结果/new_50%_combined_data_results2_72B_0115_01.json'
# f_stats='/root/yuchen_llm_eval/data/新的验证结果/new_50%_combined_data_results2_72B_0115_01_stats1.json'
# f_group='/root/yuchen_llm_eval/data/新的验证结果/new_50%_combined_data_results2.5_7B_0110_01_stats2_group.json'
# log_file_path = "/root/yuchen_llm_eval/data/新的验证结果/output_log_0115_qwen2_72B.txt"
sys.stdout = open(log_file_path, 'w', encoding='utf-8')

print("result file name:",f_res)

qwen25_res=[]
# 打开jsonl文件并逐行读取
with open(f_res, 'r', encoding='utf-8') as file:
    for line in file:
        # 将每行解析为字典并添加到列表中
        dict_data = json.loads(line)
        qwen25_res.append(dict_data)

prompts = []
for it in qwen25_res:
    prompts.append(it['prompt'])
print(model, '独特prompt数目:',len(set(prompts)))

#加载参考文件
ref_path = '/data/yuchen_llm_eval/data/3000_sample.json'
with open(ref_path, 'r', encoding='utf-8') as file:
    ref_data = json.load(file)
#检查模型结果文件和参考文件是否一致
cnt_match=0
for ind in range(0,len(ref_data)):
    output1 = qwen25_res[ind]['output']
    output2 = ref_data[ind]['output']
    if output1 != output2:
        print('error, output mismatch')
    else:
        cnt_match=cnt_match+1
if cnt_match==len(ref_data):
    print('模型结果匹配原始数据')
else:
    print('模型结果不匹配原始数据，有错误',ref_data-cnt_match,'处')
print('-----------------')
print('不规则输出：')

length_s = []
for ite in qwen25_res:
    response = ite.get('response')
    length = len(response)
    length_s.append(length)

from collections import Counter
print('长度统计',Counter(length_s))
# 获取正确答案
def get_right_answer(res:dict, raw_answer: str):

    # :res: 模型输出结果
    # :param raw_answer: 完整原回答
    # :return: 此位置上的正确答案候选
    # if model == "gpt_4o":
    #     div_string = "\n***********\n请只关注'***********'之前内容的参考资料部分。输出下面这段话中的最后一句话挂载的引证，不输出这段话，形式如'[abcd1234]'：\n"
    # else:
    #     div_string = "<|im_end|>\n<|im_start|>assistant\n" 

    #div_string="<user_end><im_assistant>" # 闭源模型
    div_string = "<|im_end|>\n<|im_start|>assistant\n"  # 千问
    #div_string = "\n\nAssistant:"  # deepseek
    # div_string = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # Llama
    #div_string="<|assistant|>\n" # glm
    
    prefix = res.get('prompt').partition(div_string)[-1][:-1] # 去掉"["
    answer_after_prefix = raw_answer.partition(prefix)[-1]
    answer_candidates = []
    if prefix not in raw_answer:
        print('错误匹配，原文无本句')
        print("句子: ",prefix)
        return []
    while True:
        #匹配文本anser_after_prefix中的第一个形如[abcd1234]的引证
        candidate = re.match(r"\[([a-zA-Z0-9]{8})\]", answer_after_prefix)
        #如果有，就去掉这个引证，匹配下一个引证。
        if candidate:
            answer_candidates.append(candidate.group())
            answer_after_prefix = answer_after_prefix.partition(candidate.group())[-1]
        else:
            break
    return answer_candidates

cnt=0
irrg=0
corr=0
new_res = []
for index in range(0,len(ref_data)):
    ref_item = ref_data[index]
    res = qwen25_res[index]
    cnt+=1
    dic={}
    category = res.get('category')
    output = ref_item.get('output') #参考文件，原始数据能够告诉我们标准答案是什么
    prompt = ref_item.get('prompt') #参考文件，原始数据能够告诉我们标准答案是什么
    response = res.get('response')

    if "Error:" in response or "RunTimeError" in response or "当前分组上游负载已饱和" in response:
        cnt-=1
        continue
    
    # if len(response)==10:
    #     pattern = r"'[a-zA-Z0-9]{8}'"
    #     #response = response[:index]
    #     matches = re.findall(pattern, response)
    #     if matches:
    #         response = matches[0][1:-1]
    #     else:
    #         response = 'no suitable result'
    pattern = r'(?<![a-zA-Z0-9])[a-zA-Z0-9]{8}(?![a-zA-Z0-9])'
    matches = re.findall(pattern, response)
    if matches:
        response = matches[0]
    else:
        response = 'no suitable result'

    response='['+response+']'
    
    # correct_answer = output.partition(single_sentence)[-1].partition("]")[0] 
    # correct_answer = res.get('correct answer')
    correct_answer=get_right_answer(res=ref_item,raw_answer=output) #原始数据告诉我们标准答案
    
    if not correct_answer:
        print('error! no right choice!')
        print('no right choice///',response)
        cnt-=1
        continue
    if 'Cite-' in output or 'Cite-' in prompt:
        print('irregular input!//')
        cnt-=1
        continue
    if response is None:
        print('err in final response')
    
    else:
        if len(response) !=10:
            irrg+=1
            print("response: ", response)
            print("     correct answer: ", correct_answer)
            #continue # 删掉错误数据
        dic['category']=category
        dic['output']=output
        dic['prompt']=prompt
        dic['response']=response
        dic['correct answer']=correct_answer

        if response in correct_answer:
            corr+=1
            dic['correctness']=True
        else:
            dic['correctness']=False

        new_res.append(dic)

print('-----------------')
print('结果文件名称:',f_res)
print('引证题目数目:', cnt) # 处理的条目数量
print('不规则数据数目:', irrg) # 有问题的条目数量
print(model, '模型引证正确数目:', corr)  # 输出模型正确的引证数目
print(model, '模型引证正确率:', round(corr / cnt * 100, 2)) 

# with open(f_stats, 'w', encoding='utf-8') as f:
#     json.dump(new_res, f, indent=4, ensure_ascii=False)  # 将处理后的结果写入文件

new_outputs=[it['output'] for it in new_res]
print('去除不规则数据之后的输出数量:',len(set(new_outputs)))




# # 将结果保存到文件
# with open(f_group, 'w', encoding='utf-8') as f:
#     json.dump(new_stats, f, indent=4, ensure_ascii=False)

sys.stdout.close()
