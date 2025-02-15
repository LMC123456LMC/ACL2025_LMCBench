# 统计

import json
import re
from tqdm import tqdm
import sys

model = "gpt_4o_mini"
f_res='/data/yuchen_llm_eval/data/新的验证结果/full_res_gpt_4o_mini_0213_01.json'
log_file_path = "/data/yuchen_llm_eval/data/新的验证结果/full_res_gpt_4o_mini_0213_01_log.txt"
sys.stdout = open(log_file_path, 'w', encoding='utf-8')

model_div_string_dict={
    "qwen2.5_72B": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen2.5_32B": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen2.5_14B": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen2.5_7B":"<|im_end|>\n<|im_start|>assistant\n",
    "qwen2_72B": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen2_57B": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen2_7B":"<|im_end|>\n<|im_start|>assistant\n",
    "baichuan":"<user_end><im_assistant>",
    "doubao": "形式如'[abcd1234]'：\n", # doubao_oldprompt
    "moonshot": "<user_end><im_assistant>",
    "glm_4_plus":"<user_end><im_assistant>",
    "deepseek_v3":"<user_end><im_assistant>",
    "glm_4_9b_chat":"<|assistant|>\n",
    "llama3.3_70B": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "gpt_4_turbo": "<user_end><im_assistant>",
    "gpt_4o": "<user_end><im_assistant>",
    "gpt_4o_mini":"<user_end><im_assistant>"
}

with open(f_res, 'r', encoding='utf-8') as file:
    content=file.read()
if content.endswith(',]'):
    content=content[:-2]+']'
if content.endswith(','):
    print('end with comma')
    content=content[:-1]+']'
data_res = json.loads(content)
print(model, '结果数目:',len(data_res))
prompts = []
for it in data_res:
    prompts.append(it['prompt'])
print(model, '独特prompt数目:',len(set(prompts)))
print('-----------------')
print('不规则输出：')

# 获取正确答案
def get_right_answer(res:dict, raw_answer: str):

    # :res: 模型输出结果
    # :param raw_answer: 完整原回答
    # :return: 此位置上的正确答案候选

    #div_string="<user_end><im_assistant>" # 闭源模型
    #div_string="形式如'[abcd1234]'：\n" # 豆包——oldprompt
    #div_string = "<|im_end|>\n<|im_start|>assistant\n"  # 千问
    #div_string = "\n\nAssistant:"  # deepseek
    #div_string = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # Llama
    #div_string="<|assistant|>\n" # glm
    div_string = model_div_string_dict[model]
    
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
for res in data_res:
    cnt+=1
    dic={}
    category = res.get('category')
    output = res.get('output')
    prompt = res.get('prompt')
    response = res.get('response')

    if "Error:" in response or "RunTimeError" in response or "当前分组上游负载已饱和" in response:
        cnt-=1
        continue
    
    if ']' in response:
        index = response.find(']')
        response = response[max(0, index - 8):index]

    response='['+response+']'
    correct_answer=get_right_answer(res=res,raw_answer=output)
    
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
print('引证题目数目:', cnt) # 处理的条目数量
print('不规则数据数目:', irrg) # 有问题的条目数量
print(model, '模型引证正确数目:', corr)  # 输出模型正确的引证数目
print(model, '模型引证正确率:', round(corr / cnt * 100, 2)) 

new_outputs=[it['output'] for it in new_res]
print('去除不规则数据之后的输出数量:',len(set(new_outputs)))

sys.stdout.close()
