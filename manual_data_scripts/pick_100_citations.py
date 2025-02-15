import re
import json
#判断一个字符串的末尾是否包含形如 [abcd1234] 的引用ID
def has_reference_id_at_end(text):
    # 正则表达式：匹配末尾的 [abcd1234] 格式
    pattern = r"\[[a-zA-Z0-9]{8}\]$"
    return bool(re.search(pattern, text))

fname='/data/yuchen_llm_eval/data/用于人工标注数据/artifical_glm_4_9B_chat_0204——00.json'
with open(fname, 'r', encoding='utf-8') as file:
    content=file.read()
if content.endswith(',]'):
    content=content[:-2]+']'
artificial_data = json.loads(content)
print('加载的文件名:',fname)
#去除输出中奇怪的句子
def remove_irregular_statements(response,sentences):
    pattern = r'#+\s*\n+\s*参考资料：\s*'
    pattern2 = r'\n+\s*参考资料：\s*'
    pattern3 = r'\n+\s*\[参考资料：\s*'
    pattern4 = r'#+\s*参考资料\s*'

    output = response
        
    # 查找所有匹配的子串
    lis_id = re.findall(pattern, output)
    lis_id2 = re.findall(pattern2, output)
    lis_id3 = re.findall(pattern3, output)
    lis_id4 = re.findall(pattern4, output)
        
    # 将所有匹配到的子串合并到一个列表中，前提是每个列表不为空
    target_strs = []
    if lis_id:
        target_strs.extend(lis_id)
    if lis_id2:
        target_strs.extend(lis_id2)
    if lis_id3:
        target_strs.extend(lis_id3)
    if lis_id4:
        target_strs.extend(lis_id4)
           
    # 移除包含目标子串的句子
    if target_strs:
        sentences_to_remove = set()
        for i, sentence in enumerate(sentences):
            for target_str in target_strs:
                if target_str in sentence:
                    sentences_to_remove.add(i)
                    break
        # 根据索引移除句子
        filtered_sentences = [sentence for i, sentence in enumerate(sentences) if i not in sentences_to_remove]  
    else:
         filtered_sentences= sentences 
    return  filtered_sentences
citation_num_bar=100
#total_citation_cnt=0

pattern_2=r'\[[a-zA-Z0-9]{8}\]'
pattern = r'\[[a-zA-Z0-9]{8}\].+\[[a-zA-Z0-9]{8}\]'
#返回data数据集里面大约前100个citation所对应的句子，prompt和类别。句子是key，prompt和类别在句子key
#对应的value里面
#data数据集是一个list of dict，里面有prompt，response
#category三栏
def raw_label_data(data):
    final_sentences={}#最终返回的数据
    total_citation_cnt=0#记录citation数目
    for item in data:
        category_here=item.get('category', '')
        prompt_here=item.get('prompt', '')
        response = item.get('response', '')
        #print(response)
        response = re.sub(r'#######\n\[回答\]: # |\n########|#############\n\[综述\]:|#######\n\[回答\]:|\[综述\]:|\[回答\]: ', '', response)
        sentences = [s.strip() for s in response.split('。') if s.strip()]
        filtered_sentences=remove_irregular_statements(response,sentences)
        lis_id=re.findall(pattern_2,filtered_sentences[-1])
        if lis_id:
            filtered_sentences=filtered_sentences[:-1]
        for index in range(len(filtered_sentences)):
            sentence=filtered_sentences[index]
            if sentence in final_sentences.keys():
                continue
            sentence_for_test= re.sub(r'\[.*?\]|\[.*', '', sentence)
            if len(sentence_for_test)<30 and index >0:
                sentence=filtered_sentences[index-1]+'。'+sentence
                print('add sentence')
            matches = re.findall(pattern, sentence)
            if matches:
                continue
            if has_reference_id_at_end(sentence):
                num_citations=len(re.findall(pattern_2,sentence))
                total_citation_cnt=total_citation_cnt+num_citations
                
                final_sentences[sentence]={
                    "prompt":prompt_here,
                    "category":category_here
                }
            if total_citation_cnt>=100:
                print(total_citation_cnt)
                return final_sentences
    return final_sentences
    

final_sentences_dic = raw_label_data(artificial_data)
#print(final_sentences_dic.keys())
cnt_here=0
for ite in final_sentences_dic.keys():
    print("\n",ite)
    cnt_here=cnt_here+len(re.findall(pattern_2,ite))
print('复核的引证数目:',cnt_here)

#处理最后的引证
pattern_sep = re.compile(r'(\[[a-zA-Z0-9]{8}\])(.*?)(?=\[[a-zA-Z0-9]{8}\]|\Z)', re.DOTALL) # 引证id和对应引证资料

processed_data = []
for key in final_sentences_dic.keys():
    data=final_sentences_dic[key]
    category=data['category']
    prompt=data['prompt']
    answer=key
    #if "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你是一个中文大语言模型。" in prompt:
    if "你是一个中文大语言模型。你在做一个百科问答任务，" in prompt:
        chunk_ref=prompt.partition("\n\n参考资料：\n")[-1].partition("相关问答：")[0].partition("提示思路：")[0].partition("\n\n\n结构化模版：\n")[0]
    else:
        chunk_ref=prompt.partition("\n\n参考资料：\n")[-1].partition("注意遵守以下事项：\n1. 你需要在回答结果中插入引用证据的来源编号，格式为[编号]")[0]
    # print(prompt)    
    # contents=re.split(pattern_2,chunk_ref)[1:]
    # print(len(contents))
    # print(contents)
    ids=re.findall(pattern_2,chunk_ref)
    #print('id数量:',len(ids))
    
    all_refs = pattern_sep.findall(chunk_ref)
    #print('参考资料数量:',len(all_refs))
    #print(all_refs[0])
    if len(ids)!=len(all_refs):
        print('error, number mismatch')
    # 保存到字典里
    refs = {}
    for ref_id, content in all_refs:
        refs[ref_id] = content
    #print(refs)
    # 给回答中的引证依次标红
    matches=re.finditer(pattern_2,answer)
    cnt=0
    for match in matches:
        #print(match.group(),'   ',match.start(),'   ',match.end())
        id=match.group()
        start_ind=match.start()
        end_ind=match.end()
        before_citation_id=answer[:start_ind]
        new_id="<span style='color:red'>{}</span>".format(id)
        answer_for_label=before_citation_id+new_id # 标红引证及之前的所有文本
        # print(answer_for_label)

        # 在prompt中寻找id对应的参考资料
        if id in refs:
            ref = refs[id]
            id_new="<span style='color:red'>{}</span>".format(id)
            ref_for_label=id_new+ref
        else:
            ref_for_label="无对应参考资料序号"
            print(f"{id}参考资料不存在")
            #print(chunk_ref)
        cnt=cnt+1

        ref_html = ref_for_label.replace("\n", "<br/>")
        answer_html = answer_for_label.replace("\n", "<br/>")
        #label = f"参考资料：<br/>{ref_html}<br/><br/>{answer_html}"

        processed_data.append({
            'category': category,
            'reference': ref_html,
            'answer': answer_html
        })

print('单句单引证匹配数据例子',len(processed_data))

for item in processed_data[-5:]:
    print(item['reference'])
    print(item['answer'])
file_name='/data/yuchen_llm_eval/data/用于人工标注数据/label_glm4_9b_chat_100_citation_0208_new.json'
with open(file_name, "w", encoding="utf-8") as file:
    json.dump(processed_data, file, indent=4, ensure_ascii=False)