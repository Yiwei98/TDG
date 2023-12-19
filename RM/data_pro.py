import os
import operator
import json
import re
from utils import delete_extra_zero,_strip_string
from MATH.dataset import read_jsonl, last_boxed_only_string, is_equiv, remove_boxed
import random
import statistics


def extract_math_answer(pred_str):
    if('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]

        if len(ans)==0 or (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a

    else:
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if(len(pred) >= 1):
            # print(pred_str)
            pred = pred[-1]
        else: pred = ''
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    pred=_strip_string(pred)
    if 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if len(ans) and (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a = _strip_string(a)
        pred=a
    return pred

def find_math_answer(s):

    assert('boxed' in s)
    # s = s.replace(",", "")
    ans = s.split('boxed')[-1]
    if(ans[0] == '{'):
        stack = 1
        a = ''
        for c in ans[1:]:
            if(c == '{'):
                stack += 1
                a += c
            elif(c == '}'):
                stack -= 1
                if(stack == 0): break
                a += c
            else:
                a += c
    else:
        a = ans.split('$')[0].strip()
    a=_strip_string(a)
    return a


cw_score,zq_score,cw_score_,zq_score_ = 0,0,0,0
all_num , all_right_num =0,0
count_ = 0
# all_data = []
# for i in range(7):
#     all_cate = ["counting_and_probability", "intermediate_algebra", "number_theory", "precalculus", "prealgebra",
#                 "geometry", "algebra"]
#     cate = all_cate[i]
#     with open("GPT4-MATH/PHP-8_train_gpt4_{}_8.json".format(cate), "r") as f:
#         l = json.load(f)
#         f.close()
#     all_data+=l
import json
all_cate = ["counting_and_probability","intermediate_algebra","number_theory","precalculus","prealgebra","geometry","algebra"]
all_data = {}
for tem in all_cate:
    with open("gpt4_train_nat/train_{}_n16_2.jsonl".format(tem),"r")as f:
        for line in f:
                data = json.loads(line)
                all_data[data["question"]] = data
        f.close()
    # with open("best-llama-train/train_{}_n16_2.jsonl".format(tem),"r")as f:
    #     for line in f:
    #             data = json.loads(line)
    #             if data["question"] in all_data.keys():
    #                 all_data[data["question"]]['generated_answer']+=data['generated_answer']
    #             else:
    #                 all_data[data["question"]] = data
    #     f.close()
for tem in all_cate:
    with open("gpt4_train_nat/train_{}_n16_1.jsonl".format(tem),"r")as f:
         for line in f:
                 data = json.loads(line)
                 all_data[data["question"]] = data
         f.close()



def find_string_positions(text, pattern):
    positions = []
    for match in re.finditer(pattern, text):
        positions.append((match.start(), match.end()))
    return positions

pr_data = []
right_num=0
all_num=0
overall_num,over_all_right=0,0
for key in all_data.keys():
    tem = all_data[key]
    cur = {"question":tem['question'],"level1":[re.sub("\nSolution:\n","",tem['answer'])],"level2":[]}
    #cur = {"question": tem['question'], "level1": [], "level2": []}
    #output = remove_boxed(last_boxed_only_string(sample["generated_answer"]))
    answer = remove_boxed(last_boxed_only_string(tem["answer"]))
    #equiv = is_equiv(output, label)
    #answer = find_math_answer(tem['answer'])
    all_num += 1
    flag=1
    for t in range(len(tem['generated_answer'])):
        predict = remove_boxed(last_boxed_only_string(tem['generated_answer'][t]))
        #tem['generated_answer'][t] = re.sub("Solution:\n","",tem['generated_answer'][t])
        #tem['generated_answer'][t] = tem['generated_answer'][t].strip(" ")
        #tem['generated_answer'][t] = tem['generated_answer'][t].strip("\n")
        #tem['generated_answer'][t] = tem['generated_answer'][t].strip(" ")
        #tem['generated_answer'][t] = tem['generated_answer'][t].strip("\n")
        cur_s = tem['generated_answer'][t]
        overall_num+=1
        if is_equiv(answer,predict):
            over_all_right+=1
            flag=1
            cur['level1'].append(tem['generated_answer'][t])
        else:
            cur['level2'].append(tem['generated_answer'][t])
    right_num+=flag
    if flag:
        pr_data.append(cur)

random.shuffle(pr_data)
with open("gpt4_for_RM_train.json","w")as f:
    json.dump(pr_data[:-100],f)
    f.close()
with open("gpt4_for_RM_valid.json","w")as f:
    json.dump(pr_data[-100:],f)
    f.close()
print("cover:",right_num/all_num)
print("Acc:",over_all_right/overall_num)

