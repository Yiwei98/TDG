import json
from utils_ import _strip_string
import re
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

import operator
# data_model = "GPT4"
subset = "train"

with open("gpt4_for_RM_{}.json".format(subset), "r") as f:
    contents = json.load(f)
    f.close()

for i in range(len(contents)):
    tem = contents[i]
    ans = extract_math_answer(tem['level1'][0])
    pre_dict = {}
    cur_list = []
    for t in tem['level2']:
        pre = extract_math_answer(t)
        cur_list.append(pre)
        if pre in pre_dict.keys():
            pre_dict[pre] += 1
        else:
            pre_dict[pre] = 1
    pfh = 0
    for t in pre_dict.keys():
        pfh+=pre_dict[t]*pre_dict[t]
    g = [pre_dict[t]*len(tem['level2'])/pfh for t in cur_list]
    tem["weight"] = g
    contents[i] = tem

with open("best-llama-data_for_RM_{}-weight.json".format(subset), "w") as f:
    json.dump(contents,f)
    f.close()

