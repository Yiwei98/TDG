import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from MATH.dataset import read_jsonl, last_boxed_only_string, is_equiv, remove_boxed
import torch
from models_ft5 import BertForCL
from transformers import AutoTokenizer
from utils_ import _strip_string
import re
from tqdm import tqdm
import numpy
def toke(Q,S,tokenizer,max_length = 512):
    new_sent_features = {"input_ids": [], "attention_mask": [], "special_tokens_mask": []}
    sent_features = tokenizer(
        [Q,S],
        max_length=max_length,
        truncation=True,
        padding=False,)
    begin_word = sent_features["input_ids"][0][0]
    begin_at_mask = sent_features["attention_mask"][0][0]
    begin_sp_mask = 0
    tem0_true = [begin_word]
    tem2_true = [begin_at_mask]
    tem3_true = [begin_sp_mask]
    tem0_true += sent_features["input_ids"][0][1:]+sent_features["input_ids"][1][1:]
    tem2_true += sent_features["attention_mask"][0][1:]+sent_features["attention_mask"][1][1:]
    tem3_true += [0 for i in range(len(sent_features["attention_mask"][0][1:]))]+[1 for i in range(len(sent_features["attention_mask"][1][1:]))]
    tem3_true[-1] = 0
    if len(tem0_true) > max_length: return None
    else:
        new_sent_features["input_ids"].append(tem0_true)
        new_sent_features["attention_mask"].append(tem2_true)
        new_sent_features["special_tokens_mask"].append(tem3_true)
        batch = tokenizer.pad(
                new_sent_features,
                padding='longest',
                max_length=max_length,
                return_tensors="pt",
            )
        batch.data["special_tokens_mask"] = batch.data["special_tokens_mask"] * batch.data["attention_mask"]
        return batch
def cal_score(model,tokenizer,Q,S):
    batch = toke(Q,S,tokenizer)
    if batch is not None:
        input_ids = batch["input_ids"].to(device)
        special_tokens_mask = batch["special_tokens_mask"].to(device)
        inputs = {
            "input_ids": input_ids,
            "special_tokens_mask": special_tokens_mask,
            "labels": None,
        }
        score = model(**inputs)
        return score
    else: return None
devicenum = 6
hs=2048
pre_train_model_name = "flan-t5-xl"
device = torch.device("cuda" + ":" + str(devicenum) if torch.cuda.is_available() else "cpu")
model = BertForCL(hs=hs,model_name = pre_train_model_name)
model = model.to(device)
ckpt_name1 = "gpt4_full_ckpt"
ckpt_name2 = "model_4"
model.load_checkpoint("MATH/{}/{}.pth".format(ckpt_name1,ckpt_name2), device)
model.is_eval = True
model.eval()
tokenizer = AutoTokenizer.from_pretrained(pre_train_model_name)
all_cate = ["counting_and_probability","intermediate_algebra","number_theory","precalculus","prealgebra","geometry","algebra"]
datas = [6]
dataname = [all_cate[i] for i in datas]
file_names = ['best_gpt4/test_{}_n16_1'.format(dataname_) for dataname_ in dataname]
# Q = "What is 3 plus 5 ?"
# S = "3 plus 5 equals to 8"


# score = cal_score(model,tokenizer,Q,S)

import json
def extract_math_answer(pred_str):
    if('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    elif 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans)!=0 and (ans[0] == '{'):
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
        if (len(ans) and ans[0] == '{'):
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


for file_name in file_names:
    all_data = []
    with open(file_name+'.jsonl', "r") as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)
        f.close()
    right_list = []
    wrong_list = []
    GT_list = []
    right_num = 0
    all_info = []
    for tem in tqdm(all_data):
        Q = re.sub("Problem:\n","",tem["question"])
        A = re.sub("\nSolution:\n","",tem["answer"])
        ans = remove_boxed(last_boxed_only_string(A))
        P_list = []
        for p in tem["generated_answer"][:16]:
            p = re.sub("\nSolution:\n", "", p)
            p = re.sub("Solution:\n", "",p)
            p = p.strip("\n")
            p = p.strip(" ")
            pre = remove_boxed(last_boxed_only_string(p))
            score = cal_score(model, tokenizer, Q, p)
            P_list.append((pre,ans,score,is_equiv(pre,ans)))
            if is_equiv(pre,ans):
                if score is not None:
                    right_list.append(score)
                right_num+=1
            else:
                if score is not None:
                    wrong_list.append(score)
        all_info.append(P_list)
        score = cal_score(model, tokenizer, Q, A)
        if score is not None:
            GT_list.append(score)



    print("Average GT Score: {}".format(numpy.array(GT_list).mean()))
    print("Right num {}\nAverage Right Score: {}".format(right_num,numpy.array(right_list).mean()))
    print("Wrong num {}\nAverage Wrong Score: {}".format(len(all_data)*16-right_num,numpy.array(wrong_list).mean()))
    with open(file_name+"-from{}-{}.json".format(ckpt_name1,ckpt_name2),"w")as f:
        json.dump(all_info,f)
        f.close()
# print(1)





