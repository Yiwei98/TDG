import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json
from dataclasses import dataclass
import logging
import random
# import nlpaug.augmenter.word as naw
# from nlpaug.flow import Sometimes

logger = logging.getLogger(__name__)


@dataclass
class Example(object):
    src: str
    target: str
    label: Optional[List[int]] = None


class SimDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
            data_model,
            sub_set="train",
            n_obs=None,
    ):
        super().__init__()
        self.examples = self.load_file(data_model,sub_set)
        if n_obs is not None and n_obs >= 1:
            self.examples = self.examples[:n_obs]
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.subset = sub_set
        self.pad_token_id = self.tokenizer.pad_token_id

    def load_file(self, data_model,subset):
        with open("{}/best-llama-data_for_RM_{}-weight.json".format(data_model,subset), "r") as f:
            contents = json.load(f)
            f.close()
        return contents

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class SimcseDataCollator:
    def __init__(self, tokenizer: BertTokenizer,max_length,neg_sampler=None,pos_bili=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.neg_sampler = neg_sampler
        self.pos_bili = pos_bili
        assert (
                self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        # self.data_args = data_args

    def __call__(self, batch):
        batch,labels,weights= self._encode(batch)
        batch = self.tokenizer.pad(
                batch,
                padding='longest',
                max_length=self.max_length,
                return_tensors="pt",
            )
        try:
            batch.data["special_tokens_mask"]=batch.data["special_tokens_mask"]*batch.data["attention_mask"]
        except:
            return None
        return [batch,labels,weights]

    def _encode(self, sentences: List[Example]):
        sen_num = []
        label_index = []
        labels = []
        weights=[]
        all_weights = []
        all_sentences = []
        # start_index = []
        # end_index = []
        # pos_pair = []
        #aug1 = naw.SynonymAug(aug_p=0.1)
        #word_aug = naw.RandomWordAug(aug_p=0.1)
        #aug2 = Sometimes([aug1, word_aug])
        for sentence in sentences:
            cur_num = 1
            question = sentence['question']
            all_sentences.append(question)
            l1 = random.choice(sentence['level1']) if len(sentence['level1'])>0 else None
            if len(sentence['level2']) > 0:
                idx = random.choice(range(len(sentence['level2'])))
                l2 = sentence['level2'][idx]
                weights.append(sentence['weight'][idx])
            else:
                weights.append(0)
                l2 = None
            #l2 = random.choice(sentence['level2']) if len(sentence['level2']) > 0 else None
            # l3 = random.choice(sentences)
            # l3 = random.choice(l3['level1']+l3['level2'])
            if l1 is not None:
                cur_num+=1
                all_sentences.append(l1)
                #all_sentences.append(aug2.augment(l1, 1))
            if l2 is not None:
                cur_num+=1
                all_sentences.append(l2)
                # all_sentences.append(aug2.augment(l2, 1))
            # if l3 is not None:
            #     cur_num+=1
            #     all_sentences.append(aug2.augment(l3, 1))
            sen_num.append(cur_num)


        sent_features = self.tokenizer(
            all_sentences,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        begin_word = sent_features["input_ids"][0][0]
        begin_at_mask = sent_features["attention_mask"][0][0]
        begin_sp_mask = 0


        cur_index = 0

        new_sent_features = {"input_ids": [],  "attention_mask": [], "special_tokens_mask": []}

        for k in range(len(sen_num)):
            for x in range(1,sen_num[k]):
                for y in range(x+1,sen_num[k]):


                    tem0_true,tem0_false = [begin_word],[begin_word]
                    tem2_true,tem2_false = [begin_at_mask],[begin_at_mask]
                    tem3_true,tem3_false = [begin_sp_mask],[begin_sp_mask]

                    flag = 1
                    tem0_true += sent_features["input_ids"][cur_index][1:]
                    # tem1_true += [flag for i in range(len(sent_features["token_type_ids"][cur_index][1:]))]
                    tem2_true += sent_features["attention_mask"][cur_index][1:]
                    tem3_true += [0 for i in range(len(sent_features["attention_mask"][cur_index][1:]))]
                    tem0_false += sent_features["input_ids"][cur_index][1:]
                    # tem1_false += [flag for i in range(len(sent_features["token_type_ids"][cur_index][1:]))]
                    tem2_false += sent_features["attention_mask"][cur_index][1:]
                    tem3_false += [0 for i in range(len(sent_features["attention_mask"][cur_index][1:]))]

                    tem0_true += sent_features["input_ids"][cur_index+x][1:]
                    # tem1_true += [flag for i in range(len(sent_features["token_type_ids"][cur_index+x][1:]))]
                    tem2_true += sent_features["attention_mask"][cur_index+x][1:]
                    tem3_true += [1 for i in range(len(sent_features["attention_mask"][cur_index+x][1:]))]
                    tem3_true[-1] -= 1
                    tem0_false += sent_features["input_ids"][cur_index+y][1:]
                    # tem1_false += [flag for i in range(len(sent_features["token_type_ids"][cur_index+y][1:]))]
                    tem2_false += sent_features["attention_mask"][cur_index+y][1:]
                    tem3_false += [1 for i in range(len(sent_features["attention_mask"][cur_index+y][1:]))]
                    tem3_false[-1] -= 1

                    if len(tem0_true) < self.max_length and len(tem0_false) < self.max_length:
                        new_sent_features["input_ids"].append(tem0_true)
                        # new_sent_features["token_type_ids"].append(tem1_true)
                        new_sent_features["attention_mask"].append(tem2_true)
                        new_sent_features["special_tokens_mask"].append(tem3_true)
                        new_sent_features["input_ids"].append(tem0_false)
                        # new_sent_features["token_type_ids"].append(tem1_false)
                        new_sent_features["attention_mask"].append(tem2_false)
                        new_sent_features["special_tokens_mask"].append(tem3_false)
                        tt = 1 if y==(sen_num[k]-1) else 0
                        labels.append(tt)
                        all_weights.append(weights[k])

            cur_index+=sen_num[k]
        return new_sent_features,labels,all_weights

class EvalDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_length,
    ):
        super().__init__()
        data_file = Path(data_dir)
        self.examples = self.load_file(data_file)
        self.max_length = max_length
        self.tokenizer = tokenizer

    def load_file(self, filename):
        with open(filename, 'r') as f:
            contents = json.load(f)
            f.close()
        return contents

    def __len__(self):
        return len(self.examples["contexts"])

    def __getitem__(self, index) -> Dict[str, str]:
        if "references" in self.examples.keys():
            return [self.examples["contexts"][index], self.examples["responses"][index],
                    self.examples["scores"][index], self.examples["references"][index]]
        return [self.examples["contexts"][index], self.examples["responses"][index], self.examples["scores"][index],
                "none"]


class EvalDataCollator:
    def __init__(self, tokenizer: BertTokenizer,max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        assert (
                self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        # self.data_args = data_args

    def __call__(self, batches):
        batch,labels,contexts,responses,references,c_r_mask = self._encode(batches)
        batch = self.tokenizer.pad(
                batch,
                padding='longest',
                max_length=self.max_length,
                return_tensors="pt",
            )
        length = batch.data["special_tokens_mask"].shape[1]
        c_r_mask=[tem+[0]*(length - len(tem)) for tem in c_r_mask]
        batch.data["c_r_ids"] = torch.tensor(c_r_mask)
        batch.data["labels"] = labels
        batch.data["special_tokens_mask"]=batch.data["special_tokens_mask"]*batch.data["attention_mask"]
        batch.data["responses"] = responses
        batch.data["contexts"] = contexts
        batch.data["references"] = references
        return batch

    def _encode(self, batches):
        all_sentences = []
        labels = []
        sen_len = []
        contexts = []
        responses = []
        references = []
        c_r_mask = []
        for tem in batches:
            all_sentences += tem[0]
            sen_len.append(len(tem[0]))
            all_sentences.append(tem[1])
            labels.append(tem[2])
            contexts.append(tem[0])
            responses.append(tem[1])
            references.append(tem[3])

        sent_features = self.tokenizer(
            all_sentences,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        begin_word = sent_features["input_ids"][0][0]
        begin_at_mask = sent_features["attention_mask"][0][0]
        begin_sp_mask = 0
        cur_index = 0
        new_sent_features = {"input_ids": [], "token_type_ids": [], "attention_mask": [], "special_tokens_mask": []}
        for k in range(len(batches)):
            num = sen_len[k]  #################3
            flag = 1
            tem0, tem1, tem2, tem3, tem4 = [], [], [], [], []
            tem0 += sent_features["input_ids"][cur_index + sen_len[k]][1:]
            tem1 += [flag for i in range(len(sent_features["token_type_ids"][cur_index + sen_len[k]][1:]))]
            tem2 += sent_features["attention_mask"][cur_index + sen_len[k]][1:]
            tem3 += [1 for i in range(len(sent_features["attention_mask"][cur_index + sen_len[k]][1:]))]
            tem3[-1] -= 1
            tem4 += [2 for i in range(len(sent_features["attention_mask"][cur_index + sen_len[k]][1:]))]

            for j in range(num):
                i = num - j - 1
                it = i + cur_index
                flag = (flag + 1) % 2
                if len(sent_features["input_ids"][it][1:]) + len(tem0) + 1 > self.max_length:
                    break
                tem0 = sent_features["input_ids"][it][1:] + tem0
                tem1 = [flag for i in range(len(sent_features["token_type_ids"][it][1:]))] + tem1
                tem2 = sent_features["attention_mask"][it][1:] + tem2
                tem3 = [0 for i in range(len(sent_features["attention_mask"][it][1:]))] + tem3
                if j == 0:
                    tem4 = [1 for i in range(len(sent_features["attention_mask"][it][1:]))] + tem4
                else:
                    tem4 = [0 for i in range(len(sent_features["attention_mask"][it][1:]))] + tem4
            tem0 = [begin_word] + tem0
            tem1 = [tem1[0]] + tem1
            tem2 = [begin_at_mask] + tem2
            tem3 = [begin_sp_mask] + tem3
            tem4 = [0] + tem4

            new_sent_features["input_ids"].append(tem0)
            new_sent_features["token_type_ids"].append(tem1)
            new_sent_features["attention_mask"].append(tem2)
            new_sent_features["special_tokens_mask"].append(tem3)
            c_r_mask.append(tem4)
            cur_index += num + 1

        return new_sent_features,labels,contexts,responses,references,c_r_mask




class RobustDataCollator:
    def __init__(self, tokenizer: BertTokenizer,max_length,neg_sampler=None,pos_bili=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.neg_sampler = neg_sampler
        self.pos_bili = pos_bili
        assert (
                self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        # self.data_args = data_args

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        batch,labels,_ ,c_r_mask= self._encode(batch,self.pos_bili)
        batch = self.tokenizer.pad(
                batch,
                padding='longest',
                max_length=self.max_length,
                return_tensors="pt",
            )
        length = batch.data["special_tokens_mask"].shape[1]
        c_r_mask=[tem+[0]*(length - len(tem)) for tem in c_r_mask]
        batch.data["special_tokens_mask"]=batch.data["special_tokens_mask"]*batch.data["attention_mask"]
        batch.data["labels"] = torch.tensor(labels)
        batch.data["pos_pair"] = torch.tensor([])
        batch.data["c_r_ids"] = torch.tensor(c_r_mask)
        return batch

    def _encode(self, sentences: List[Example], pos_bili):
        sen_num = []
        label_index = []
        labels = []
        all_sentences1,all_sentences2 = [],[]
        start_index = []
        end_index = []
        pos_pair = []
        aug1 = naw.SynonymAug(aug_p=0.1)
        word_aug = naw.RandomWordAug(aug_p=0.15)
        aug2 = Sometimes([aug1, word_aug])
        for sentence in sentences:
            split_sentence = sentence.split("__eou__")
            if len(split_sentence) < 2:
                continue
            length_r = random.randint(2,min(len(split_sentence),10))
            start = random.randint(0,len(split_sentence)-length_r)
            end = start+length_r-1
            start_index.append(start)
            end_index.append(end)
            label_index.append(end)
            #split_sentence = [aug2.augment(split_sentence[it], 1) if it != end else aug1.augment(split_sentence[it], 1) for it in range(len(split_sentence))]
            split_sentence1 = split_sentence
            split_sentence2 = []
            for it in range(len(split_sentence)):
                if it != end and it != end-1:
                    split_sentence2.append(aug2.augment(split_sentence[it], 1))
                elif it == end-1:
                    split_sentence2.append(aug1.augment(split_sentence[it], 1))
                else:
                    split_sentence2.append(split_sentence[it])
            sen_num.append(len(split_sentence1))
            all_sentences1 += split_sentence1
            all_sentences2 += split_sentence2

        total_length = len(all_sentences1)
        sent_features = self.tokenizer(
            all_sentences1+all_sentences2,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        begin_word = sent_features["input_ids"][0][0]
        begin_at_mask = sent_features["attention_mask"][0][0]
        begin_sp_mask = 0
        begin_to_type = sent_features["token_type_ids"][0][0]

        cur_index = 0
        new_sent_features = {"input_ids": [], "token_type_ids": [], "attention_mask": [], "special_tokens_mask": []}
        c_r_mask = []
        for k in range(len(sen_num)):
            write_flag = False
            num = sen_num[k]
            tem0_true,tem0_false,tem0_true_,tem0_false_ = [begin_word],[begin_word],[begin_word],[begin_word]
            tem1_true,tem1_false,tem1_true_,tem1_false_ = [begin_to_type],[begin_to_type],[begin_to_type],[begin_to_type]
            tem2_true,tem2_false,tem2_true_,tem2_false_ = [begin_at_mask],[begin_at_mask],[begin_at_mask],[begin_at_mask]
            tem3_true,tem3_false,tem3_true_,tem3_false_ = [begin_sp_mask],[begin_sp_mask],[begin_sp_mask],[begin_sp_mask]
            tem4_true, tem4_false,tem4_true_, tem4_false_ = [0],[0],[0],[0]
            flag = 1
            sen_type = 0
            exchanger = []
            c_len = end_index[k]+1 - start_index[k] - 2
            if c_len >= 3:
                if random.random()<pos_bili:
                    while True:
                        exchanger = random.sample(range(start_index[k],start_index[k]+c_len),2)
                        if (exchanger[0]+exchanger[1])%2==0:
                            break

            for i in range(start_index[k],end_index[k]+1):
                predict_flag = 0
                if i in exchanger:
                    i = exchanger[0]+exchanger[1]-i
                it = i + cur_index
                flag = (flag + 1) % 2
                if it == cur_index + label_index[k]-1:
                    sen_type = 1
                # 在label位置
                if it == cur_index + label_index[k]:
                    sen_type = 2
                    predict_flag = 1
                    if random.random()<(sen_num[k]/total_length)*2:
                        r_num = random.choice(range(0,num))+cur_index
                        neg_sample = {"input_ids": [sent_features["input_ids"][r_num]],
                                           "attention_mask":[sent_features["attention_mask"][r_num]],
                                           "token_type_ids":[sent_features["token_type_ids"][r_num]]}
                    else:
                        neg_sen = self.neg_sampler.get_neg(all_sentences1[it])
                        neg_sample = self.tokenizer(
                                    [neg_sen],
                                    max_length=self.max_length,
                                    truncation=True,
                                    padding=False,)
                    if len(tem0_true) + len(sent_features["input_ids"][it]) - 1 > self.max_length:
                        break
                    if len(tem0_false) + len(neg_sample["input_ids"][0]) - 1 > self.max_length:
                        break
                    if len(tem0_true_) + len(sent_features["input_ids"][it]) - 1 > self.max_length:
                        break
                    if len(tem0_false_) + len(neg_sample["input_ids"][0]) - 1 > self.max_length:
                        break
                    if len(neg_sample["input_ids"][0])<=2 or len(sent_features["input_ids"][it])<=2:
                        break
                    write_flag = True
                else:
                    # 如果下一句加入句列后句列长度已经超过max_length，则不再加入
                    if len(tem0_true)+len(sent_features["input_ids"][it])-1>self.max_length:
                        break
                    if len(tem0_true_)+len(sent_features["input_ids"][it+total_length])-1>self.max_length:
                        break

                if it != cur_index + label_index[k]:
                    tem0_true += sent_features["input_ids"][it][1:]
                    tem1_true += [flag for i in range(len(sent_features["token_type_ids"][it][1:]))]
                    tem2_true += sent_features["attention_mask"][it][1:]
                    tem3_true += [predict_flag for i in range(len(sent_features["attention_mask"][it][1:]))]
                    tem3_true[-1] -= predict_flag
                    tem4_true += [sen_type for i in range(len(sent_features["attention_mask"][it][1:]))]
                    tem0_true_ += sent_features["input_ids"][it+total_length][1:]
                    tem1_true_ += [flag for i in range(len(sent_features["token_type_ids"][it+total_length][1:]))]
                    tem2_true_ += sent_features["attention_mask"][it+total_length][1:]
                    tem3_true_ += [predict_flag for i in range(len(sent_features["attention_mask"][it+total_length][1:]))]
                    tem3_true_[-1] -= predict_flag
                    tem4_true_ += [sen_type for i in range(len(sent_features["attention_mask"][it+total_length][1:]))]

                    tem0_false += sent_features["input_ids"][it][1:]
                    tem1_false += [flag for i in range(len(sent_features["token_type_ids"][it][1:]))]
                    tem2_false += sent_features["attention_mask"][it][1:]
                    tem3_false += [predict_flag for i in range(len(sent_features["attention_mask"][it][1:]))]
                    tem3_false[-1] -= predict_flag
                    tem4_false += [sen_type for i in range(len(sent_features["attention_mask"][it][1:]))]
                    tem0_false_ += sent_features["input_ids"][it+total_length][1:]
                    tem1_false_ += [flag for i in range(len(sent_features["token_type_ids"][it+total_length][1:]))]
                    tem2_false_ += sent_features["attention_mask"][it+total_length][1:]
                    tem3_false_ += [predict_flag for i in range(len(sent_features["attention_mask"][it+total_length][1:]))]
                    tem3_false_[-1] -= predict_flag
                    tem4_false_ += [sen_type for i in range(len(sent_features["attention_mask"][it+total_length][1:]))]

                else:
                    tem0_true += sent_features["input_ids"][it][1:]
                    tem1_true += [flag for i in range(len(sent_features["token_type_ids"][it][1:]))]
                    tem2_true += sent_features["attention_mask"][it][1:]
                    tem3_true += [predict_flag for i in range(len(sent_features["attention_mask"][it][1:]))]
                    tem3_true[-1] -= predict_flag
                    tem4_true  += [sen_type for i in range(len(sent_features["attention_mask"][it][1:]))]
                    tem0_true_ += sent_features["input_ids"][it][1:]
                    tem1_true_ += [flag for i in range(len(sent_features["token_type_ids"][it][1:]))]
                    tem2_true_ += sent_features["attention_mask"][it][1:]
                    tem3_true_ += [predict_flag for i in range(len(sent_features["attention_mask"][it][1:]))]
                    tem3_true_[-1] -= predict_flag
                    tem4_true_  += [sen_type for i in range(len(sent_features["attention_mask"][it][1:]))]

                    tem0_false += neg_sample["input_ids"][0][1:]
                    tem1_false += [flag for i in range(len(neg_sample["token_type_ids"][0][1:]))]
                    tem2_false += neg_sample["attention_mask"][0][1:]
                    tem3_false += [1 for i in range(len(neg_sample["attention_mask"][0][1:]))]
                    tem3_false[-1] -= predict_flag
                    tem4_false += [sen_type for i in range(len(neg_sample["attention_mask"][0][1:]))]
                    tem0_false_ += neg_sample["input_ids"][0][1:]
                    tem1_false_ += [flag for i in range(len(neg_sample["token_type_ids"][0][1:]))]
                    tem2_false_ += neg_sample["attention_mask"][0][1:]
                    tem3_false_ += [1 for i in range(len(neg_sample["attention_mask"][0][1:]))]
                    tem3_false_[-1] -= predict_flag
                    tem4_false_ += [sen_type for i in range(len(neg_sample["attention_mask"][0][1:]))]

            if write_flag:
                new_sent_features["input_ids"].append(tem0_true)
                new_sent_features["token_type_ids"].append(tem1_true)
                new_sent_features["attention_mask"].append(tem2_true)
                new_sent_features["special_tokens_mask"].append(tem3_true)
                c_r_mask.append(tem4_true)
                new_sent_features["input_ids"].append(tem0_true_)
                new_sent_features["token_type_ids"].append(tem1_true_)
                new_sent_features["attention_mask"].append(tem2_true_)
                new_sent_features["special_tokens_mask"].append(tem3_true_)
                c_r_mask.append(tem4_true_)
                new_sent_features["input_ids"].append(tem0_false)
                new_sent_features["token_type_ids"].append(tem1_false)
                new_sent_features["attention_mask"].append(tem2_false)
                new_sent_features["special_tokens_mask"].append(tem3_false)
                c_r_mask.append(tem4_false)
                new_sent_features["input_ids"].append(tem0_false_)
                new_sent_features["token_type_ids"].append(tem1_false_)
                new_sent_features["attention_mask"].append(tem2_false_)
                new_sent_features["special_tokens_mask"].append(tem3_false_)
                c_r_mask.append(tem4_false_)
            cur_index += num
        return new_sent_features,labels,pos_pair,c_r_mask
# # 检验simcse data loader及collector
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# dataset = SimDataset(tokenizer,"raw_data/dialogues_text.txt",128)
# data_sampler = RandomSampler(dataset)
# train_dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=16,
#                               collate_fn=SimcseDataCollator(tokenizer,128,batch_form = "c_r"))
# for step, batch in enumerate(train_dataloader):
#     for i in range(16):
#         print(batch.data['attention_mask'][i])
#         print(batch.data['special_tokens_mask'][i])
#     for i in range(16):
#         print(batch.data['input_ids'][i])
#     print("!!!!!!")


# # 检验simcse data loader及collector
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# dataset = EvalDataset(tokenizer,"eval_data/dailydialog_grade_transformer_generator_with_fut_his.json",128,"c_r_c")
# data_sampler = RandomSampler(dataset)
# train_dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=16,
#                               collate_fn=EvalDataCollator(tokenizer,128,batch_form = "c_r_c"))
# for step, batch in enumerate(train_dataloader):
#     for i in range(16):
#         print(batch.data['attention_mask'][i])
#         print(batch.data['special_tokens_mask'][i])
#     for i in range(16):
#         print(batch.data['input_ids'][i])
#     print("!!!!!!")
