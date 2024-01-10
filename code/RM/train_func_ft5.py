import numpy
import tqdm
import torch
from data_ft5 import EvalDataset,EvalDataCollator
from torch.utils.data import (
    DataLoader,
    RandomSampler
)
import matplotlib.pyplot as plt
import os
import time
import csv
from scipy import stats


def ft_epoch(model, ftDataLoader, optimizer, device):
    model.train()
    model.is_ft = True
    total_loss, steps = 0, 0
    epoch_iterator = tqdm.tqdm(ftDataLoader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
        input_ids = batch["input_ids"].to(device)
        special_tokens_mask = batch["special_tokens_mask"].to(device)
        c_r_ids = batch["c_r_ids"].to(device)
        labels = batch['labels'].to(device)
        inputs = {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "token_type_ids" : token_type_ids,
            "special_tokens_mask" : special_tokens_mask,
            "c_r_ids":c_r_ids,
            "labels": labels
        }

        # forward
        optimizer.zero_grad()
        output = model(**inputs)
        loss = output
        # backward and update parameters
        loss.backward()
        optimizer.step()

        # note keeping
        total_loss += loss.item()
        steps += 1

    loss_per_step = total_loss / steps
    model.is_ft = False
    return loss_per_step



def train_epoch(model, train_dataloader, optimizer, device):
    model.train()
    total_loss, steps,t_l1,t_l2,t_acc = 0, 0,0,0,0
    epoch_iterator = tqdm.tqdm(train_dataloader, desc=f"Loss: 0.0000")
    for step, batch_ in enumerate(epoch_iterator):
        if batch_ is not None:
            batch = batch_[0]
            labels = batch_[1]
            weights = batch_[2]
            input_ids = batch["input_ids"].to(device)
            special_tokens_mask = batch["special_tokens_mask"].to(device)
            inputs = {
                "input_ids" : input_ids,
                "special_tokens_mask" : special_tokens_mask,
                "labels":torch.tensor(labels).to(device),
                "weights": torch.tensor(weights).to(device),
            }

            # forward
            optimizer.zero_grad()
            output,l1,l2 ,acc= model(**inputs)
            loss = output
            # backward and update parameters
            loss.backward()
            optimizer.step()

            # note keeping
            if loss.item()<1e-5:
                total_loss+=0.25
            else:
                total_loss += loss.item()
            t_l1+=l1.item()
            t_l2+=l2.item()
            t_acc += acc.item()
            steps+=1
            epoch_iterator.set_description(f"Loss: {loss.item():.4f}  ACC:{t_acc / steps:.4f}  L1: {l1.item():.4f}  L2: {l2.item():.4f}")
    loss_per_step = total_loss / steps
    l1_per_step = t_l1 / steps
    l2_per_step = t_l2 / steps
    t_acc_per_step = t_acc / steps

    return loss_per_step, l1_per_step, l2_per_step, t_acc_per_step


def valid_epoch(model, valid_dataloader, device):
    model.eval()

    with torch.no_grad():
        total_loss, steps,t_l1,t_l2,t_acc = 0, 0,0,0,0
        epoch_iterator = tqdm.tqdm(valid_dataloader, desc="Iteration")
        for step, batch_ in enumerate(epoch_iterator):
            if batch_ is not None:
                batch = batch_[0]
                labels = batch_[1]
                weights = batch_[2]
                if batch is not None:
                    input_ids = batch["input_ids"].to(device)
                    special_tokens_mask = batch["special_tokens_mask"].to(device)
                    inputs = {
                        "input_ids": input_ids,
                        "special_tokens_mask": special_tokens_mask,
                        "labels":torch.tensor(labels).to(device),
                        "weights": torch.tensor(weights).to(device),
                    }

                    # forward
                    output,l1,l2,acc = model(**inputs)
                    loss = output
                    # backward and update parameters

                    # note keeping

                    total_loss += loss.item()
                    t_l1 += l1.item()
                    t_l2 += l2.item()
                    t_acc += acc.item()
                    steps += 1
                    epoch_iterator.set_description(f"Loss: {loss.item():.4f}  ACC:{t_acc / steps:.4f}  L1: {l1.item():.4f}  L2: {l2.item():.4f}")
        loss_per_step = total_loss / steps
        l1_per_step = t_l1 / steps
        l2_per_step = t_l2 / steps
        t_acc_per_step = t_acc / steps

        return loss_per_step, l1_per_step, l2_per_step,t_acc_per_step

def eval_epoch(model, eval_dataloader, device):
    model.eval()
    model.is_ft = False
    model.to_eval(True)
    with torch.no_grad():
        epoch_iterator = tqdm.tqdm(eval_dataloader, desc="Iteration")
        scores,all_labels,contexts,responses,references = [], [], [], [], []
        for step, batch_ in enumerate(epoch_iterator):
            batch = batch_[0]
            labels = batch_[1]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            special_tokens_mask = batch["special_tokens_mask"].to(device)
            labels = batch["labels"]
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "special_tokens_mask": special_tokens_mask,
                "c_r_ids":attention_mask,
                "labels":labels
            }

            # forward
            score,_ = model(**inputs)
            scores+=score
            all_labels+=label
            contexts+=batch["contexts"]
            responses+=batch["responses"]
            references+=batch["references"]
        model.to_eval(False)
        return scores,all_labels,contexts,responses,references



def eval_func_multi(models, eval_list1,eval_list2,check_list, device,output_dirs,tokenizer,output_name="test",reload_best_model=True, epoch_name="Final"):
    if os.path.exists(output_name) is False:
        os.makedirs(output_name)
    if os.path.exists(output_name + "/item_check") is False:
        os.makedirs(output_name + "/item_check")
    if reload_best_model:
        for i in range(len(output_dirs)):
            models[i].load_checkpoint(output_dirs[i] + "/model.pth", device)
    if len(eval_list1) > 0:
        # 重新载入模型
        print("Start Eval_list1!!")
        with open(output_name + "/result.txt", "a") as f:
            f.write("\n\n\n{}:\n".format(epoch_name))
            f.close()
        sp_all = 0
        pe_all = 0
        # 开始检测自动评估
        for data in eval_list1:
            data_dir = "eval_data/" + data
            start = time.time()
            scores = []
            for model in models:
                dataset = EvalDataset(tokenizer, data_dir, 128)
                eval_dataloader = DataLoader(dataset, batch_size=32,
                                             collate_fn=EvalDataCollator(tokenizer, 128))
                score, label, contexts, responses, references = eval_epoch(model, eval_dataloader, device=device)
                scores.append(numpy.array(score))
            score = numpy.array(scores)
            score_min = score.min(0)
            score = list(score.sum(0)-score_min)
            if data in ["topicalchat_usr.json", "personachat_usr.json", "fed.json"]:
                factor_list = list(label[0].keys())
                for factor in factor_list:
                    f_label = [tem[factor] for tem in label]
                    r, p = stats.pearsonr(score, f_label)
                    spearmanr = stats.spearmanr(score, f_label)
                    if factor == "Overall":
                        print("Add overall!")
                        sp_all += spearmanr.correlation
                        pe_all += r
                    if epoch_name == "Final":
                        print(data + ":  " + factor)
                        print("pearson_score: {}   p_value: {}".format(r, p))
                        print("spearmanr_score: {}   p_value: {}".format(spearmanr.correlation, spearmanr.pvalue))
                        if data in check_list:
                            with open(output_name+"/item_check/"+data+".csv","a")as f:
                                writer = csv.writer(f)
                                writer.writerow([factor])
                                writer.writerow(["Sp","Sp-p","Pe","pe-p"])
                                writer.writerow([spearmanr.correlation, spearmanr.pvalue,r, p])
                                f.close()
                    with open(output_name + "/result.txt", "a") as f:
                        f.write("\n{}:\n".format(data + ":  " + factor))
                        f.write("pearson_score: {}   p_value: {}\n".format(r, p))
                        f.write("spearmanr_score: {}   p_value: {}\n".format(spearmanr.correlation, spearmanr.pvalue))
                        f.close()

            else:
                label = [float(tem) for tem in label]
                r, p = stats.pearsonr(score, label)
                spearmanr = stats.spearmanr(score, label)
                sp_all += spearmanr.correlation
                pe_all += r
                if epoch_name == "Final":
                    print(data)
                    print("pearson_score: {}   p_value: {}".format(r, p))
                    print("spearmanr_score: {}   p_value: {}".format(spearmanr.correlation, spearmanr.pvalue))
                    if data in check_list:
                        with open(output_name + "/item_check/" + data + ".csv", "a") as f:
                            writer = csv.writer(f)
                            writer.writerow(["Sp", "Sp-p", "Pe", "pe-p"])
                            writer.writerow([spearmanr.correlation, spearmanr.pvalue, r, p])
                            f.close()
                with open(output_name + "/result.txt", "a") as f:
                    f.write("\n{}:\n".format(data))
                    f.write("pearson_score: {}   p_value: {}\n".format(r, p))
                    f.write("spearmanr_score: {}   p_value: {}\n".format(spearmanr.correlation, spearmanr.pvalue))
                    f.close()

            if epoch_name=="Final" and data in check_list:
                score_= score.copy()
                score_.sort()
                score_dict = {}
                to_length = len(score_)
                for i in range(to_length):
                    score_dict[score_[i]] = (i+1)/to_length
                score_rank = [score_dict[tem] for tem in score]
                label_= label.copy()
                label_.sort()
                to_length = len(label_)
                label_dict = {}
                for i in range(to_length):
                    label_dict[label_[i]] = (i+1)/to_length
                label_rank = [label_dict[tem] for tem in label]
                with open(output_name + "/item_check/" + data + ".csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([])
                    writer.writerow(["contexts", "response", "references", "label", "score", "label_rank", "score_rank"])
                    for i in range(min(len(score),200)):
                        writer.writerow([contexts[i],responses[i],references[i],label[i],score[i],label_rank[i],score_rank[i]])
                    f.close()

        with open(output_name + "/result.txt", "a") as f:
            f.write("\nOver_all Average pearson_score:{}\nOver_all Average spearmanr_score:{}\n".format(pe_all/len(eval_list1),sp_all/len(eval_list1)))
            f.close()
        print("Over_all Average pearson_score:{}\nOver_all Average spearmanr_score:{}".format(pe_all/len(eval_list1),sp_all/len(eval_list1)))

    if len(eval_list2)>0:
        # 重新载入模型
        print("Start Eval_list2!!")
        with open(output_name + "/result.txt", "a") as f:
            f.write("\n\nDSTC10 DATASETS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            f.close()
        sp_all = 0
        pe_all = 0
        sp_oall = 0
        pe_oall = 0
        metric_num = 0
        o_metric_num = 0
        # 开始检测自动评估
        for data in eval_list2:
            data_dir = "dstc10_eval_data/" + data
            scores = []
            for model in models:
                dataset = EvalDataset(tokenizer, data_dir, 128)
                eval_dataloader = DataLoader(dataset, batch_size=32,
                                             collate_fn=EvalDataCollator(tokenizer, 128))
                score, label, contexts, responses, references = eval_epoch(model, eval_dataloader, device=device)
                score = [0.0 if str(tem) == "nan" else tem for tem in score]
                scores.append(numpy.array(score))
            score = numpy.array(scores)
            score_min = score.min(0)
            score = list(score.sum(0) - score_min)
            factor_list = list(label[0].keys())
            metric_num+=len(factor_list)
            for factor in factor_list:
                f_label = [float(tem[factor]) for tem in label]
                score = [0.0 if str(tem) == "nan" else tem for tem in score]
                r, p = stats.pearsonr(score, f_label)
                spearmanr = stats.spearmanr(score, f_label)
                if factor in ["appropriateness"]:
                    sp_oall += spearmanr.correlation
                    pe_oall += r
                    o_metric_num += 1
                sp_all += spearmanr.correlation
                pe_all += r
                if epoch_name == "Final":
                    print(data + ":  " + factor)
                    print("pearson_score: {}   p_value: {}".format(r, p))
                    print("spearmanr_score: {}   p_value: {}".format(spearmanr.correlation, spearmanr.pvalue))
                    if data in check_list:
                        with open(output_name + "/item_check/" + data + ".csv", "a") as f:
                            writer = csv.writer(f)
                            writer.writerow([factor])
                            writer.writerow(["Sp", "Sp-p", "Pe", "pe-p"])
                            writer.writerow([spearmanr.correlation, spearmanr.pvalue, r, p])
                            f.close()
                with open(output_name + "/result.txt", "a") as f:
                    f.write("\n{}:\n".format(data + ":  " + factor))
                    f.write("pearson_score: {}   p_value: {}\n".format(r, p))
                    f.write("spearmanr_score: {}   p_value: {}\n".format(spearmanr.correlation, spearmanr.pvalue))
                    f.close()
            if data in check_list:
                score_= score.copy()
                score_.sort()
                score_dict = {}
                to_length = len(score_)
                for i in range(to_length):
                    score_dict[score_[i]] = (i+1)/to_length
                score_rank = [score_dict[tem] for tem in score]
                label_= f_label.copy()
                label_.sort()
                label_set = set(label_)
                label_dict = {}
                to_length = len(label_)
                for i in range(to_length):
                    label_dict[label_[i]] = (i+1)/to_length
                label_rank = [label_dict[tem] for tem in f_label]
                with open(output_name + "/item_check/" + data + ".csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([])
                    writer.writerow(["contexts", "response", "references", "label", "score", "label_rank", "score_rank"])
                    for i in range(min(len(score),200)):
                        writer.writerow([contexts[i],responses[i],references[i],f_label[i],score[i],label_rank[i],score_rank[i]])
                    f.close()

        with open(output_name + "/result.txt", "a") as f:
            f.write("\nAll DSTC10 Average pearson_score:{}\nAll DSTC10 Average spearmanr_score:{}\n".format(
                pe_all / metric_num, sp_all / metric_num))
            f.write("\nAll DSTC10 Overall Average pearson_score:{}\nAll DSTC10 Overall Average spearmanr_score:{}\n".format(
                pe_oall / o_metric_num, sp_oall / o_metric_num))
            f.close()
        print("Over_all DSTC10 Average pearson_score:{}\nOver_all DSTC10 Average spearmanr_score:{}".format(pe_all / metric_num,
                                                                                              sp_all / metric_num))
        print("All DSTC10 Overall Average pearson_score:{}\nAll DSTC10 Overall Average spearmanr_score:{}".format(
                pe_oall / o_metric_num, sp_oall / o_metric_num))


def plot_hist(data,save_dir,data_name,hists_num=100):
    plt.figure(figsize=(6, 3))
    plt.xlabel('score', color='gray',fontsize=10,labelpad=-2)
    plt.ylabel('density', color='gray',fontsize=10)
    weights = numpy.ones_like(data) / float(len(data))*100
    plt.hist(data, bins=hists_num,weights=weights,color='indianred')
    plt.savefig(save_dir+"/"+data_name+"_pl.pdf")
    plt.close()


def eval_robust(model, eval_dataloader, device,output_dir):
    model.eval()
    model.to_eval(True)
    scores1, scores2 = [], []
    with torch.no_grad():
        epoch_iterator = tqdm.tqdm(eval_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            special_tokens_mask = batch["special_tokens_mask"].to(device)
            c_r_ids = batch["c_r_ids"].to(device)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "special_tokens_mask": special_tokens_mask,
                "c_r_ids":c_r_ids,
            }

            # forward
            score,_ = model(**inputs)
            scores1+=[score[i*2] for i in range(int(len(score)/2))]
            scores2 += [score[i*2+1] for i in range(int(len(score) / 2))]
        model.to_eval(False)
    plot_hist(scores1, output_dir + "/score_check", "robust_data", hists_num=100)
    r, p = stats.pearsonr(scores1, scores2)
    spearmanr = stats.spearmanr(scores1, scores2)
    print("Robust:")
    print("pearson_score: {}   p_value: {}".format(r, p))
    print("spearmanr_score: {}   p_value: {}".format(spearmanr.correlation, spearmanr.pvalue))
    with open(output_dir + "/result.txt", "a") as f:
        f.write("\nrobust:\npearsonr:{} p:{}\nspearmanr:{}  p:{}\n".format(r,p,spearmanr.correlation, spearmanr.pvalue))
        f.close()
    cut = 1/100
    scores1 = numpy.array(scores1)
    score_qj = [((scores1>i*0.01)*(scores1<=(i+1)*0.01)).sum()/len(scores1) for i in range(100)]
    score_bias = ((numpy.array(score_qj) - cut)*(numpy.array(score_qj) - cut)).mean()
    print("Robust bias: {}".format(score_bias))
    with open(output_dir + "/result.txt", "a") as f:
        f.write("\nRobust bias: {}\n".format(score_bias))
        f.close()
    return None

def eval_func(model, eval_list1,eval_list2,check_list, device,output_dir,tokenizer,reload_best_model=True, epoch_name="Final"):
    if reload_best_model:
        model.load_checkpoint(output_dir + "/model.pth", device)
    if len(eval_list1) > 0:
        # 重新载入模型
        print("Start Eval_list1!!")
        with open(output_dir + "/result.txt", "a") as f:
            f.write("\n\n\n{}:\n".format(epoch_name))
            f.close()
        sp_all = 0
        pe_all = 0
        # 开始检测自动评估
        for data in eval_list1:
            data_dir = "eval_data/" + data
            dataset = EvalDataset(tokenizer, data_dir, 128)
            eval_dataloader = DataLoader(dataset, batch_size=32,
                                         collate_fn=EvalDataCollator(tokenizer, 128))

            start = time.time()
            score, label, contexts, responses, references = eval_epoch(model, eval_dataloader, device=device)

            mid_samples_score,mid_samples_label,polar_samples_score,polar_samples_label,mid_samples_score2,mid_samples_label2,polar_samples_score2,polar_samples_label2 = [],[],[],[],[],[],[],[]
            for i in range(len(score)):
                if label[i]["rank"]<0.25:
                    polar_samples_score.append(score[i])
                    polar_samples_label.append(float(label[i]["Overall"]))
                elif label[i]["rank"]>0.75:
                    polar_samples_score2.append(score[i])
                    polar_samples_label2.append(float(label[i]["Overall"]))
                elif label[i]["rank"]>0.5:
                    mid_samples_score2.append(score[i])
                    mid_samples_label2.append(float(label[i]["Overall"]))
                else:
                    mid_samples_score.append(score[i])
                    mid_samples_label.append(float(label[i]["Overall"]))
            r1, p1 = stats.pearsonr(polar_samples_score, polar_samples_label)
            spearmanr1 = stats.spearmanr(polar_samples_score, polar_samples_label)
            r2, p2 = stats.pearsonr(polar_samples_score2, polar_samples_label2)
            spearmanr2 = stats.spearmanr(polar_samples_score2, polar_samples_label2)
            print(data + ":  " + "polar overall")
            print("pearson_score: {}   p_value: {}".format((r1+r2)/2, max(p1,p2)))
            print("spearmanr_score: {}   p_value: {}".format((spearmanr1.correlation+spearmanr2.correlation)/2, max(spearmanr1.pvalue,spearmanr2.pvalue)))
            with open(output_dir + "/result.txt", "a") as f:
                f.write("\n{}:\n".format(data + ":  " + "polar overall"))
                f.write("pearson_score: {}   p_value: {}\n".format((r1+r2)/2, max(p1,p2)))
                f.write("spearmanr_score: {}   p_value: {}\n".format((spearmanr1.correlation+spearmanr2.correlation)/2, max(spearmanr1.pvalue,spearmanr2.pvalue)))
                f.close()
            r1, p1 = stats.pearsonr(mid_samples_score, mid_samples_label)
            spearmanr1 = stats.spearmanr(mid_samples_score, mid_samples_label)
            r2, p2 = stats.pearsonr(mid_samples_score2, mid_samples_label2)
            spearmanr2 = stats.spearmanr(mid_samples_score2, mid_samples_label2)
            print(data + ":  " + "mid overall")
            print("pearson_score: {}   p_value: {}".format((r1+r2)/2, max(p1,p2)))
            print("spearmanr_score: {}   p_value: {}".format((spearmanr1.correlation+spearmanr2.correlation)/2, max(spearmanr1.pvalue,spearmanr2.pvalue)))
            with open(output_dir + "/result.txt", "a") as f:
                f.write("\n{}:\n".format(data + ":  " + "mid overall"))
                f.write("pearson_score: {}   p_value: {}\n".format((r1+r2)/2, max(p1,p2)))
                f.write("spearmanr_score: {}   p_value: {}\n".format((spearmanr1.correlation+spearmanr2.correlation)/2, max(spearmanr1.pvalue,spearmanr2.pvalue)))
                f.close()

            factor_list = list(label[0].keys())
            for factor in factor_list:
                if factor in ["id","rank"]:continue
                f_label = [float(tem[factor]) for tem in label]
                if epoch_name == "Final":
                    plot_hist(score, output_dir + "/score_check", data + "-" + factor + "-score", hists_num=100)
                    plot_hist(f_label, output_dir + "/score_check", data + "-" + factor + "-label", hists_num=100)
                r, p = stats.pearsonr(score, f_label)
                spearmanr = stats.spearmanr(score, f_label)
                if factor == "Overall":
                    print("Add overall!")
                    sp_all += spearmanr.correlation
                    pe_all += r
                if epoch_name == "Final":
                    print(data + ":  " + factor)
                    print("pearson_score: {}   p_value: {}".format(r, p))
                    print("spearmanr_score: {}   p_value: {}".format(spearmanr.correlation, spearmanr.pvalue))
                    if data in check_list:
                        with open(output_dir + "/item_check/" + data + ".csv", "a") as f:
                            writer = csv.writer(f)
                            writer.writerow([factor])
                            writer.writerow(["Sp", "Sp-p", "Pe", "pe-p"])
                            writer.writerow([spearmanr.correlation, spearmanr.pvalue, r, p])
                            f.close()
                with open(output_dir + "/result.txt", "a") as f:
                    f.write("\n{}:\n".format(data + ":  " + factor))
                    f.write("pearson_score: {}   p_value: {}\n".format(r, p))
                    f.write("spearmanr_score: {}   p_value: {}\n".format(spearmanr.correlation, spearmanr.pvalue))
                    f.close()

            # else:
            #     label = [float(tem) for tem in label]
            #     if epoch_name == "Final":
            #         plot_hist(score, output_dir + "/score_check", data  + "-score", hists_num=100)
            #         plot_hist(label, output_dir + "/score_check", data + "-label", hists_num=100)
            #     r, p = stats.pearsonr(score, label)
            #     spearmanr = stats.spearmanr(score, label)
            #     sp_all += spearmanr.correlation
            #     pe_all += r
            #     if epoch_name == "Final":
            #         print(data)
            #         print("pearson_score: {}   p_value: {}".format(r, p))
            #         print("spearmanr_score: {}   p_value: {}".format(spearmanr.correlation, spearmanr.pvalue))
            #         if data in check_list:
            #             with open(output_dir + "/item_check/" + data + ".csv", "a") as f:
            #                 writer = csv.writer(f)
            #                 writer.writerow(["Sp", "Sp-p", "Pe", "pe-p"])
            #                 writer.writerow([spearmanr.correlation, spearmanr.pvalue, r, p])
            #                 f.close()
            #     with open(output_dir + "/result.txt", "a") as f:
            #         f.write("\n{}:\n".format(data))
            #         f.write("pearson_score: {}   p_value: {}\n".format(r, p))
            #         f.write("spearmanr_score: {}   p_value: {}\n".format(spearmanr.correlation, spearmanr.pvalue))
            #         f.close()

            if epoch_name=="Final" and data in check_list:
                label = [float(tem["Overall"]) for tem in label]
                score_= score.copy()
                score_.sort()
                score_dict = {}
                to_length = len(score_)
                for i in range(to_length):
                    score_dict[score_[i]] = (i+1)/to_length
                score_rank = [score_dict[tem] for tem in score]
                label_= label.copy()
                label_.sort()
                to_length = len(label_)
                label_dict = {}
                for i in range(to_length):
                    label_dict[label_[i]] = (i+1)/to_length
                label_rank = [label_dict[tem] for tem in label]
                with open(output_dir + "/item_check/" + data + ".csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([])
                    writer.writerow(["contexts", "response", "references", "label", "score", "label_rank", "score_rank"])
                    for i in range(min(len(score),200)):
                        writer.writerow([contexts[i],responses[i],references[i],label[i],score[i],label_rank[i],score_rank[i]])
                    f.close()

        with open(output_dir + "/result.txt", "a") as f:
            f.write("\nOver_all Average pearson_score:{}\nOver_all Average spearmanr_score:{}\n".format(pe_all/len(eval_list1),sp_all/len(eval_list1)))
            f.close()
        print("Over_all Average pearson_score:{}\nOver_all Average spearmanr_score:{}".format(pe_all/len(eval_list1),sp_all/len(eval_list1)))

    if len(eval_list2)>0:
        # 重新载入模型
        print("Start Eval_list2!!")
        with open(output_dir + "/result.txt", "a") as f:
            f.write("\n\nDSTC10 DATASETS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            f.close()
        sp_all = 0
        pe_all = 0
        sp_oall = 0
        pe_oall = 0
        metric_num = 0
        o_metric_num = 0
        # 开始检测自动评估
        for data in eval_list2:
            data_dir = "dstc10_eval_data/" + data
            dataset = EvalDataset(tokenizer, data_dir, 128)
            data_sampler = RandomSampler(dataset)
            eval_dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=32,
                                         collate_fn=EvalDataCollator(tokenizer, 128))

            start = time.time()
            score, label, contexts, responses, references  = eval_epoch(model, eval_dataloader, device=device)
            factor_list = list(label[0].keys())
            metric_num+=len(factor_list)-2
            score = [0.0 if str(tem) == "nan" else tem for tem in score]

            mid_samples_score,mid_samples_label,polar_samples_score,polar_samples_label,mid_samples_score2,mid_samples_label2,polar_samples_score2,polar_samples_label2 = [],[],[],[],[],[],[],[]
            for i in range(len(score)):
                if label[i]["rank"]<0.25:
                    polar_samples_score.append(score[i])
                    polar_samples_label.append(float(label[i]["appropriateness"]))
                elif label[i]["rank"]>0.75:
                    polar_samples_score2.append(score[i])
                    polar_samples_label2.append(float(label[i]["appropriateness"]))
                elif label[i]["rank"]>0.5:
                    mid_samples_score2.append(score[i])
                    mid_samples_label2.append(float(label[i]["appropriateness"]))
                else:
                    mid_samples_score.append(score[i])
                    mid_samples_label.append(float(label[i]["appropriateness"]))
            r1, p1 = stats.pearsonr(polar_samples_score, polar_samples_label)
            spearmanr1 = stats.spearmanr(polar_samples_score, polar_samples_label)
            r2, p2 = stats.pearsonr(polar_samples_score2, polar_samples_label2)
            spearmanr2 = stats.spearmanr(polar_samples_score2, polar_samples_label2)
            print(data + ":  " + "polar appropriateness")
            print("pearson_score: {}   p_value: {}".format((r1+r2)/2, max(p1,p2)))
            print("spearmanr_score: {}   p_value: {}".format((spearmanr1.correlation+spearmanr2.correlation)/2, max(spearmanr1.pvalue,spearmanr2.pvalue)))
            with open(output_dir + "/result.txt", "a") as f:
                f.write("\n{}:\n".format(data + ":  " + "polar appropriateness"))
                f.write("pearson_score: {}   p_value: {}\n".format((r1+r2)/2, max(p1,p2)))
                f.write("spearmanr_score: {}   p_value: {}\n".format((spearmanr1.correlation+spearmanr2.correlation)/2, max(spearmanr1.pvalue,spearmanr2.pvalue)))
                f.close()
            r1, p1 = stats.pearsonr(mid_samples_score, mid_samples_label)
            spearmanr1 = stats.spearmanr(mid_samples_score, mid_samples_label)
            r2, p2 = stats.pearsonr(mid_samples_score2, mid_samples_label2)
            spearmanr2 = stats.spearmanr(mid_samples_score2, mid_samples_label2)
            print(data + ":  " + "mid appropriateness")
            print("pearson_score: {}   p_value: {}".format((r1+r2)/2, max(p1,p2)))
            print("spearmanr_score: {}   p_value: {}".format((spearmanr1.correlation+spearmanr2.correlation)/2, max(spearmanr1.pvalue,spearmanr2.pvalue)))
            with open(output_dir + "/result.txt", "a") as f:
                f.write("\n{}:\n".format(data + ":  " + "mid appropriateness"))
                f.write("pearson_score: {}   p_value: {}\n".format((r1+r2)/2, max(p1,p2)))
                f.write("spearmanr_score: {}   p_value: {}\n".format((spearmanr1.correlation+spearmanr2.correlation)/2, max(spearmanr1.pvalue,spearmanr2.pvalue)))
                f.close()


            for factor in factor_list:
                if factor in ["id", "rank"]: continue
                f_label = [float(tem[factor]) for tem in label]
                if epoch_name == "Final":
                    plot_hist(score, output_dir + "/score_check", data+"-"+factor+ "-score", hists_num=100)
                    plot_hist(f_label, output_dir + "/score_check", data+"-"+factor + "-label", hists_num=100)
                r, p = stats.pearsonr(score, f_label)
                spearmanr = stats.spearmanr(score, f_label)
                if factor in ["appropriateness"]:
                    sp_oall += spearmanr.correlation
                    pe_oall += r
                    o_metric_num += 1
                sp_all += spearmanr.correlation
                pe_all += r
                if epoch_name == "Final":
                    print(data + ":  " + factor)
                    print("pearson_score: {}   p_value: {}".format(r, p))
                    print("spearmanr_score: {}   p_value: {}".format(spearmanr.correlation, spearmanr.pvalue))
                    if data in check_list:
                        with open(output_dir + "/item_check/" + data + ".csv", "a") as f:
                            writer = csv.writer(f)
                            writer.writerow([factor])
                            writer.writerow(["Sp", "Sp-p", "Pe", "pe-p"])
                            writer.writerow([spearmanr.correlation, spearmanr.pvalue, r, p])
                            f.close()
                with open(output_dir + "/result.txt", "a") as f:
                    f.write("\n{}:\n".format(data + ":  " + factor))
                    f.write("pearson_score: {}   p_value: {}\n".format(r, p))
                    f.write("spearmanr_score: {}   p_value: {}\n".format(spearmanr.correlation, spearmanr.pvalue))
                    f.close()
            if data in check_list:
                score_= score.copy()
                score_.sort()
                score_dict = {}
                to_length = len(score_)
                for i in range(to_length):
                    score_dict[score_[i]] = (i+1)/to_length
                score_rank = [score_dict[tem] for tem in score]
                label_= f_label.copy()
                label_.sort()
                label_set = set(label_)
                label_dict = {}
                to_length = len(label_)
                for i in range(to_length):
                    label_dict[label_[i]] = (i+1)/to_length
                label_rank = [label_dict[tem] for tem in f_label]
                with open(output_dir + "/item_check/" + data + ".csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([])
                    writer.writerow(["contexts", "response", "references", "label", "score", "label_rank", "score_rank"])
                    for i in range(min(len(score),200)):
                        writer.writerow([contexts[i],responses[i],references[i],f_label[i],score[i],label_rank[i],score_rank[i]])
                    f.close()

        with open(output_dir + "/result.txt", "a") as f:
            f.write("\nAll DSTC10 Average pearson_score:{}\nAll DSTC10 Average spearmanr_score:{}\n".format(
                pe_all / metric_num, sp_all / metric_num))
            f.write("\nAll DSTC10 Overall Average pearson_score:{}\nAll DSTC10 Overall Average spearmanr_score:{}\n".format(
                pe_oall / o_metric_num, sp_oall / o_metric_num))
            f.close()
        print("Over_all DSTC10 Average pearson_score:{}\nOver_all DSTC10 Average spearmanr_score:{}".format(pe_all / metric_num,
                                                                                              sp_all / metric_num))
        print("All DSTC10 Overall Average pearson_score:{}\nAll DSTC10 Overall Average spearmanr_score:{}".format(
                pe_oall / o_metric_num, sp_oall / o_metric_num))

