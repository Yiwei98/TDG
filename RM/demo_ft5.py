import json
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from data_ft5 import SimcseDataCollator,SimDataset,RobustDataCollator
import random
import numpy.random
import torch.optim.lr_scheduler as lr_scheduler
from transformers.optimization import AdamW
from transformers import BertTokenizer,AutoTokenizer
from train_func_ft5 import *
import math
from models_ft5 import BertForCL
from utils import *


for batch_size in [1]:
    for seed in [0]:
        # 参数区
        resume_dir = "MATH/flan-t5-xl_datamodel-ChatGPT_seed1_bs1"
        lr = 2e-5 #5e-5
        lr_decay = 0.02
        epoch = 5
        # seed = 2
        devicenum = 7
        # batch_size = 3
        max_length = 512
        hard_neg_bili = 0 #效果不明显，可设为0
        hard_pos_bili = 1 #设为1最好
        train_num = -1  #-1为使用所有训练数据训练，如果想快速实践改后的模型有没有明显问题，可以设置为3000快速跑一下试试
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # eval集组成，如果未来信息可见，则为"c_r_c"，若不可见则为"c_r", "c_r_c"经测试无效
        bert_name = "google/flan-t5-xxl"#"bert-large-uncased"
        data_model = "gpt4_train"#"ChatGPT" #"GPT4"
        pre_train_model_name = "flan-t5-xl"#"google/flan-t5-large"#"flan-t5-xl"#"bert-large-uncased"
        hs = 2048#1024#2048
        pre_train_model_dir = pre_train_model_name
        device = torch.device("cuda" + ":" + str(devicenum) if torch.cuda.is_available() else "cpu")
        to_train = True
        train_list = ["daily"]
        # train_list = ["persona", "empa", "topical"]
        eval_list = ["fed-addrank.json", "engage-addrank.json", "convai2_grade_bert_ranker-addrank.json",
                    "convai2_grade_dialogGPT-addrank.json",
                    "topicalchat_usr-addrank.json", "personachat_usr-addrank.json",
                    "dailydialog_grade_transformer_generator-addrank.json",
                    "dailydialog_grade_transformer_ranker-addrank.json",
                    "empatheticdialogues_grade_transformer_generator-addrank.json",
                    "empatheticdialogues_grade_transformer_ranker-addrank.json",
                    "convai2_grade_transformer_generator-addrank.json", "convai2_grade_transformer_ranker-addrank.json",
                    "dstc6-addrank.json"]
        for train_data in train_list:
            eval_dstc10_list = ["jsalt_eval-addrank.json", "esl_eval-addrank.json", "ncm_eval-addrank.json",
                                "dstc10-persona_clean_eval-addrank.json",
                                "dstc10-topical_clean_eval-addrank.json"]
            check_list = ["dailydialog_grade_transformer_generator-addrank.json",
                        "dailydialog_grade_transformer_ranker-addrank.json",
                        "engage-addrank.json", "jsalt_eval-addrank.json", "esl_eval-addrank.json", "ncm_eval-addrank.json"]
            save_model = True
            # output_dir = "result/" + f"L{pow1}+L{pow2}-{train_data}{train_num}_4.5.1_lr{lr}_decay_{lr_decay}_negbili_{hard_neg_bili}_posbili_{hard_pos_bili}_epoch_{epoch}_seed{seed}"
            output_dir = "MATH/gpt4_full_ckpt"
            #output_dir = "MATH/{}_datamodel-{}_seed{}_bs{}".format(pre_train_model_name,data_model,seed,batch_size)
            train_data_dir = "raw_data/{}_train.txt".format(train_data)
            valid_data_dir = "raw_data/{}_valid.txt".format(train_data)
            # train_data_dir = "data_for_RM.json"
            # prepare_output_dir
            if os.path.exists(output_dir) is False:
                os.makedirs(output_dir)
            if os.path.exists(output_dir + "/item_check") is False:
                os.makedirs(output_dir + "/item_check")
            if os.path.exists(output_dir + "/score_check") is False:
                os.makedirs(output_dir + "/score_check")

            # prepare data:

            tokenizer = AutoTokenizer.from_pretrained(pre_train_model_name)
            train_dataset = SimDataset(tokenizer, train_data_dir, max_length,data_model, sub_set="train",n_obs=train_num)
            train_data_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_data_sampler, batch_size=batch_size,
                                        collate_fn=SimcseDataCollator(tokenizer, max_length,
                                                                        pos_bili=hard_pos_bili))
            valid_dataset = SimDataset(tokenizer, valid_data_dir, max_length,data_model,sub_set="valid")
            valid_data_sampler = RandomSampler(valid_dataset)
            valid_dataloader = DataLoader(valid_dataset, sampler=valid_data_sampler, batch_size=batch_size,
                                        collate_fn=SimcseDataCollator(tokenizer, max_length,
                                                                        pos_bili=hard_pos_bili))

            # prepare model:
            model = BertForCL(hs=hs,model_name = pre_train_model_name)
            model = model.to(device)
            if resume_dir is not None:
                model.load_checkpoint(resume_dir + "/model.pth", device)
            total = sum([param.nelement() for param in model.parameters()])
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.pownum1 = 3
            model.pownum2 = 7
            print("Number of total parameters: %.2fM" % (total / 1e6))
            print("Number of trainable parameters: %.2fM" % (trainable_num / 1e6))

            # prepare lr_scheduler and optimizer:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in filter(lambda np: np[1].requires_grad, model.named_parameters()) if
                            not any(nd in n for nd in no_decay)],
                    "weight_decay": 1e-5,
                },
                {"params": [p for n, p in filter(lambda np: np[1].requires_grad, model.named_parameters()) if
                            any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
            lf = lambda x: ((1 + math.cos(x * math.pi / epoch)) / 2) * (1 - lr_decay) + lr_decay
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 动态学习率

            # start to train!
            if to_train:
                valid_losses = []
                max_acc = 0
                for epoch_i in range(1, epoch + 1):
                    print('[ Epoch', epoch_i, ']')

                    start = time.time()
                    train_loss,l1,l2,acc = train_epoch(
                        model, train_dataloader, optimizer, device=device)
                    #model.update_density()
                    scheduler.step()
                    print("Epoch:{}    ACC:{}  train_loss:{}  L1:{}  L2:{}   cost_time:{}".format(epoch_i,acc, train_loss,l1,l2, time.time() - start))

                    if epoch_i % 1 == 0:
                        start = time.time()
                        valid_loss ,l1,l2,acc= valid_epoch(model, valid_dataloader, device=device)

                        print("Validation  Epoch:{}    ACC:{}  valid_loss:{}  L1:{}  L2:{}    cost_time:{}".format(epoch_i,acc, valid_loss,l1,l2, time.time() - start))

                        if max_acc < acc:
                            print("New best Acc:{}".format(acc))
                        if save_model and (acc > max_acc):
                            model.save_checkpoint(output_dir + "/model.pth")
                        model.save_checkpoint(output_dir + "/model_cur.pth")
                        if epoch_i%1 == 0:
                            model.save_checkpoint(output_dir + "/model_{}.pth".format(epoch_i))
                        max_acc = max(max_acc, acc)
                        print("Current best Acc:{}".format(min(max_acc, 100)))
                        valid_losses += [valid_loss]
                        with open(output_dir + "/result.txt", "a") as f:
                            # f.write("train params:\n")
                            # f.write("lr={}\nepoch={}\npre-train model={}\n".format(lr, epoch, pre_train_model_dir))
                            f.write("\ncur loss:{}\n".format(valid_loss))
                    # if epoch_i % 10 == 0 and epoch_i != epoch:
                    #     eval_func(model, eval_list, eval_dstc10_list, check_list, device, output_dir, tokenizer,
                    #             reload_best_model=False, epoch_name=str(epoch_i))
                with open(output_dir + "/result.txt", "a") as f:
                    f.write("train params:\n")
                    f.write("lr={}\nepoch={}\npre-train model={}\n".format(lr, epoch, pre_train_model_dir))
                    f.write("\nbest acc:{}\n".format(max_acc))

            # model.load_checkpoint(output_dir + "/model.pth", device)
            # 用第40个epoch的模型最终评测
            # eval_func(model, eval_list, eval_dstc10_list, check_list, device, output_dir, tokenizer, False)
        # # 测试模型鲁棒性，需要robust_data.txt，该数据集由dstc10前三个数据集组成
        # valid_dataset = SimDataset(tokenizer, "dstc10_eval_data/robust_data.txt", max_length)
        # neg_sampler.bili = 0
        # robust_dataloader = DataLoader(valid_dataset, sampler=valid_data_sampler, batch_size=batch_size,
        #                             collate_fn=RobustDataCollator(tokenizer, max_length, neg_sampler=neg_sampler,
        #                                                             pos_bili=hard_pos_bili))
        # eval_robust(model, robust_dataloader, device, output_dir)
        # with open(output_dir + "/density.txt", "w") as f:
        #     for i in range(len(model.density_sum)):
        #         f.write(str(model.density_sum[i]) + "\n")
        #     f.close()
        # with open(output_dir + "/result.txt", "a") as f:
        #     f.write("\ntrain score bias of average:{}\n".format(model.get_density_bias()))
