import json
import operator
import math
import random
all_cate = ["counting_and_probability","intermediate_algebra","number_theory","precalculus","prealgebra","geometry","algebra"]
datas = [0,1,2,3,4,5,6]
datanames = [all_cate[i] for i in datas]
file_names = ['best_gpt4/test_{}_n16_1'.format(dataname) for dataname in datanames]
ckpt_name1 = "gpt4_full_ckpt"
ckpt_name2 = "model_4"
scores1 = {0:0,1:0,2:0,3:0,4:0}
scores2 = {0:0,1:0,2:0,3:0,4:0}
for file_name in file_names:
    try:
        with open(file_name+"-from{}-{}.json".format(ckpt_name1,ckpt_name2),"r")as f:
            data = json.load(f)
            f.close()
    except:
        continue
    for flag in [0,1,2,3]:
        right_num = 0
        all_num=0
        winnum=0
        lossnum=0
        tienum=0
        for it in range(1):
            for tem in data:
                all_num+=1
                ans = tem[0][1]
                mean_score = 0
                mean_fm = 0
                for t in tem:
                    if t[2] is not None:
                        mean_score+=t[2][0]
                        mean_fm+=1
                if mean_fm == 0:
                    mean_score = 1
                else:
                    mean_score /= mean_fm

                pre_dict = {}
                random.shuffle(tem)
                for t in tem:
                    if t[2] is None:
                        t[2] = [mean_score]

                    if t[0] in pre_dict.keys():
                        if flag==0:
                            pre_dict[t[0]] +=1#+ t[2][0]#1/(math.pow(t[2][0],5)+1)
                        elif flag==1:
                            pre_dict[t[0]] += 1   + 2*t[2][0]
                        elif flag==2:
                            pre_dict[t[0]] +=  1+1*t[2][0]
                        elif flag==3:
                            pre_dict[t[0]] += t[2][0]
                    else:
                        if flag == 0:
                            pre_dict[t[0]] = 1# + t[2][0]  # 1/(math.pow(t[2][0],5)+1)
                        elif flag == 1:
                            pre_dict[t[0]] = 1 + 2*t[2][0]
                        elif flag==2:
                            pre_dict[t[0]] = 1+1*t[2][0]
                        elif flag==3:
                            pre_dict[t[0]] = t[2][0]

                res1 = sorted(pre_dict.items(), key=operator.itemgetter(1), reverse=True)
                right_tems = set()
                right_tems.add(res1[0][0])
                #print("!!!!!!!!!")
                #print(right_tems)
                #print(res1)
                for tem in res1[1:]:
                    #print(tem)
                    if tem[1] == res1[0][1]:
                        right_tems.add(tem[0])
                    else:
                        break
                #print(right_tems)
                if ans in right_tems:
                    right_num+=1/len(right_tems)
                    #print(1/len(right_tems)) 

        # print("Win num {}".format(winnum))
        # print("Loss num {}".format(lossnum))
        # print("Tie num {}".format(tienum))
        print(file_name)
        scores1[flag]+=right_num
        scores2[flag]+=all_num
        #print("vote-type:{}".format(flag))
        #print("right_num:{}".format(right_num))
        print("Acc: {}".format(right_num/all_num))
scores = [scores1[i]/(1e-5+scores2[i]) for i in range(5)]


for i in range(len(scores)):
    print("type:{}   avg score:{}".format(i,scores[i]))
