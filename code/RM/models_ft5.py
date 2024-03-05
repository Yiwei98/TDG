import math
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer, T5EncoderModel

# from Transformers.src.transformers.models.t5.modeling_t5 import
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))



def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.pooler_type
    cls.pooler = Pooler(cls.pooler_type)
    if cls.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    special_tokens_mask=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    if special_tokens_mask is not None:
        special_tokens_mask = special_tokens_mask.view((-1, special_tokens_mask.size(-1)))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids=input_ids,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    if special_tokens_mask is not None:
        pooler_output = cls.pooler(attention_mask, outputs)
    else:
        pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    loss = loss_fct(cos_sim, labels)

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return {"loss":loss,
        "logits":cos_sim,
        "hidden_states":outputs.hidden_states,
        "attentions":outputs.attentions}


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    return_dict=None,
    special_tokens_mask=None,
):

    return_dict = True

    outputs = encoder(
        input_ids,
    )


    pooler_output = cls.pooler(special_tokens_mask, outputs)


    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class FinalPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense1_activation = nn.ELU()
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense2_activation = nn.ELU()
        self.pool = nn.Linear(hidden_size, 1)
        self.pool_activation = nn.Sigmoid()

    def forward(self, hidden_state):
        # 加一个residual，可以看一下有没有效果
        output = self.dense1_activation(self.dense1(hidden_state))+hidden_state
        output = self.dense2_activation(self.dense2(output))+output
        output = self.pool_activation(self.pool(output))
        return output


class BertForCL(nn.Module):
    # _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self,  gap=0.3, pooler_type = "avg_top2", temp=0.05,hs=1024,model_name = "google/flan-t5-large"):
        super(BertForCL, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.output = FinalPooler(hs)
        self.pooler_type = pooler_type
        self.temp = temp
        self.gap = gap
        self.kk = 1
        self.pownum1 = 7
        self.pownum2 = 3
        self.alpha = 0.01
        self.is_eval = False
        self.is_ft = False
        self.CEloss = nn.CrossEntropyLoss()
        self.mark_embedding = nn.Embedding(2, hs)
        self.pooler = Pooler("avg")

    def forward(self,
        input_ids=None,
        special_tokens_mask=None,
        labels=None,
        return_dict=None,
        weights=None,
    ):
        sen_results = sentemb_forward(self, self.encoder,
                input_ids=input_ids,
                return_dict=return_dict,
                special_tokens_mask=special_tokens_mask,
            )
        score = self.output(sen_results.pooler_output)
        if self.is_ft:
            loss = (labels.flatten() - score.flatten()).pow(2).mean()
            return loss
        if not self.is_eval:
            score1 = score.view(-1, 2)
            loss1 = (torch.pow(1-score1[:, 0], self.pownum1) + torch.pow(score1[:, 1], self.pownum1))*1#weights
            acc = (score1[:, 0]>score1[:, 1]).sum()/len(score1[:, 1])
            loss = loss1
            l_1 = (loss*labels).sum()/(labels.sum()+1e-6)
            l_2 = (loss * (1-labels)).sum() / ((1-labels).sum()+1e-6)
            return loss.mean(),l_1,l_2,acc
        else:
            return [tem.cpu().item() for tem in score]

    def save_checkpoint(self, to_dir):
        state_dict = {t: v for t, v in self.state_dict().items()}
        torch.save(state_dict, to_dir)
        print("successfully save model to {} !!".format(to_dir))

    def load_checkpoint(self, from_dir, device):
        checkpoint = torch.load(from_dir, map_location=lambda storage, loc: storage.cuda(device))
        self.load_state_dict(checkpoint, strict=False)
        print("successfully params from: {} !!".format(from_dir))

    def to_eval(self, eval=False):
        self.is_eval = eval

