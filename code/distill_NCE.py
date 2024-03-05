import os
import sys
from typing import List
from MATH.dataset import MATHCHATFILEDataset
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import TrainerCallback


assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaPreTrainedModel
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from peft_NAT import (
    prepare_model_for_int8_training as prepare_model_for_int8_training2,
    get_peft_model as get_peft_model2,
    get_peft_model_state_dict as get_peft_model_state_dict2,
    set_peft_model_state_dict as set_peft_model_state_dict2,
)

class Transpose_State(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.print_step = 100
        self.all_step_count = 0
        kwargs['model'].teacher_model.base_model.model.model.layers[31].self_attn.q_proj.log_alpha=True
        kwargs['model'].teacher_model.base_model.model.model.layers[31].self_attn.v_proj.log_alpha = True
        print("Starting training")

    def on_step_end(self, args, state, control, **kwargs):
        self.all_step_count += 1
        kwargs['train_dataloader'].dataset.type=-kwargs['train_dataloader'].dataset.type
        cur_mode = 'pos' if kwargs['model'].teacher_model.base_model.cur_mode=="neg" else "neg"
        kwargs['model'].teacher_model.base_model.cur_mode = cur_mode
        for i in range(32):
            try:
                kwargs['model'].teacher_model.base_model.model.model.layers[i].self_attn.v_proj.cur_mode = cur_mode
                kwargs['model'].teacher_model.base_model.model.model.layers[i].self_attn.q_proj.cur_mode = cur_mode
            except:
                cur_mode=cur_mode
        if self.all_step_count % self.print_step == 0:
            alpha_sum_v=0
            cur_alpha=kwargs['model'].teacher_model.base_model.model.model.layers[31].self_attn.v_proj.alpha
            for tt in cur_alpha:
                alpha_sum_v+=tt
            alpha_sum_v/=(1e-5+len(cur_alpha))
            kwargs['model'].base_model.model.model.layers[31].self_attn.v_proj.alpha=[]

            alpha_sum_q=0
            cur_alpha=kwargs['model'].teacher_model.base_model.model.model.layers[31].self_attn.q_proj.alpha
            for tt in cur_alpha:
                alpha_sum_q+=tt
            alpha_sum_q/=(1e-5+len(cur_alpha))
            print(f"Step {self.all_step_count}, Average pos Loss: {kwargs['model'].base_model.pos_loss/(self.print_step)}"
                  f", Average neg Loss: {kwargs['model'].base_model.neg_loss/(self.print_step)}, Alpha_v {alpha_sum_v},Alpha_q {alpha_sum_q}")
            kwargs['model'].teacher_model.base_model.model.model.layers[31].self_attn.q_proj.alpha=[]
            kwargs['model'].teacher_model.base_model.pos_loss = 0
            kwargs['model'].teacher_model.base_model.neg_loss = 0

class Distill(nn.Module):
    def __init__(self, teacher_model, student_model):
        super().__init__()
        self.teacher_model = teacher_model
        for p in self.parameters():
            p.requires_grad = False
        self.student_model = student_model
        self.T = 1.0
        self.alpha = 0.5

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ques=None,
        **kwargs,
    ):
        student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
        )
        self.teacher_model.eval()
        cur_mode = 'mix'
        for i in range(32):
            self.teacher_model.base_model.model.model.layers[i].self_attn.v_proj.cur_mode = cur_mode
            self.teacher_model.base_model.model.model.layers[i].self_attn.q_proj.cur_mode = cur_mode
        with torch.no_grad():
            pos_teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
            )
        cur_mode = 'neg'
        for i in range(32):
            self.teacher_model.base_model.model.model.layers[i].self_attn.v_proj.cur_mode = cur_mode
            self.teacher_model.base_model.model.model.layers[i].self_attn.q_proj.cur_mode = cur_mode
        with torch.no_grad():
            neg_teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
            )
        loss_ce = student_outputs["loss"]
        pred = F.log_softmax(student_outputs["logits"] / self.T, dim=-1)
        pos_teacher_pred = pos_teacher_outputs["logits"]
        neg_teacher_pred = neg_teacher_outputs["logits"]
        pos_log_outputs = F.log_softmax(pos_teacher_pred / self.T, dim=-1)
        pos_outputs = F.softmax(pos_teacher_pred / self.T, dim=-1) + 10 ** (-7)
        neg_outputs = F.softmax(neg_teacher_pred / self.T, dim=-1) + 10 ** (-7)
        beta = torch.tanh((F.kl_div(pos_log_outputs, neg_outputs, reduction='none') * attention_mask.unsqueeze(-1)).sum(-1).sum(-1) / attention_mask.sum(-1))
        print(ques)

        loss_kd = - self.T * self.T * pred * pos_outputs * (0.5 +  beta.unsqueeze(-1).unsqueeze(-1))
        loss_kd = (loss_kd * attention_mask.unsqueeze(-1)).sum() / attention_mask.sum()
        student_outputs["loss"] = (1 - self.alpha) * loss_ce + self.alpha * loss_kd
        return student_outputs

def load_model(base_model, lora_path, config, device_map):

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map=device_map,
    )

    model = prepare_model_for_int8_training(model)

    model = get_peft_model(model, config)
    
    if lora_path:
        checkpoint_name = os.path.join(
            lora_path, "adapter_model.bin"
        )
        print(f"Loading lora parameters from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name,map_location="cuda:0")
        model = set_peft_model_state_dict(model, adapters_weights)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    return model


def load_teacher_model(base_model, lora_path, config, device_map):
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map=device_map,
    )

    model = prepare_model_for_int8_training2(model)

    model = get_peft_model2(model, config)

    if lora_path:
        checkpoint_name = os.path.join(
            lora_path, "adapter_model.bin"
        )
        print(f"Loading lora parameters from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name,map_location="cuda:0")
        model = set_peft_model_state_dict2(model, adapters_weights)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    return model


def train(
    # model/data params
    base_model: str = "",  # required argument
    teacher_checkpoint: str = "",  # required argument
    teacher_model: str = "",
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 1,
    micro_batch_size: int = 1,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 0,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve,
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"teacher_checkpoint: {teacher_checkpoint}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    if teacher_model == '':
        teacher_model = base_model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    teacher_model = load_teacher_model(teacher_model, teacher_checkpoint, config, device_map)
    student_model = load_model(base_model, resume_from_checkpoint, config, device_map)

    resume_from_checkpoint = False

    if val_set_size > 0:
        pass
    else:
        train_data = MATHCHATFILEDataset(tokenizer, data_path + '/*/*')
        val_data = None

    student_model.config.use_cache = False
    teacher_model.config.use_cache = False

    
    old_state_dict = student_model.state_dict
    teacher_old_state_dict = teacher_model.state_dict
    student_model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(student_model, type(student_model))
    teacher_model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict2(self, teacher_old_state_dict())
    ).__get__(teacher_model, type(teacher_model))
    
    
    model = Distill(teacher_model, student_model)
    '''
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    '''
    print(gradient_accumulation_steps)
    gradient_accumulation_steps=1
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=1000,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=400,
            output_dir=output_dir,
            save_total_limit=None,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        #data_collator=transformers.DataCollatorForSeq2Seq(
        #    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        #),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    student_model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
