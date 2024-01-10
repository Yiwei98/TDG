import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
from typing import List
from dataset import MATHCHATFILEZFDataset

import fire
import torch
import transformers
from transformers import set_seed,TrainerCallback
set_seed(42)
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from code.peft_NAT import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

class Transpose_State(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.print_step = 200
        self.all_step_count = 0
        kwargs['model'].base_model.model.model.layers[31].self_attn.q_proj.log_alpha = True
        kwargs['model'].base_model.model.model.layers[31].self_attn.v_proj.log_alpha = True
        print("Starting training")

    def on_step_end(self, args, state, control, **kwargs):
        self.all_step_count += 1
        if self.all_step_count % self.print_step == 0:
            alpha_sum_v=0
            cur_alpha=kwargs['model'].base_model.model.model.layers[31].self_attn.v_proj.alpha
            for tt in cur_alpha:
                alpha_sum_v+=tt
            alpha_sum_v/=(1e-5+len(cur_alpha))
            kwargs['model'].base_model.model.model.layers[31].self_attn.v_proj.alpha=[]

            alpha_sum_q=0
            cur_alpha=kwargs['model'].base_model.model.model.layers[31].self_attn.q_proj.alpha
            for tt in cur_alpha:
                alpha_sum_q+=tt
            alpha_sum_q/=(1e-5+len(cur_alpha))
            print(f"Step {self.all_step_count}, Alpha_v {alpha_sum_v},Alpha_q {alpha_sum_q}")
            kwargs['model'].base_model.model.model.layers[31].self_attn.q_proj.alpha=[]
            kwargs['model'].base_model.pos_loss = 0
            kwargs['model'].base_model.neg_loss = 0


def train(
    # model/data params
    stage=2,
    base_model: str = "llama-7b-hf-converted",  # the only required argument
    data_path: str = "MATH_ChatGPT_8_t0.5",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 16,
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
    resume_from_checkpoint: str = None,#"neg_ckpt",#None,  # either training checkpoint or final adapter
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
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

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if stage==1:
        cur_mode = 'neg'
    elif stage==2:
        cur_mode= 'mix'
    else:
        cur_mode= 'pos'
    model = get_peft_model(model, config,cur_mode=cur_mode)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if cur_mode=='mix':
        for i in range(32):
            model.base_model.model.model.layers[i].self_attn.q_proj.lora_B_neg.weight.requires_grad = False
            model.base_model.model.model.layers[i].self_attn.q_proj.lora_A_neg.weight.requires_grad = False
            model.base_model.model.model.layers[i].self_attn.v_proj.lora_B_neg.weight.requires_grad = False
            model.base_model.model.model.layers[i].self_attn.v_proj.lora_A_neg.weight.requires_grad = False

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        pass
    else:
        train_data = MATHCHATFILEZFDataset(tokenizer, data_path + '/*/*',stage=stage)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=200,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=500,
            output_dir=output_dir,
            save_total_limit=None,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        #data_collator=transformers.DataCollatorForSeq2Seq(
        #    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        #),
        callbacks=[Transpose_State()],
        data_collator=None,
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    set_seed(42)
    trainer.train()

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")



if __name__ == "__main__":
    fire.Fire(train)
