BASEMODEL=llama-7b
LORAPATH=chat_neg
OUTPATH=chat_NAT
wandb online
export WORLD_SIZE=2
#cd $ROOTPATH
torchrun --nproc_per_node=8 $ROOTPATH/finetune.py \
    --base_model $BASEMODEL \
    --resume_from_checkpoint $LORAPATH \
    --data_path MATH_ChatGPT_8_t0.5 \
    --output_dir $OUTPATH \
    --stage 2
