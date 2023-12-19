BASEMODEL=llama-7b
LORAPATH=None
OUTPATH=chat_NCE
TEACHERMODEL=chat_NAT
TEACHERBASE=llama-7b
export WORLD_SIZE=2
wandb online
torchrun --nproc_per_node=8 $ROOTPATH/distill_NCE.py \
    --base_model $BASEMODEL \
    --teacher_model $TEACHERBASE \
    --teacher_checkpoint $TEACHERMODEL \
    --resume_from_checkpoint '' \
    --data_path MATH_ChatGPT_8_t0.5 \
    --output_dir $OUTPATH
