#export CUDA_VISIBLE_DEVICES=0
BASEMODEL=llama-7b
LORAPATH=chat_NCE
OUTPATH=sint
python generate_batch_adaptive.py \
    --base_model  $BASEMODEL \
    --lora_weights $LORAPATH \
    --data_path MATH/test/algebra \
    --load_8bit
