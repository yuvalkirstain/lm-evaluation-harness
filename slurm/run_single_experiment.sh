#!/bin/bash -x

MODEL=$1
TASK=$2
N_SHOT=$3
OUT_PATH=$4
DROPOUT=$5
DECAY=$6
LEARNING_RATE=$7
TRAIN=$8
OPTIMIZER_TYPE=$9
MODEL_TYPE=${10}
LR_SCHEDULER_TYPE=${11}
TRAIN_BATCH_SIZE=${12}
GRAD_ACCU=${13}
SAVE_PRE=${14}
GRADIENT_CLIP_VAL=${15}
MIN_STEP=${16}
SEED=${17}

if [[ $TRAIN == "train" ]]; then
  python main.py \
    --model $MODEL_TYPE \
    --model_args "device=cuda:0,pretrained=$MODEL,dropout=$DROPOUT" \
    --tasks "$TASK" \
    --provide_description \
    --num_fewshot $N_SHOT \
    --no_cache \
    --train_args "weight_decay=$DECAY,learning_rate=$LEARNING_RATE,optimizer_type=$OPTIMIZER_TYPE,model_type=$MODEL_TYPE,lr_scheduler_type=$LR_SCHEDULER_TYPE,per_device_train_batch_size=$TRAIN_BATCH_SIZE,gradient_accumulation_steps=$GRAD_ACCU,save_prefix=$SAVE_PRE,gradient_clip_val=$GRADIENT_CLIP_VAL,min_train_steps=$MIN_STEP" \
    --output_path $OUT_PATH \
    --seed "$SEED"
else
  python main.py \
    --model $MODEL_TYPE \
    --model_args "device=cuda:0,pretrained=$MODEL,dropout=$DROPOUT" \
    --tasks $TASK \
    --provide_description \
    --num_fewshot $N_SHOT \
    --no_cache \
    --output_path $OUT_PATH \
    --seed "$SEED"
fi