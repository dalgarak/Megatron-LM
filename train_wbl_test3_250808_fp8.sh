#!/bin/bash

# Runs Mixtral 8x7B model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# 인자 순서, checkpoint, tokenizer, data path
CHECKPOINT_PATH=$1
TOKENIZER_MODEL=$2
DATA_PATH=$3

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# MLA이므로 position-embedding-type을 none으로 둔다.
#  --no-position-embedding은 deprecated. 쓰지 않는다.
#  --max-position-embeddings 4096 이외의 값을 사용하면 경고가 나온다.
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 4096
    --num-layers 48 
    --position-embedding-type none
    --hidden-size 512 
    --ffn-hidden-size 4096 
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
)

# YaRN을 사용하지 않으므로 mscale을 넣지 않는다.
#    --mscale 1.0
#    --mscale-all-dim 1.0
MLA_ARGS=(
    --multi-latent-attention
    --q-lora-rank 512
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 40
    --normalization RMSNorm
    --rope-type rope
    --rotary-base 10000
    --rotary-base-global 1000000
)

#    --moe-router-pre-softmax false
# FIXME DeepEP 설치 후에 활성화 필요함   --moe-enable-deepep
# --moe-token-dispatcher-type flex 는 deepep와 연동.
MOE_ARGS=(
    --num-experts 128 
    --moe-layer-freq '([0]*3+[1]*45)'
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm 
    --moe-aux-loss-coeff 1e-4
    --moe-router-num-groups 8
    --moe-enable-deepep
    --moe-token-dispatcher-type flex
    --moe-permute-fusion 
    --moe-router-dtype fp32
    --overlap-param-gather
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 99990,8,2
)

# 토큰 수로 bactch size를 조정하도록 변경 필요함
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256 
    --lr 2e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 2.0e-4
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --use-flash-attn
)

FP8_ARGS=(
    --fp8-format 'hybrid'
    --fp8-recipe 'delayed'
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo 'max'
    --fp8-param-gather
)

# 아래 parallel-size의 곱이 world_size와 같아야 한다.
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 2
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng \
    --use-pytorch-profiler 
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"WBL_MOE_100B"}
        --wandb-exp-name ${WANDB_NAME:-"WBL_MOE_TEST1"}
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${FP8_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
