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
    --hidden-size 4096 
    --ffn-hidden-size 12288
    --num-attention-heads 64 
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
    --q-lora-rank 1024 
    --kv-lora-rank 1024
    --qk-head-dim 96 
    --qk-pos-emb-head-dim 32
    --v-head-dim 128
    --rotary-scaling-factor 40
    --normalization RMSNorm
    --rope-type rope
    --rotary-base 10000
    --rotary-base-global 1000000
)

#    --moe-router-pre-softmax false
# FIXME DeepEP 설치 후에 활성화 필요함   --moe-enable-deepep
# --moe-token-dispatcher-type flex 는 deepep와 연동. ep <=8 인 경우에는 allgather 또는 alltoall이 더 유리함
# deepep가 유리한 포인트는 cross-node expert parallism 일 경우기 때문. 100B 이상 세팅에서는 고려해야 함.
#    --moe-enable-deepep
MOE_ARGS=(
    --num-experts 128 
    --moe-layer-freq '([0]*3+[1]*45)'
    --moe-ffn-hidden-size 12288 
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm 
    --moe-aux-loss-coeff 1e-4
    --moe-router-num-groups 8
    --moe-token-dispatcher-type allgather 
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

# global-batch-size * sequence length = Effective batch size임
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1024 
    --lr 2e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style constant
    --min-lr 2.0e-4
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --use-flash-attn
    --seed 20250809
    --recompute-activations
)

#    --fp8-recipe 'delayed'
#    blockwise scaling에는 CUDA 12.9 세팅이 필요.
#    --fp8-recipe 'blockwise'
FP8_ARGS=(
    --fp8-format 'hybrid'
    --fp8-recipe 'blockwise'
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo 'max'
    --fp8-param-gather
)

# 아래 parallel-size의 곱이 world_size와 같아야 한다.
# --cp-comm-type 'a2a+p2p' 를 하는 경우, --hierarchical-context-parallel-sizes [숫자] 설정을 필요.
# MoE-specific TP size를 세팅하기 위해서는 --expert-tensor-parallel-size [숫자]로 세팅할 것
# tensor-model-parallel-size가 2 이상일 경우 --tp-comm-overlap을 사용.
# tensor-model-parallel-size가 2 미만일 경우 --sequence-parallel은 disable 됨에 주의.
# 추가 문제점: 일단 현재 context-parallel이 FP8 + MLA에서는 작동하지 않음. TransformerEngine 수정 필요.
# see https://github.com/NVIDIA/TransformerEngine/pull/1896
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 2
    --context-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
    --tp-comm-overlap
    --cp-comm-type 'p2p'
)

# 속도차이?   --use-pytorch-profiler 차이는 크게 없음을 확인 
# save-interval은 0.5 day / 1 day로 잡아도 되고, retain-interval은 그것의 4~7배로 잡는 것을 추천 (4일~1주일 분량)
LOGGING_ARGS=(
    --log-interval 1 \
    --eval-interval 500 \
    --save-interval 500 \
    --save-retain-interval 3000 \
    --ckpt-format torch_dist \
    --eval-iters 10 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-log-interval 1 \
    --tensorboard-queue-size 100 \
    --log-params-norm \
    --log-throughput \
    --log-progress \
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
