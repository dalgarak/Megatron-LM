#!/bin/bash

# Runs MLA+MoE+Peri-LN with 48 layers.

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_FLASH_ATTN=1
export UB_SKIPMC=1
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
#  swiglu를 꺼도 gated_linear_unit은 True로 세팅되어 있음.
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024 
    --max-position-embeddings 4096
    --num-layers 8
    --position-embedding-type none
    --hidden-size 3072
    --ffn-hidden-size 18432 
    --num-attention-heads 32
    --init-method-std 0.0134
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
)

# YaRN을 사용하지 않으므로 mscale을 넣지 않는다.
#    --mscale 1.0
#    --mscale-all-dim 1.0
# rotary-scaling-factor는 나중에(사전학습 3단계) 8.0 으로 바꾼다 (for PI)
MLA_ARGS=(
    --multi-latent-attention
    --q-lora-rank 1536 
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 1.0
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
#    dispatcher type에서 allgather는 EP가 1일 때만 사용하고, 2 이상이면 alltoall을 사용할 것.
#    group-limited routing 등 속도 개선을 위한 작업이 필요함
#    --moe-router-group-topk 4 \
#    --moe-router-num-groups 4 \
MOE_ARGS=(
    --num-experts 128 \
    --moe-layer-freq '([0]*3+[1]*5)' \
    --moe-ffn-hidden-size 2048	\
    --moe-shared-expert-intermediate-size 2048 \
    --moe-shared-expert-overlap \
    --moe-router-padding-for-fp8 \
    --moe-router-load-balancing-type aux_loss \
    --moe-router-topk 8 \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-score-function 'sigmoid' \
    --moe-token-dispatcher-type alltoall \
    --moe-permute-fusion \
    --moe-router-dtype fp32 \
    --overlap-param-gather \
    --overlap-grad-reduce
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path $DATA_PATH \
    --split 99990,8,2
)

# global-batch-size * sequence length = Effective batch size임
# --init-model-with-meta-device
#    --data-parallel-sharding-strategy 'optim_grads_params'
#    --no-gradient-accumulation-fusion
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --lr 2e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style constant
    --min-lr 2.0e-4
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --grad-reduce-in-bf16
    --use-flash-attn
    --attention-backend 'flash'
    --seed 20250809
)

#    --fp8-recipe 'delayed'
#    blockwise scaling에는 CUDA 12.9 세팅이 필요.
#    --fp8-recipe 'blockwise'
#    blockwise가 좀 더 빠르긴 한데, loss 흔들림이 소형 모델 (5B) 테스트에서 발견되었음.
FP8_ARGS=(
    --fp8-format 'hybrid'
    --fp8-recipe 'delayed'
    --fp8-amax-history-len 1024 
    --fp8-amax-compute-algo 'max'
    --fp8-param-gather
    --num-layers-at-start-in-bf16 1
    --num-layers-at-end-in-bf16 1
)

# 아래 parallel-size의 곱이 world_size와 같아야 한다.
# --cp-comm-type 'a2a+p2p' 를 하는 경우, --hierarchical-context-parallel-sizes [숫자] 설정을 필요.
# MoE-specific TP size를 세팅하기 위해서는 --expert-tensor-parallel-size [숫자]로 세팅할 것
# tensor-model-parallel-size가 2 이상일 경우 --tp-comm-overlap을 사용.
# tensor-model-parallel-size가 2 미만일 경우 --sequence-parallel은 disable 됨에 주의.
# context-parallel과 expert-parallel 수는 동일하게 맞춰도 된다 (배수로 적용되지 X)
# Sliding Window를 사용하는 현재 메커니즘에서는 context parallel이 동작하지 않음.
# 어떤 부분이 더 이득인지 살펴볼 필요가 있음
# --context-parallel-size 1
# --tp-comm-overlap
#    --optimizer-cpu-offload
#    --use-precision-aware-optimizer
#    --recompute-activations
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 2
    --use-distributed-optimizer
    --sequence-parallel
    --cp-comm-type 'p2p'
)

# 속도차이?   --use-pytorch-profiler 차이는 크게 없음을 확인 
# save-interval은 0.5 day / 1 day로 잡아도 되고, retain-interval은 그것의 4~7배로 잡는 것을 추천 (4일~1주일 분량)
LOGGING_ARGS=(
    --log-interval 1 \
    --eval-interval 500 \
    --save-interval 100 \
    --save-retain-interval 300 \
    --ckpt-format torch_dist \
    --eval-iters 10 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --tensorboard-log-interval 1 \
    --tensorboard-queue-size 100 \
    --use-pytorch-profiler \
    --log-validation-ppl-to-tensorboard \
    --log-params-norm \
    --log-throughput \
    --log-progress \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"WBL_MOE_100B"}
        --wandb-exp-name ${WANDB_NAME:-"WBL_MOE_TEST1"}
    )
fi


#WORLD_SIZE=8 python -u report_theoretical_memory.py \
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${FP8_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
