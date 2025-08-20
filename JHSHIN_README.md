## Megatron-LM 설치
pip install -e .[dev]

## 추가 설치 필요한 요소
pip install tensorboard sentencepiece

## Flash atention-3
아래 3,4,5를 하지 않으면 동작하지 않음에 주의.
하고 나서도 TransformerEngine이 나중에 빌드되어야 함.
문제는 flash attention 역시, qk_nope_head_dim + qk_rope_head_dim == v_head_dim 이어야 정상 동작한다.
현재 qk_head_dim=192(=128+64), v_head_dim=128 세팅의 backward를 위한 fused/flash attention이 존재하지 않는다.
forward 구현은 있어도 backward가 없음. 그래서 일단 qk_nope_head_dim=96, qk_rope_head_dim=32로 3:1 구성으로
proportion을 잡는다. 

(1) git clone https://github.com/Dao-AILab/flash-attention.git
(2) cd flash-attention/ && git checkout 27f501d && cd hopper/ && python setup.py install
(3) python_path=`python -c "import site; print(site.getsitepackages()[0])"`
(4) mkdir -p $python_path/flash_attn_3
(5) wget -p $python_path/flash_attn_3 https://raw.githubusercontent.com/dao-ailab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.py


(1) git clone https://github.com/Dao-AILab/flash-attention.git
(2) cd flash-attention/ && git checkout 3ba6f82 && git submodule update --init && cd hopper/ && python setup.py install
(3) python_path=`python -c "import site; print(site.getsitepackages()[0])"`
(4) mkdir -p $python_path/flash_attn_3
(5) cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py


## FP8 연산
--fp8-recipe 'blockwise'
를 하고 싶으면 CUDA 12.9로 환경을 세팅해야 함. 그리고 TransformerEngine, flash-attn 도 모두 이걸로 빌드되어야 함.

## APEX 설치
git clone https://github.com/NVIDIA/apex
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_XENTROPY=1 APEX_NCCL_P2P=1 APEX_DISTRIBUTED_ADAM=1 APEX_NCCL_ALLOCATOR=1 APEX_FAST_LAYER_NORM=1 APEX_FAST_MULTIHEAD_ATTN=1 APEX_FUSED_CONV_BIAS_RELU=1 pip install -v --no-build-isolation .

## TransformerEngine Rebuild
pip uninstall transformer-engine-cu12 transformer-engine-torch transformer-engine
pip install pybind11
NVTE_FRAMEWORK=pytorch pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable

## DeepEP 설치 요령

1. NVSHMEM 설치

간단하게는 
apt install nvshmem nvshmem-12
pip install nvidia-nvshmem-cu12

2. DeepEP 설치
그 전에 CPLUS_INCLUDE_DIR을 추가: 이 환경변수를 바꾸면 APEX, TransformerEngine 빌드에서 오류가 날 수 있다.
export CPLUS_INCLUDE_DIR=$CPLUS_INCLUDE_DIR:/usr/include:/usr/include/x86_64-linux-gnu/

따라서 기존 CPLUS_INCLUDE_DIR 환경변수를 백업하고, deepep 설치 후 다시 원래대로 복원하는 것을 권장.

git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
python setup.py install

## Context Parallism을 위한 코드 수정 필요 사항
see also

https://github.com/NVIDIA/TransformerEngine/pull/1907

## 트러블 슈팅
WorkNCCL Watchdog Timeout이 뜨는 경우 ==> --distributed-timeout-minutes 60 으로 세팅. (일단 30부터 시작)
데이터셋이 커질 수록, 색인하는데 시간이 많이 걸릴 수 있으므로 이 크기를 늘려놓아야 한다.
