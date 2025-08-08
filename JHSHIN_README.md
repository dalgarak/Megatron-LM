## DeepEP 설치 요령

1. NVSHMEM 설치

간단하게는 
pip install nvidia-nvshmem-cu12

2. DeepEP 설치
그 전에 CPLUS_INCLUDE_DIR을 추가
export CPLUS_INCLUDE_DIR=$CPLUS_INCLUDE_DIR:/usr/include:/usr/include/x86_64-linux-gnu/

git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
python setup.py install

## TransformerEngine Rebuild
pip install pybind11
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable

## FP8 연산
--fp8-recipe 'blockwise'
를 하고 싶으면 CUDA 12.9로 환경을 세팅해야 함. 그리고 TransformerEngine, flash-attn 도 모두 이걸로 빌드되어야 함.

