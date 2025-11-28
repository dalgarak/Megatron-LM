# Register model

```shell
export PLUGIN_PATH=/path/to/plugin
export PYTHONPATH=$PLUGIN_PATH:$PYTHONPATH
pip install $PLUGIN_PATH
```


# Serving example

```shell
vllm serve path/to/ckpt --served-model-name <MODEL_NAME> --trust-remote-code --tensor-parallel-size 8
```


# Request example

```shell
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_NAME>",
    "prompt": "Albert Einstein (14 March 1879 - 18 April 1955) was"
  }'
```
