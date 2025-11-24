# Serving example

```shell
export PLUGIN_ROOT_PATH=<path_to_plugin_root>
export PYTHONPATH=$PLUGIN_ROOT_PATH:$PYTHONPATH
cd $PLUGIN_ROOT_PATH/vllm_plugin
pip install -e .
vllm serve path/to/ckpt --served-model-name wbl_model --trust-remote-code --tensor-parallel-size 8
```


# Request example

```shell
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": wbl_model,
    "prompt": "Hello world"
  }'
```
