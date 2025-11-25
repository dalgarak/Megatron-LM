def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "WBLForCausalLM",
        "vllm_plugin.wbl.model:WBLForCausalLM",
    )
