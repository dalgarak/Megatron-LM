def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "WBLForCausalLM",
        "wbl.model:WBLForCausalLM",
    )
