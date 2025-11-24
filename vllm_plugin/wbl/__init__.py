def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "WBLForCausalLM",
        "vllm_plugin.wbl.model:WBLForCausalLM",
    )


if __name__ == "__main__":
    import os
    from vllm import LLM, SamplingParams
    register()
    prompts = [
        "Hello world",
    ]
    sampling_params = SamplingParams(temperature=0.1, top_p=0.1)
    llm = LLM(
        model=os.getenv("CKPT_PATH"),
        trust_remote_code=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
