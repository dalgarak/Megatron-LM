if __name__ == "__main__":
    import os
    from vllm import LLM, SamplingParams
    prompts = [os.getenv("PROMPT")]
    sampling_params = SamplingParams(temperature=0., max_tokens=100)
    llm = LLM(
        model=os.getenv("CKPT_PATH"),
        trust_remote_code=True,
        pipeline_parallel_size=8,
        enforce_eager=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
