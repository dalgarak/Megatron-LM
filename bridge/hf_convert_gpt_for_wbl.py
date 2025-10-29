import os
import shutil
import json
import torch
from megatron.core import parallel_state


BACKEND = "nccl"


def provider_to_config(model, save_directory):
    pass


def copy_codes(save_directory):
    os.makedirs(save_directory, exist_ok=True)
    shutil.copy("./bridge/configuration_wbl.py", f"{save_directory}/")
    shutil.copy("./bridge/modeling_wbl.py", f"{save_directory}/")


def copy_tokenizer_files(tokenizer_path, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    for item in os.listdir(tokenizer_path):
        shutil.copy(f"{tokenizer_path}/{item}", f"{save_directory}/")


def save_model_shard_idx(model, save_directory):
    from huggingface_hub import split_torch_state_dict_into_shards
    from transformers.utils import SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME

    state_dict = model.state_dict()
    weights_name = SAFE_WEIGHTS_NAME
    filename_pattern = weights_name.replace(".safetensors", "{suffix}.safetensors")
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size="5GB"
    )
    if state_dict_split.is_sharded:
        index = {
            "metadata": {"total_parameters": model.num_parameters(), **state_dict_split.metadata},
            "weight_map": state_dict_split.tensor_to_filename,
        }
        save_index_file = os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)


def create_dummy_model(save_directory):
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM, AutoConfig
    # from transformers import DeepseekV3Config

    config = AutoConfig.from_pretrained(save_directory, trust_remote_code=True)
    # config = DeepseekV3Config(num_hidden_layers=2)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    save_model_shard_idx(model, save_directory)    


def megatron_to_hf(model, save_directory):
    from megatron.bridge import AutoBridge

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(BACKEND)
    parallel_state.initialize_model_parallel()

    output_path = f"{save_directory}/hf"
    bridge = AutoBridge.from_hf_pretrained(save_directory, trust_remote_code=True)
    bridge.save_hf_pretrained([model], output_path)
    copy_codes(output_path)


def load_megatron_model(checkpoint_path):
    from megatron.bridge.training.model_load_save import load_model_config, temporary_distributed_context
    from megatron.bridge.training.mlm_compat.arguments import _tokenizer_config_from_args
    from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
    from megatron.core.enums import ModelType
    from pretrain_gpt_for_wbl import model_provider_with_args

    _, mlm_args = load_model_config(checkpoint_path)

    mlm_args.context_parallel_size = 1
    mlm_args.expert_model_parallel_size = 1
    mlm_args.expert_tensor_parallel_size = 1
    mlm_args.pipeline_model_parallel_size = 1
    mlm_args.tensor_model_parallel_size = 1
    mlm_args.transformer_pipeline_model_parallel_size = 1

    with temporary_distributed_context(backend=BACKEND):
        cfg = _tokenizer_config_from_args(mlm_args)
        tokenizer = build_tokenizer(cfg)
        vocab_size = tokenizer.vocab_size

        mlm_args.padded_vocab_size = calculate_padded_vocab_size(
            vocab_size,
            mlm_args.make_vocab_size_divisible_by,
            tensor_model_parallel_size = 1,
        )
        # with torch.device("meta"):
        pre_process = parallel_state.is_pipeline_first_stage()
        post_process = parallel_state.is_pipeline_last_stage()
        model = model_provider_with_args(mlm_args, pre_process=pre_process, post_process=post_process)
        model.model_type = ModelType.encoder_or_decoder
        _load_model_weights_from_checkpoint(checkpoint_path, [model])
    return model


if __name__ == "__main__":
    from .wbl_bridge import WBLBridge     # register bridge
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model-path", type=str)
    parser.add_argument("--megatron-model-path", type=str)
    parser.add_argument("--tokenizer-path", type=str)
    args = parser.parse_args()

    copy_codes(args.hf_model_path)
    copy_tokenizer_files(args.tokenizer_path, args.hf_model_path)

    megatron_model = load_megatron_model(args.megatron_model_path)
    provider_to_config(megatron_model, args.hf_model_path)
    create_dummy_model(args.hf_model_path)

    megatron_to_hf(megatron_model, args.hf_model_path)
