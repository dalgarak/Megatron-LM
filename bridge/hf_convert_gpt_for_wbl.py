import os
import shutil
import torch
from megatron.core import parallel_state


BACKEND = "nccl"


def provider_to_config(model, path):
    pass


def copy_codes(path):
    shutil.copy("./bridge/configuration_wbl.py", f"{path}/")
    shutil.copy("./bridge/modeling_wbl.py", f"{path}/")


def create_dummy_model(path):
    from transformers import AutoModelForCausalLM, AutoConfig
    # from transformers import DeepseekV3Config

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    # config = DeepseekV3Config(num_hidden_layers=2)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.save_pretrained(path)


def megatron_to_hf(model, path):
    from megatron.bridge import AutoBridge

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(BACKEND)
    parallel_state.initialize_model_parallel()

    output_path = f"{path}/hf"
    os.makedirs(output_path, exist_ok=True)
    bridge = AutoBridge.from_hf_pretrained(path, trust_remote_code=True)
    bridge.save_hf_pretrained([model], output_path)


def load_megatron_model(checkpoint_path):
    from megatron.bridge.training.model_load_save import load_model_config, temporary_distributed_context
    from megatron.bridge.training.mlm_compat.arguments import _tokenizer_config_from_args
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
    from megatron.core.enums import ModelType
    from pretrain_gpt_for_wbl import model_provider_with_args

    model_cfg, mlm_args = load_model_config(checkpoint_path)
    with temporary_distributed_context(backend=BACKEND):
        cfg = _tokenizer_config_from_args(mlm_args)
        tokenizer = build_tokenizer(cfg)
        vocab_size = tokenizer.vocab_size

        mlm_args.padded_vocab_size = calculate_padded_vocab_size(
            vocab_size,
            mlm_args.make_vocab_size_divisible_by,
            model_cfg.tensor_model_parallel_size,
        )
        pre_process = parallel_state.is_pipeline_first_stage()
        post_process = parallel_state.is_pipeline_last_stage()
        model = model_provider_with_args(mlm_args, pre_process=pre_process, post_process=post_process)
        model.model_type = ModelType.encoder_or_decoder
    return model


if __name__ == "__main__":
    from .wbl_bridge import WBLBridge     # register bridge

    hf_model_path = "./local/test_model"
    megatron_model_path = "/mnt/output/jaycha/pretrain_stage1_checkpoints/checkpoints_20251014/100b_moe_initial_128pair/iter_0033000"
    megatron_model = load_megatron_model(megatron_model_path)
    copy_codes(hf_model_path)
    provider_to_config(megatron_model, hf_model_path)
    # create_dummy_model(hf_model_path)
    megatron_to_hf(megatron_model, hf_model_path)
