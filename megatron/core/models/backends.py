# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from abc import abstractmethod
from typing import Optional, Protocol, Tuple, Callable

import torch

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


class BackendSpecProvider(Protocol):
    """A protocol for providing the submodules used in Spec building."""

    @abstractmethod
    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        ...

    @abstractmethod
    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses"""
        ...

    @abstractmethod
    def row_parallel_linear_layer_norm(self) -> type:
        """ JHSHIN: Which row parallel linear module the backend uses, with Post-LN """
        ...

    @abstractmethod
    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend support a single module for layernorm and linear"""
        ...

    @abstractmethod
    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        ...

    @abstractmethod
    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """Which module for layernorm"""
        ...

    @abstractmethod
    def core_attention(self) -> type:
        """Which module to use for attention"""
        ...

    @abstractmethod
    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        """Which module and submodules to use for grouped mlp"""
        ...

# JHSHIN added
class RowParallelLinearLayerNorm(RowParallelLinear):
    """ JHSHIN: Modified From RowParallelLinear with an additional Post-LN,
        for Peri-LN Implementation. 
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(input_size, output_size, config=config, init_method=init_method,
                         bias=bias, input_is_parallel=input_is_parallel, skip_bias_add=skip_bias_add,
                         stride=stride, keep_master_weight_for_test=keep_master_weight_for_test,
                         is_expert=is_expert, tp_comm_buffer_name=tp_comm_buffer_name,
                         tp_group=tp_group)
        # 우리가 사용할 게 RMSNorm이므로 apex 유무와 상관없이 Torch LayerNorm을 사용.
        self.post_layernorm = WrappedTorchNorm(config, output_size)

    def forward(self, x):
        """ Forward with additional Post-LN on output. """
        output, output_bias = super().forward(x)
        return self.post_layernorm(output), output_bias


class LocalSpecProvider(BackendSpecProvider):
    """A protocol for providing Local submodules used in Spec building."""

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        return ColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses"""
        return RowParallelLinear

    def row_parallel_linear_layer_norm(self) -> type:
        """Which row parallel linear module the backend uses with Post-LayerNorm."""
        return RowParallelLinearLayerNorm

    def fuse_layernorm_and_linear(self) -> bool:
        """Does the backend choose a single module for layernorm and linear"""
        return False

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return None

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """Which module to use for layer norm"""
        if rms_norm:
            # Matching get_gpt_layer_local_spec.
            # Why does the global need to be updated?
            global LNImpl
            LNImpl = WrappedTorchNorm
        return LNImpl

    def core_attention(self) -> type:
        """Which module to use for attention"""
        return DotProductAttention

    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        """Which module and submodules to use for grouped mlp"""
        if moe_use_grouped_gemm:
            warnings.warn(
                "The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. "
                "Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP."
            )
            return GroupedMLP, None
        else:
            return SequentialMLP, MLPSubmodules(
                linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
            )

