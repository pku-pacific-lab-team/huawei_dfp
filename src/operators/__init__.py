"""Public GEMM, element-wise Vector and FP32 normalization operator API."""
from .actions import Action, ActionLedger
from .adapters import adapt_gemm_report
from .primitives import Gemm, simulate_gemm
from .types import OperatorSpec, OpResult, TensorSpec
from .vector import Vector, simulate_vector
from .composite import RMSNorm, Softmax, simulate_composite
from .connections import plan_connection
from .blocks import Residual, SwiGLU, SwiGLUFFN, simulate_block
from .attention import KVCacheSpec, KVCacheAccess, DecodeAttention, simulate_kv_cache, simulate_attention
from .projections import RMSNormINT8, QKVProjection, AttentionOutputProjection

__all__ = ["Action", "ActionLedger", "Gemm", "OperatorSpec", "OpResult", "TensorSpec",
           "adapt_gemm_report", "simulate_gemm", "Vector", "simulate_vector",
           "RMSNorm", "Softmax", "simulate_composite", "plan_connection",
           "Residual", "SwiGLU", "SwiGLUFFN", "simulate_block",
           "KVCacheSpec", "KVCacheAccess", "DecodeAttention", "simulate_kv_cache", "simulate_attention",
           "RMSNormINT8", "QKVProjection", "AttentionOutputProjection"]
