"""
Hardware and Architectural Constants for TPU v5e & vLLM Serving.
================================================================
Defines physical memory limits, static XLA bucket profiles, and model constants.
"""

from typing import Dict, Tuple

# Gemma-4 31B BF16 Model Specifications
MODEL_NAME: str = "gemma-4-31b"
MODEL_PATH: str = "/models/gemma-4-31b"
WEIGHTS_HBM_BYTES: int = 61_400_000_000  # ~61.4 GB weights hot in HBM

# Cloud TPU v5e Hardware Limits (16 GB HBM per chip)
# TP=8 (2x4 ICI mesh, 8 chips): 128 GB total HBM
TP8_CHIPS: int = 8
TP8_TOTAL_HBM_BYTES: int = 128 * (1024**3)  # 137.4 GB
TP8_KV_POOL_BYTES: int = 46_800_000_000     # ~46.8 GB KV pool headroom
TP8_DEFAULT_KV_CAPACITY_TOKENS: int = 120_000 # ~120,000 token capacity pool

# TP=16 (4x4 ICI mesh, 16 chips): 256 GB total HBM
TP16_CHIPS: int = 16
TP16_TOTAL_HBM_BYTES: int = 256 * (1024**3)
TP16_DEFAULT_KV_CAPACITY_TOKENS: int = 400_000

# Static XLA Compilation Buckets on Cloud TPU v5e
DEFAULT_XLA_BUCKETS: Tuple[int, ...] = (16, 32, 64, 128, 256)

# Ingestion Cohort Size (Starvation prevention and cohort isolation)
BATCH_COHORT_SIZE: int = 64

# Empirical Measured Step Latencies for Gemma-4 31B (TP=8, 2x4 mesh)
# Demonstrates strongly sub-linear latency scaling across batch sizes
EMPIRICAL_STEP_LATENCIES_S: Dict[int, float] = {
    16: 0.02414,  # ~24.14 ms
    32: 0.02584,  # ~25.84 ms (+7.0% latency for 2x concurrency)
    64: 0.02888,  # ~28.88 ms (+11.8% latency for 2x concurrency)
    128: 0.03450, # ~34.50 ms
    256: 0.04850, # ~48.50 ms
}

# Linear Regression Model Coefficients: Latency = alpha + beta * BucketSize
DEFAULT_ALPHA_S: float = 0.0200   # Base kernel launch / step overhead
DEFAULT_BETA_S: float = 0.00014   # Marginal latency per batch stream

# Colocated Chunked Prefill & Decode Constraints
DEFAULT_STEP_TOKEN_BUDGET: int = 2048
DEFAULT_CHUNK_SIZE_CAP: int = 512
DEFAULT_BLOCK_SIZE: int = 16
