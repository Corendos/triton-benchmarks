import torch
import random
import time

from kernels.chunked_prefill_paged_decode import chunked_prefill_paged_decode

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def benchmark_decode_only():
    """Represents a scenario where all the sequence have 1 input token"""
    token_count = 256
    batch_size = 256
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 32768
    max_input_len = 1
    scale = 0.08838834765

    max_seq_len = 8192
    max_block_per_seq = max_seq_len // block_size

    seq_len = 2048

    query = torch.randn(token_count, num_heads, head_size, dtype=torch.bfloat16)
    o = torch.zeros_like(query)
    k = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16)
    v = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16)
    x = 8
    key_cache = torch.randn(num_blocks, num_kv_heads, head_size // x, block_size, x, dtype=torch.bfloat16)
    value_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.bfloat16)

    values = torch.arange(0, num_blocks, dtype=torch.long)
    values = values[torch.randperm(num_blocks)]

    block_tables = torch.zeros((batch_size, max_block_per_seq), dtype=torch.int32)
    for i in range(batch_size):
        block_tables[i][0:seq_len//block_size] = torch.randint(0, num_blocks, (seq_len // block_size,), dtype=torch.int32)

    query_lens = [1 for _ in range(batch_size)]
    context_lens = [seq_len for query_len in query_lens]
    seqlen_q = torch.tensor(query_lens, dtype=torch.long)
    seqlen_kv = torch.tensor([a + b for a, b in zip(query_lens, context_lens)], dtype=torch.long)
    start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)

    # warmup
    chunked_prefill_paged_decode(
        query,
        k,
        v,
        o,
        "auto",
        key_cache,
        value_cache,
        block_tables,
        start_loc,
        seqlen_kv,
        max_input_len,
        1.0,
        1.0,
        alibi_slopes=None,
        sliding_window=None,
        sm_scale=None,
    )

    start_time = time.time()
    ITERATIONS = 3000
    for _ in range(ITERATIONS):
        chunked_prefill_paged_decode(
            query,
            k,
            v,
            o,
            "auto",
            key_cache,
            value_cache,
            block_tables,
            start_loc,
            seqlen_kv,
            max_input_len,
            1.0,
            1.0,
            alibi_slopes=None,
            sliding_window=None,
            sm_scale=None,
        )
    torch.cuda.synchronize()
    end_time = time.time()

    print("Decode only scenario")
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")
    print(f"triton Time per invocation: {((end_time - start_time) / ITERATIONS)*1000:.2f} ms\n")

def benchmark_prefill_only():
    """Represents a scenario where all the sequence have 256 input tokens"""
    token_count = 256 * 256
    batch_size = 256
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 32768
    max_input_len = 256
    scale = 0.08838834765

    max_seq_len = 8192
    max_block_per_seq = max_seq_len // block_size

    seq_len = 2048

    query = torch.randn(token_count, num_heads, head_size, dtype=torch.bfloat16)
    o = torch.zeros_like(query)
    k = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16)
    v = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16)
    x = 8
    key_cache = torch.randn(num_blocks, num_kv_heads, head_size // x, block_size, x, dtype=torch.bfloat16)
    value_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.bfloat16)

    values = torch.arange(0, num_blocks, dtype=torch.long)
    values = values[torch.randperm(num_blocks)]

    block_tables = torch.zeros((batch_size, max_block_per_seq), dtype=torch.int32)
    for i in range(batch_size):
        block_tables[i][0:seq_len//block_size] = torch.randint(0, num_blocks, (seq_len // block_size,), dtype=torch.int32)

    query_lens = [256 for _ in range(batch_size)]
    context_lens = [seq_len for query_len in query_lens]
    seqlen_q = torch.tensor(query_lens, dtype=torch.long)
    seqlen_kv = torch.tensor([a + b for a, b in zip(query_lens, context_lens)], dtype=torch.long)
    start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)

    # warmup
    chunked_prefill_paged_decode(
        query,
        k,
        v,
        o,
        "auto",
        key_cache,
        value_cache,
        block_tables,
        start_loc,
        seqlen_kv,
        max_input_len,
        1.0,
        1.0,
        alibi_slopes=None,
        sliding_window=None,
        sm_scale=None,
    )

    start_time = time.time()
    ITERATIONS = 300
    for _ in range(ITERATIONS):
        chunked_prefill_paged_decode(
            query,
            k,
            v,
            o,
            "auto",
            key_cache,
            value_cache,
            block_tables,
            start_loc,
            seqlen_kv,
            max_input_len,
            1.0,
            1.0,
            alibi_slopes=None,
            sliding_window=None,
            sm_scale=None,
        )
    torch.cuda.synchronize()
    end_time = time.time()

    print("Prefill only scenario")
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")
    print(f"triton Time per invocation: {((end_time - start_time) / ITERATIONS)*1000:.2f} ms\n")

def benchmark_highly_skewed():
    """Represents a scenario where all the sequence have 256 input tokens"""
    token_count = 256 + 255
    batch_size = 256
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 32768
    max_input_len = 256
    scale = 0.08838834765

    max_seq_len = 8192
    max_block_per_seq = max_seq_len // block_size

    seq_len = 2048

    query = torch.randn(token_count, num_heads, head_size, dtype=torch.bfloat16)
    o = torch.zeros_like(query)
    k = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16)
    v = torch.randn(token_count, num_kv_heads, head_size, dtype=torch.bfloat16)
    x = 8
    key_cache = torch.randn(num_blocks, num_kv_heads, head_size // x, block_size, x, dtype=torch.bfloat16)
    value_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.bfloat16)

    values = torch.arange(0, num_blocks, dtype=torch.long)
    values = values[torch.randperm(num_blocks)]

    block_tables = torch.zeros((batch_size, max_block_per_seq), dtype=torch.int32)
    for i in range(batch_size):
        block_tables[i][0:seq_len//block_size] = torch.randint(0, num_blocks, (seq_len // block_size,), dtype=torch.int32)

    query_lens = [256] + [1 for _ in range(batch_size - 1)]
    context_lens = [seq_len for query_len in query_lens]
    seqlen_q = torch.tensor(query_lens, dtype=torch.long)
    seqlen_kv = torch.tensor([a + b for a, b in zip(query_lens, context_lens)], dtype=torch.long)
    start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0)

    # warmup
    chunked_prefill_paged_decode(
        query,
        k,
        v,
        o,
        "auto",
        key_cache,
        value_cache,
        block_tables,
        start_loc,
        seqlen_kv,
        max_input_len,
        1.0,
        1.0,
        alibi_slopes=None,
        sliding_window=None,
        sm_scale=None,
    )

    start_time = time.time()
    ITERATIONS = 3000
    for _ in range(ITERATIONS):
        chunked_prefill_paged_decode(
            query,
            k,
            v,
            o,
            "auto",
            key_cache,
            value_cache,
            block_tables,
            start_loc,
            seqlen_kv,
            max_input_len,
            1.0,
            1.0,
            alibi_slopes=None,
            sliding_window=None,
            sm_scale=None,
        )
    torch.cuda.synchronize()
    end_time = time.time()

    print("Highly skewed scenario")
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")
    print(f"triton Time per invocation: {((end_time - start_time) / ITERATIONS)*1000:.2f} ms\n")

if __name__ == '__main__':
    seed_everything(0)
    torch.set_default_device("cuda")

    benchmark_decode_only()
    benchmark_prefill_only()
    benchmark_highly_skewed()