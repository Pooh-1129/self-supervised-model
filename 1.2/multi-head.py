import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os
import seaborn as sns
import gc
import tracemalloc # 导入 tracemalloc 模块

# --- 1. Multi-Head Self-Attention Module ---
class MultiHeadSelfAttention(nn.Module):
    """
    A standard Multi-Head Self-Attention module implementation in PyTorch.
    This is more efficient than creating separate linear layers for Q, K, V.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # A single, batched projection for Q, K, V is more efficient
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape  # Batch size, Sequence length, Embedding dimension

        # Project to Q, K, V and split into heads
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        output = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)

        # Final output projection
        output = self.output_proj(output)
        return output

# --- 2. Manual FLOPS Calculation ---
def calculate_flops(seq_len, embed_dim, num_heads, batch_size=1):
    """
    Calculates the approximate FLOPS for a multi-head self-attention layer.
    FLOPs are estimated as 2 * MACs (Multiply-Accumulate operations).
    """
    C = embed_dim
    N = seq_len
    
    flops_qkv = batch_size * N * C * (3 * C) * 2
    flops_scores = batch_size * num_heads * N * (C // num_heads) * N * 2
    flops_output = batch_size * num_heads * N * N * (C // num_heads) * 2
    flops_out_proj = batch_size * N * C * C * 2
    
    total_flops = flops_qkv + flops_scores + flops_output + flops_out_proj
    return total_flops / 1e9 # Return in GFLOPS

# --- 3. Profiling Function ---
def profile_attention(input_lengths, embed_size=512, num_heads=4, device='cuda', warmup_iters=5, num_trials=10, dtype=torch.float32):
    all_metrics = ['flops', 'memory', 'time']
    results = {f'{m}_list': [] for m in all_metrics}
    ses = {f'{m}_se_list': [] for m in all_metrics}
    
    model = MultiHeadSelfAttention(embed_size, num_heads).to(device=device, dtype=dtype)
    model.eval()

    if device == 'cpu':
        tracemalloc.start()

    for length in input_lengths:
        print(f"Profiling on {device.upper()} for sequence length: {length} with dtype: {dtype}")
        
        trial_metrics = {f'{m}_trials': [] for m in all_metrics}

        try:
            input_tensor = torch.randn((1, length, embed_size), device=device, dtype=dtype)
            
            # Warm-up iterations
            for _ in range(warmup_iters):
                with torch.no_grad():
                    _ = model(input_tensor)

            for _ in range(num_trials):
                if device == 'cuda': torch.cuda.synchronize()
                
                start_time = time.time()
                with torch.no_grad():
                    _ = model(input_tensor)
                if device == 'cuda': torch.cuda.synchronize()
                end_time = time.time()
                trial_metrics['time_trials'].append(end_time - start_time)

                # --- MEMORY (in Bytes) ---
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                    with torch.no_grad():
                        _ = model(input_tensor)
                    peak_mem_bytes = torch.cuda.max_memory_allocated(device) # Bytes
                else: # CPU
                    # 使用 tracemalloc 来准确测量内存峰值
                    tracemalloc.clear_traces() # 清除之前的追踪记录
                    with torch.no_grad():
                         _ = model(input_tensor)
                    _, peak_mem_bytes = tracemalloc.get_traced_memory() # 获取峰值内存
                trial_metrics['memory_trials'].append(peak_mem_bytes)

                flops = calculate_flops(length, embed_size, num_heads)
                trial_metrics['flops_trials'].append(flops)
        
            for m in all_metrics:
                mean_val = np.mean(trial_metrics[f'{m}_trials'])
                se_val = np.std(trial_metrics[f'{m}_trials']) / np.sqrt(num_trials)
                results[f'{m}_list'].append(mean_val)
                ses[f'{m}_se_list'].append(se_val)

        except Exception as e:
            print(f"Failed for length {length} on {device}: {e}")
            for m in all_metrics:
                results[f'{m}_list'].append(np.nan)
                ses[f'{m}_se_list'].append(np.nan)
        
        del input_tensor
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # 如果是CPU，停止tracemalloc
    if device == 'cpu':
        tracemalloc.stop()

    return {**results, **ses}

# --- 4. Visualization ---
def plot_results(input_lengths, results, device):
    sns.set_theme(style="darkgrid")

    metrics_map = {
        'flops': {'title': 'Computational Complexity vs. Input Length', 'ylabel': 'GFLOPS'},
        'memory': {'title': 'Peak Memory Usage vs. Input Length', 'ylabel': 'Memory (Bytes)'},
        'time': {'title': 'Wall Clock Time vs. Input Length', 'ylabel': 'Time (s)'}
    }

    # Generate and save a separate plot for each metric
    for metric, props in metrics_map.items():
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Filter out NaN values for plotting
        valid_indices = ~np.isnan(results[f'{metric}_list'])
        valid_lengths = np.array(input_lengths)[valid_indices]
        valid_means = np.array(results[f'{metric}_list'])[valid_indices]
        valid_ses = np.array(results[f'{metric}_se_list'])[valid_indices]

        ax.errorbar(valid_lengths, valid_means, yerr=valid_ses, 
                    fmt='-o', capsize=5, label=f'{metric.capitalize()} on {device.upper()}')
        
        ax.set_title(f'{props["title"]} on {device.upper()}', fontsize=16)
        ax.set_xlabel("Input Length (log scale)", fontsize=12)
        ax.set_ylabel(f'{props["ylabel"]} (log scale)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.7)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'profile_{metric}_{device}.png', dpi=300)
        plt.show()


if __name__ == '__main__':
    input_lengths = [10, 100, 1000, 5000, 10000]
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    else:
        print("CUDA not available. Running on CPU only.")

    all_results = {}
    for device in devices:
        # Use float16 on GPU to prevent OOM errors in memory-constrained envs
        dtype_to_use = torch.float16 if device == 'cuda' else torch.float32
        profile_data = profile_attention(input_lengths, device=device, dtype=dtype_to_use)
        all_results[device] = profile_data
        plot_results(input_lengths, profile_data, device)
    
    print("\n--- Experiment Complete ---")

