import torch

from src.attn_utils import add_self_attentions
from src.flash_attn_triton import flash_attn_func
from src.hyper_attn_triton import hyper_attn_func
from src.angular_lsh_triton import AngularLSHTriton
import torch.backends.cuda

torch.backends.cuda.preferred_linalg_library("default")
class HyperAttention(torch.nn.Module):

    def __init__(self, input_dim=64, lsh_num_projs=8, block_size=256, sample_size=256, min_seq_len=2048,
                 smooth_block=False,top_k=128, **kwargs):
        """
        - block_size and sample_size must be divisible by 128
        """
        super().__init__()
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.top_k = top_k
        self.smooth_block = smooth_block
        self.lsh = AngularLSHTriton(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim))

    def compute_lev_scores(self, key):
        """
        Compute leverage scores and select top-k keys based on the scores.
        """
        batch_size, num_heads, kv_length, dim = key.shape

        # Reshape for batch processing
        key_reshaped = key.view(batch_size * num_heads, kv_length, dim)

        # Convert to float32 for QR decomposition to avoid 'BFloat16' error
        key_reshaped = key_reshaped.to(torch.float32)
        key_reshaped = key_reshaped.contiguous()
        # Compute QR decomposition (Orthogonalization)
        orth_key, _ = torch.linalg.qr(key_reshaped+1e-6*torch.randn_like(key_reshaped))

        # Compute leverage scores as squared norm
        lev_scores = torch.norm(orth_key, dim=-1) ** 2

        # Ensure top_k is properly initialized
        top_k = self.top_k if self.top_k is not None else kv_length  # Default: keep all keys

        _, top_indices = torch.topk(lev_scores, top_k, dim=-1, largest=True, sorted=True)
        #print(f"Max index in top_indices: {top_indices.max().item()}, kv_length: {kv_length}")
        #print(f"Min index in top_indices: {top_indices.min().item()}")
        top_indices = torch.clamp(top_indices, max=kv_length - 1)
        # ✅ Ensure `top_indices` has correct shape
        top_indices = top_indices.view(batch_size, num_heads, top_k)

        # ✅ Gather top-k keys
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, dim)


        selected_keys = torch.gather(key, dim=2, index=top_indices_expanded)

        return selected_keys, top_indices  # ✅ Return both values

    
    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor, scale=None, causal=False,
                return_lse=False):
        """
        Forward function for HyperAttention. If no causal masking, simply invokes forward_no_causal_mask method.
        If there is causal masking, it partitions the attention matrix and recurses on the partitions.
        inputs:
            - query, key, and valu: must have same sequence lengths but dimension of values vectors can be different
            from that of query or key
            - sequence lengths must be divisible by block_size
        output:
            - attn: (approximation of) the final attention output tensor
            - lse: (approximation of) log sum exp of the qk matrix
        """
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        # Get top-k keys based on Leverage Scores
        '''selected_keys, top_indices = self.compute_lev_scores(key)
        #print(f"selected_keys shape: {selected_keys.shape}")  # Should be (batch, num_heads, top_k, dim)
        #print(f"top_indices shape: {top_indices.shape}")      # Should be (batch, num_heads, top_k)
        # Select corresponding values using the same indices
        batch_size, num_heads, kv_length, dim_v = value.shape
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, dim_v)
        selected_values = torch.gather(value, dim=2, index=top_indices_expanded)  
        # (batch, num_heads, top_k, dim_v)
        #selected_values = selected_values.view(batch_size, num_heads, self.top_k, dim_v)

        # Apply HyperAttention with selected keys and values
        attn, lse = self.forward_no_causal_mask(query=query, key=selected_keys, value=selected_values, scale=scale)
        if not return_lse:
            return attn
        else:
            return attn, lse'''
        n_query = query.shape[2]
        batch_size, n_heads, n_key, dim = key.shape
        scale = scale or dim ** (-0.5)
        assert n_query == n_key

        # without causal masking
        if causal is False:
            attn, lse = self.forward_no_causal_mask(query, key, value, scale)

        else:  # with causal masking
            if n_key <= self.min_seq_len:
                attn, lse = flash_attn_func(query.transpose(1, 2),
                                            key.transpose(1, 2),
                                            value.transpose(1, 2),
                                            None, True, scale)
                attn = attn.transpose(1, 2)

            else:
                # If n_query is odd we pad inputs by zero rows
                if n_query % 2:
                    query = torch.nn.functional.pad(query, (0, 0, 0, 1), mode='constant', value=0.)
                    key = torch.nn.functional.pad(key, (0, 0, 0, 1), mode='constant', value=0.)
                    value = torch.nn.functional.pad(value, (0, 0, 0, 1), mode='constant', value=0.)

                # extract block diagonal parts
                q_bd = query.view(batch_size, 2 * n_heads, query.shape[2] // 2, query.shape[-1])
                k_bd = key.view(batch_size, 2 * n_heads, key.shape[2] // 2, key.shape[-1])
                v_bd = value.view(batch_size, 2 * n_heads, key.shape[2] // 2, value.shape[-1])

                attn_bd, lse_bd = self.forward(q_bd, k_bd, v_bd, scale, True, True)

                if attn_bd.shape[2] not in attn_bd.stride():
                    attn_bd = attn_bd.contiguous()
                attn_bd = attn_bd.view(batch_size, n_heads, -1, dim)

                if lse_bd.shape[2] not in lse_bd.stride():
                    lse_bd = lse_bd.contiguous()
                lse_bd = lse_bd.view(batch_size, n_heads, -1, 1)

                # lowe diagonal block is an unmasked attention
                attn_unmasked, lse_unmasked = self.forward_no_causal_mask(
                    query[:, :, key.shape[2] // 2:, :], key[:, :, :key.shape[2] // 2, :],
                    value[:, :, :key.shape[2] // 2, :], scale)

                attn_up, lse_up = attn_bd[:, :, :query.shape[2] // 2, :], lse_bd[:, :, :query.shape[2] // 2, :]
                attn_down, lse_down = add_self_attentions(attn_bd[:, :, query.shape[2] // 2:, :],
                                                          lse_bd[:, :, query.shape[2] // 2:, :],
                                                          attn_unmasked, lse_unmasked)

                attn = torch.cat((attn_up, attn_down), dim=-2)
                lse = torch.cat((lse_up, lse_down), dim=-2)

                if n_query % 2:
                    attn = attn[:, :, :-1, :]
                    lse = lse[:, :, :-1, :]

        if not return_lse:
            return attn
        else:
            return attn, lse

    def forward_no_causal_mask(self, query, key, value, scale):
        """
            - sequence lengths must be divisible by block_size
        """
        batch_size, head_size, n_query, dim = query.shape

        if self.min_seq_len > n_query:
            attn, lse = flash_attn_func(query.transpose(1, 2),
                                        key.transpose(1, 2),
                                        value.transpose(1, 2),
                                        None, False, scale)
        else:
            # Hash keys and queries via SortLSH and obtain buckets
            _, query_sort_idx = torch.sort(self.lsh.hash_triton(query), dim=2, stable=True)  # batch_size x head_size x n
            _, key_sort_idx = torch.sort(self.lsh.hash_triton(key), dim=2, stable=True)

            # Now run hyper attention function on q,k,v and the permutations
            attn, lse = hyper_attn_func(query.transpose(1, 2),
                                        key.transpose(1, 2),
                                        value.transpose(1, 2),
                                        query_sort_idx.transpose(1, 2),
                                        key_sort_idx.transpose(1, 2),
                                        self.block_size,
                                        self.sample_size,
                                        scale,
                                        self.smooth_block,
                                        )
        attn = attn.transpose(1, 2)

        return attn, lse.unsqueeze(-1)
