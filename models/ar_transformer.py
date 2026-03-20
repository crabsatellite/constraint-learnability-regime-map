"""
Autoregressive Transformer for generating 8x8x8 VQ-VAE latent codes.

Generates Minecraft buildings by predicting codebook indices in raster order
(z-first, then y, then x) in the 8³ latent space.

Supports optional tag conditioning via prefix tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb3D(nn.Module):
    """3D sinusoidal positional embedding for 8x8x8 grid."""

    def __init__(self, dim, grid_size=8):
        super().__init__()
        self.dim = dim
        self.grid_size = grid_size

        # Precompute 3D position embeddings
        pos = torch.arange(grid_size)
        gx, gy, gz = torch.meshgrid(pos, pos, pos, indexing='ij')
        coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).float()  # (512, 3)

        d = dim // 6 * 2  # dims per axis, must be even
        freq = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))

        pe = torch.zeros(grid_size ** 3, dim)
        for axis in range(3):
            offset = axis * (d)
            pe[:, offset:offset + d:2] = torch.sin(coords[:, axis:axis+1] * freq)
            pe[:, offset + 1:offset + d:2] = torch.cos(coords[:, axis:axis+1] * freq)

        self.register_buffer('pe', pe)  # (512, dim)

    def forward(self, seq_len):
        return self.pe[:seq_len]


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention with causal mask
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ARTransformer3D(nn.Module):
    """
    Autoregressive Transformer for 8³ VQ-VAE latent code generation.

    Input: sequence of codebook indices (0..num_codes-1)
    Output: next-token logits over codebook

    Supports two conditioning modes:
    1. Tag conditioning: tags embedded and prepended as prefix tokens (legacy)
    2. Structural conditioning: 6 structural features as prefix tokens (v3)
       - height_bucket (0-3), size_bucket (0-3), footprint_bucket (0-2)
       - symmetry_flag (0-1), enclosure_flag (0-1), complexity_bucket (0-2)
    """

    # Structural feature vocabulary sizes (number of valid values per feature)
    STRUCT_FEATURE_VOCABS = [4, 4, 3, 2, 2, 3]  # height, size, footprint, symmetry, enclosure, complexity
    STRUCT_FEATURE_NAMES = ['height', 'size', 'footprint', 'symmetry', 'enclosure', 'complexity']
    N_STRUCT_FEATURES = 6

    def __init__(self, num_codes=1024, dim=512, n_layers=8, n_heads=8,
                 dropout=0.1, max_seq_len=512, num_tags=0, grid_size=8,
                 struct_cond=False):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_tags = num_tags
        self.grid_size = grid_size
        self.struct_cond = struct_cond

        # Code embedding (codebook index -> vector)
        # +1 for BOS token, +1 for padding
        self.code_embed = nn.Embedding(num_codes + 2, dim)
        self.bos_token_id = num_codes
        self.pad_token_id = num_codes + 1

        # 3D positional embedding
        self.pos_embed = SinusoidalPosEmb3D(dim, grid_size)

        # Tag conditioning (optional, legacy)
        if num_tags > 0:
            self.tag_embed = nn.Embedding(num_tags + 1, dim)  # +1 for no-tag
            self.tag_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            # Learned "tag position" embeddings (up to 8 tag prefix tokens)
            self.tag_pos = nn.Embedding(8, dim)

        # Structural constraint conditioning (v3)
        if struct_cond:
            # Each feature gets its own embedding table (+1 for "unconditioned" mask token)
            self.struct_embeddings = nn.ModuleList([
                nn.Embedding(vocab_size + 1, dim)  # last index = "any" (unconditional)
                for vocab_size in self.STRUCT_FEATURE_VOCABS
            ])
            # Projection to combine feature embedding with position info
            self.struct_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            # Learned position embeddings for the 6 prefix slots
            self.struct_pos = nn.Embedding(self.N_STRUCT_FEATURES, dim)
            # Store the "unconditional" token IDs for CFG
            self.struct_uncond_ids = [v for v in self.STRUCT_FEATURE_VOCABS]  # last valid index = mask

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_codes, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)

    def _embed_struct_cond(self, struct_features, device):
        """Embed structural features into prefix tokens.

        Args:
            struct_features: (B, 6) tensor of feature bucket indices
        Returns:
            prefix: (B, 6, dim) prefix token embeddings
        """
        B = struct_features.shape[0]
        prefix_tokens = []
        for i, embed_layer in enumerate(self.struct_embeddings):
            feat_emb = embed_layer(struct_features[:, i])  # (B, dim)
            prefix_tokens.append(feat_emb)
        prefix = torch.stack(prefix_tokens, dim=1)  # (B, 6, dim)
        prefix = self.struct_proj(prefix)
        pos = self.struct_pos(torch.arange(self.N_STRUCT_FEATURES, device=device))
        prefix = prefix + pos.unsqueeze(0)
        return prefix

    def forward(self, indices, tags=None, struct_features=None):
        """
        Args:
            indices: (B, T) codebook indices, T <= 512
            tags: (B, num_tag_slots) tag indices, optional (legacy)
            struct_features: (B, 6) structural feature buckets, optional (v3)

        Returns:
            logits: (B, T, num_codes) next-token prediction logits
        """
        B, T = indices.shape
        device = indices.device
        n_prefix = 0

        # Embed code tokens
        tok_emb = self.code_embed(indices)  # (B, T, dim)

        # Add 3D positional encoding
        pos_emb = self.pos_embed(T).unsqueeze(0)  # (1, T, dim)
        x = tok_emb + pos_emb

        # Prepend structural conditioning tokens (v3)
        if struct_features is not None and self.struct_cond:
            prefix = self._embed_struct_cond(struct_features, device)  # (B, 6, dim)
            x = torch.cat([prefix, x], dim=1)  # (B, 6 + T, dim)
            n_prefix = self.N_STRUCT_FEATURES

        # Prepend tag conditioning tokens (legacy)
        elif tags is not None and self.num_tags > 0:
            tag_emb = self.tag_embed(tags)  # (B, n_tags, dim)
            tag_emb = self.tag_proj(tag_emb)
            n_tag = tag_emb.shape[1]
            tag_pos = self.tag_pos(torch.arange(n_tag, device=device))
            tag_emb = tag_emb + tag_pos
            x = torch.cat([tag_emb, x], dim=1)  # (B, n_tags + T, dim)
            n_prefix = n_tag

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)

        # Strip prefix tokens from output
        if n_prefix > 0:
            x = x[:, n_prefix:]  # (B, T, dim)

        logits = self.head(x)  # (B, T, num_codes)
        return logits

    @torch.no_grad()
    def generate(self, tags=None, struct_features=None, temperature=1.0,
                 top_k=None, top_p=None, device='cuda'):
        """
        Autoregressively generate 8³ = 512 codebook indices.

        Args:
            tags: (1, num_tag_slots) or None
            struct_features: (1, 6) structural feature buckets, or None
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
        Returns:
            indices: (1, 512) generated codebook indices
        """
        self.eval()
        seq = torch.tensor([[self.bos_token_id]], device=device)  # (1, 1) BOS

        for i in range(self.max_seq_len):
            logits = self.forward(seq, tags=tags, struct_features=struct_features)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[mask] = float('-inf')
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            seq = torch.cat([seq, next_token], dim=1)

        return seq[:, 1:]

    @torch.no_grad()
    def generate_batch(self, batch_size=16, tags=None, struct_features=None,
                       temperature=1.0, top_k=50, device='cuda',
                       cfg_scale=0.0):
        """Generate a batch with KV cache. Supports classifier-free guidance.

        Args:
            struct_features: (batch_size, 6) or None. For CFG, also generates
                unconditional in parallel when cfg_scale > 0.
            cfg_scale: classifier-free guidance weight. 0 = no guidance.
        """
        self.eval()
        use_cfg = cfg_scale > 0 and struct_features is not None and self.struct_cond

        if use_cfg:
            # Double batch: [conditioned; unconditioned]
            uncond_features = torch.tensor(
                [self.struct_uncond_ids] * batch_size,
                dtype=torch.long, device=device
            )
            cfg_features = torch.cat([struct_features, uncond_features], dim=0)
            effective_bs = batch_size * 2
        else:
            cfg_features = struct_features
            effective_bs = batch_size

        seq = torch.full((effective_bs, 1), self.bos_token_id,
                         dtype=torch.long, device=device)

        # Process structural prefix through transformer first if needed
        kv_cache = [None] * len(self.blocks)

        if cfg_features is not None and self.struct_cond:
            prefix = self._embed_struct_cond(cfg_features, device)  # (effective_bs, 6, dim)
            x = prefix
            for layer_idx, block in enumerate(self.blocks):
                x, kv_cache[layer_idx] = self._cached_block_forward(
                    block, x, kv_cache[layer_idx]
                )
            # Prefix is now cached in KV cache

        for i in range(self.max_seq_len):
            if i == 0:
                tok_emb = self.code_embed(seq)
                pos_emb = self.pos_embed(1).unsqueeze(0)
                x = tok_emb + pos_emb
            else:
                tok_emb = self.code_embed(seq[:, -1:])
                pos_emb = self.pos_embed(i + 1)[i:i+1].unsqueeze(0)
                x = tok_emb + pos_emb

            for layer_idx, block in enumerate(self.blocks):
                x, kv_cache[layer_idx] = self._cached_block_forward(
                    block, x, kv_cache[layer_idx]
                )

            logits = self.head(self.ln_out(x[:, -1:, :]))[:, 0, :]  # (effective_bs, num_codes)

            if use_cfg:
                # Split conditioned and unconditioned logits
                cond_logits = logits[:batch_size]
                uncond_logits = logits[batch_size:]
                # CFG: logits = uncond + scale * (cond - uncond)
                logits_combined = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
                next_logits = logits_combined / temperature
            else:
                next_logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # (batch_size, 1)

            if use_cfg:
                # Both branches get the same token (conditioned output)
                seq = torch.cat([seq, next_token.repeat(2, 1)], dim=1)
            else:
                seq = torch.cat([seq, next_token], dim=1)

        if use_cfg:
            return seq[:batch_size, 1:]
        return seq[:, 1:]

    def _cached_block_forward(self, block, x, kv_pair):
        """Forward one transformer block with KV caching."""
        # Pre-norm attention
        normed = block.ln1(x)
        B, T, C = normed.shape

        qkv = block.attn.qkv(normed).reshape(B, T, 3, block.attn.n_heads, block.attn.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Append to cache
        if kv_pair is not None:
            prev_k, prev_v = kv_pair
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        # Attention (no causal mask needed — cache only has past tokens)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, C)
        attn_out = block.attn.proj(out)

        x = x + attn_out
        x = x + block.mlp(block.ln2(x))

        return x, (k, v)
