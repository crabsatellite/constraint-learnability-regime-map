"""
Conditioning diagnostic test — run at any checkpoint to verify
whether the model is actually using structural constraints.

Test A: Condition Swap — same seed, different conditions → output should differ
Test B: Condition Drop — conditioned vs unconditioned → output should differ
Test C: Gradient attribution — does loss gradient flow through conditioning?
"""

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vqvae import VQVAE3D
from models.ar_transformer import ARTransformer3D


def load_models(vqvae_ckpt, ar_ckpt, device='cuda'):
    vqvae_data = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
    vqvae_args = vqvae_data['args']
    vqvae = VQVAE3D(
        vocab_size=vqvae_data['vocab_size'],
        embed_dim=vqvae_args['embed_dim'],
        hidden_dim=vqvae_args['hidden_dim'],
        code_dim=vqvae_args['code_dim'],
        num_codes=vqvae_args['num_codes'],
        n_downsample=vqvae_args.get('n_downsample', 3),
    ).to(device)
    vqvae.load_state_dict(vqvae_data['model_state_dict'])
    vqvae.eval()

    ar_data = torch.load(ar_ckpt, map_location=device, weights_only=False)
    ar_args = ar_data['args']
    pe_shape = ar_data['model_state_dict']['pos_embed.pe'].shape[0]
    grid_size = round(pe_shape ** (1/3))
    seq_len = grid_size ** 3
    ar = ARTransformer3D(
        num_codes=ar_data['num_codes'],
        dim=ar_args['dim'],
        n_layers=ar_args['n_layers'],
        n_heads=ar_args['n_heads'],
        dropout=0.0,
        max_seq_len=seq_len,
        num_tags=0,
        grid_size=grid_size,
        struct_cond=ar_data.get('struct_cond', True),
    ).to(device)
    ar.load_state_dict(ar_data['model_state_dict'])
    ar.eval()
    return vqvae, ar, grid_size


def generate_with_seed(ar, features, seed, n=4, temperature=0.9, top_k=100, device='cuda'):
    """Generate samples with fixed random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    codes = ar.generate_batch(
        batch_size=n,
        struct_features=features,
        temperature=temperature,
        top_k=top_k,
        device=device,
        cfg_scale=0.0,  # No CFG for raw diagnostic
    )
    return codes  # (n, 512)


def codes_to_voxels(vqvae, codes, grid_size):
    code_grid = codes.reshape(-1, grid_size, grid_size, grid_size)
    logits = vqvae.decode_from_indices(code_grid)
    return logits.argmax(dim=1).cpu().numpy()


def measure_voxel_stats(voxels):
    """Compute structural stats for a batch of voxels."""
    stats = []
    for v in voxels:
        filled = v > 0
        n_blocks = int(filled.sum())
        if n_blocks == 0:
            stats.append({'blocks': 0, 'height': 0, 'types': 0})
            continue
        ys = np.where(filled.any(axis=(0, 2)))[0]
        height = int(ys[-1] - ys[0] + 1) if len(ys) > 0 else 0
        types = len(np.unique(v[filled]))
        stats.append({'blocks': n_blocks, 'height': height, 'types': types})
    return stats


def sequence_divergence(codes_a, codes_b):
    """Measure how different two code sequences are."""
    # Token-level agreement
    agree = (codes_a == codes_b).float().mean(dim=-1)  # per sample
    return 1.0 - agree.mean().item()  # divergence: 0 = identical, 1 = fully different


def test_a_condition_swap(ar, vqvae, grid_size, device):
    """Test A: Same seed, swap conditions → should produce different outputs."""
    print("\n" + "=" * 70)
    print("TEST A: Condition Swap (same seed, different conditions)")
    print("=" * 70)

    vocabs = ARTransformer3D.STRUCT_FEATURE_VOCABS
    uncond = [v for v in vocabs]  # unconditional tokens
    seed = 42
    n = 4

    # Define contrasting condition pairs
    pairs = [
        ("height=flat", [0] + uncond[1:],
         "height=tower", [3] + uncond[1:]),
        ("size=tiny", [uncond[0], 0] + uncond[2:],
         "size=large", [uncond[0], 3] + uncond[2:]),
        ("symmetric", uncond[:3] + [1] + uncond[4:],
         "asymmetric", uncond[:3] + [0] + uncond[4:]),
        ("enclosed", uncond[:4] + [1] + uncond[5:],
         "open", uncond[:4] + [0] + uncond[5:]),
        ("all_small_flat", [0, 0, 0, 0, 0, 0],
         "all_large_tall", [3, 3, 2, 1, 1, 2]),
    ]

    results = []
    for name_a, feat_a, name_b, feat_b in pairs:
        feat_tensor_a = torch.tensor([feat_a] * n, dtype=torch.long, device=device)
        feat_tensor_b = torch.tensor([feat_b] * n, dtype=torch.long, device=device)

        codes_a = generate_with_seed(ar, feat_tensor_a, seed, n, device=device)
        codes_b = generate_with_seed(ar, feat_tensor_b, seed, n, device=device)

        div = sequence_divergence(codes_a, codes_b)

        voxels_a = codes_to_voxels(vqvae, codes_a, grid_size)
        voxels_b = codes_to_voxels(vqvae, codes_b, grid_size)
        stats_a = measure_voxel_stats(voxels_a)
        stats_b = measure_voxel_stats(voxels_b)

        avg_h_a = np.mean([s['height'] for s in stats_a])
        avg_h_b = np.mean([s['height'] for s in stats_b])
        avg_blk_a = np.mean([s['blocks'] for s in stats_a])
        avg_blk_b = np.mean([s['blocks'] for s in stats_b])

        status = "ACTIVE" if div > 0.05 else "IGNORED"
        results.append({
            'pair': f"{name_a} vs {name_b}",
            'divergence': div,
            'status': status,
            'avg_height': (avg_h_a, avg_h_b),
            'avg_blocks': (avg_blk_a, avg_blk_b),
        })

        print(f"\n  {name_a} vs {name_b}:")
        print(f"    Token divergence: {div:.4f} [{status}]")
        print(f"    Avg height: {avg_h_a:.1f} vs {avg_h_b:.1f}")
        print(f"    Avg blocks: {avg_blk_a:.0f} vs {avg_blk_b:.0f}")

    return results


def test_b_condition_drop(ar, vqvae, grid_size, device):
    """Test B: Conditioned vs fully unconditional → should differ."""
    print("\n" + "=" * 70)
    print("TEST B: Condition Drop (conditioned vs unconditional)")
    print("=" * 70)

    vocabs = ARTransformer3D.STRUCT_FEATURE_VOCABS
    uncond = [v for v in vocabs]
    seed = 123
    n = 4

    conditions = [
        ("tall_large_symmetric", [3, 3, 2, 1, 1, 2]),
        ("flat_tiny_blocky", [0, 0, 0, 0, 0, 0]),
        ("medium_enclosed_detailed", [1, 2, 1, 0, 1, 2]),
    ]

    results = []
    uncond_tensor = torch.tensor([uncond] * n, dtype=torch.long, device=device)
    codes_uncond = generate_with_seed(ar, uncond_tensor, seed, n, device=device)

    for name, feat in conditions:
        feat_tensor = torch.tensor([feat] * n, dtype=torch.long, device=device)
        codes_cond = generate_with_seed(ar, feat_tensor, seed, n, device=device)

        div = sequence_divergence(codes_cond, codes_uncond)
        status = "ACTIVE" if div > 0.05 else "IGNORED"

        voxels_cond = codes_to_voxels(vqvae, codes_cond, grid_size)
        stats = measure_voxel_stats(voxels_cond)
        avg_h = np.mean([s['height'] for s in stats])
        avg_blk = np.mean([s['blocks'] for s in stats])

        results.append({'condition': name, 'divergence': div, 'status': status})
        print(f"\n  {name} vs unconditional:")
        print(f"    Token divergence: {div:.4f} [{status}]")
        print(f"    Conditioned avg height: {avg_h:.1f}, blocks: {avg_blk:.0f}")

    return results


def test_c_gradient_attribution(ar, device):
    """Test C: Does gradient flow from loss through conditioning embeddings?"""
    print("\n" + "=" * 70)
    print("TEST C: Gradient Attribution (loss → conditioning path)")
    print("=" * 70)

    vocabs = ARTransformer3D.STRUCT_FEATURE_VOCABS
    n = 4
    seq_len = ar.max_seq_len

    # Create dummy input
    codes = torch.randint(0, ar.num_codes, (n, seq_len), device=device)
    features = torch.tensor([[2, 2, 1, 1, 1, 1]] * n, dtype=torch.long, device=device)

    bos = torch.full((n, 1), ar.bos_token_id, dtype=torch.long, device=device)
    input_seq = torch.cat([bos, codes[:, :-1]], dim=1)

    # Enable grad on conditioning embeddings
    ar.train()
    for p in ar.parameters():
        p.requires_grad_(True)

    logits = ar(input_seq, struct_features=features)
    loss = F.cross_entropy(logits.reshape(-1, ar.num_codes), codes.reshape(-1))
    loss.backward()

    # Check gradient magnitudes on conditioning layers vs backbone
    struct_grad_norm = 0
    struct_param_count = 0
    backbone_grad_norm = 0
    backbone_param_count = 0

    for name, p in ar.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            if 'struct_' in name:
                struct_grad_norm += grad_norm
                struct_param_count += 1
            else:
                backbone_grad_norm += grad_norm
                backbone_param_count += 1

    avg_struct_grad = struct_grad_norm / max(struct_param_count, 1)
    avg_backbone_grad = backbone_grad_norm / max(backbone_param_count, 1)
    ratio = avg_struct_grad / max(avg_backbone_grad, 1e-10)

    print(f"  Struct layers avg grad norm:   {avg_struct_grad:.6f} ({struct_param_count} params)")
    print(f"  Backbone layers avg grad norm: {avg_backbone_grad:.6f} ({backbone_param_count} params)")
    print(f"  Ratio (struct/backbone):       {ratio:.4f}")
    print(f"  Status: {'GRADIENT FLOWING' if ratio > 0.01 else 'GRADIENT VANISHING'}")

    ar.eval()
    for p in ar.parameters():
        p.requires_grad_(False)

    return {
        'struct_grad': avg_struct_grad,
        'backbone_grad': avg_backbone_grad,
        'ratio': ratio,
        'status': 'flowing' if ratio > 0.01 else 'vanishing',
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vqvae_ckpt = str(PROJECT_ROOT / "checkpoints" / "vqvae" / "vqvae_step100000.pt")

    # Find latest conditioned AR checkpoint
    ar_cond_dir = PROJECT_ROOT / "checkpoints" / "ar_cond"
    ckpts = sorted(ar_cond_dir.glob("ar_cond_step*.pt"),
                   key=lambda p: int(p.stem.split('step')[-1]))
    if not ckpts:
        print("No conditioned AR checkpoints found!")
        return
    ar_ckpt = str(ckpts[-1])
    print(f"Testing checkpoint: {ar_ckpt}")

    vqvae, ar, grid_size = load_models(vqvae_ckpt, ar_ckpt, device)

    # Run all tests
    results_a = test_a_condition_swap(ar, vqvae, grid_size, device)
    results_b = test_b_condition_drop(ar, vqvae, grid_size, device)
    results_c = test_c_gradient_attribution(ar, device)

    # Summary verdict
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    swap_divs = [r['divergence'] for r in results_a]
    drop_divs = [r['divergence'] for r in results_b]
    avg_swap_div = np.mean(swap_divs)
    avg_drop_div = np.mean(drop_divs)

    print(f"  Avg swap divergence:  {avg_swap_div:.4f} (>0.05 = conditioning active)")
    print(f"  Avg drop divergence:  {avg_drop_div:.4f} (>0.05 = conditioning active)")
    print(f"  Gradient ratio:       {results_c['ratio']:.4f} (>0.01 = gradients flowing)")

    if avg_swap_div < 0.02 and avg_drop_div < 0.02:
        print(f"\n  [FAIL] VERDICT: Conditioning is being IGNORED")
        print(f"  -> Model has collapsed to unconditional mode")
        print(f"  -> Recommend: increase conditioning strength or reduce backbone")
    elif avg_swap_div < 0.05:
        print(f"\n  [WARN] VERDICT: Conditioning is WEAK")
        print(f"  -> Some signal but not enough for controllable generation")
        print(f"  -> May improve with more training, but consider architecture changes")
    else:
        print(f"\n  [PASS] VERDICT: Conditioning is ACTIVE")
        print(f"  -> Model is using structural constraints in generation")

    # Save results
    output = {
        'checkpoint': ar_ckpt,
        'test_a_swap': results_a,
        'test_b_drop': results_b,
        'test_c_gradient': results_c,
        'summary': {
            'avg_swap_divergence': avg_swap_div,
            'avg_drop_divergence': avg_drop_div,
            'gradient_ratio': results_c['ratio'],
        },
    }
    out_path = PROJECT_ROOT / "outputs" / "conditioning_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
