"""
Overnight training pipeline v2 — improved architecture.

Key changes from v1:
  - VQ-VAE: 8³ latent grid (was 4³) — 8x more spatial detail
  - VQ-VAE: hidden_dim 256 (was 128) — more capacity
  - AR: dim 512, 12 layers (was dim 256, 8 layers) — stronger generator
  - AR: 512-token sequences (was 64) — matches 8³ latent

Total: ~6h, fits in overnight window.
"""

import os
import subprocess
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = sys.executable


def run_phase(name, cmd, timeout_hours=None):
    """Run a training phase and log output."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print(f"\n{'='*80}")
    print(f"PHASE: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    start = time.time()
    timeout_sec = int(timeout_hours * 3600) if timeout_hours else None

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    with open(log_path, 'w', encoding='utf-8') as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()

                if timeout_sec and (time.time() - start) > timeout_sec:
                    print(f"\n*** TIMEOUT after {timeout_hours}h ***")
                    proc.terminate()
                    break

            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        except KeyboardInterrupt:
            proc.terminate()
            raise

    elapsed = time.time() - start
    status = "SUCCESS" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
    print(f"\n{name}: {status} in {elapsed/60:.1f} minutes")
    return proc.returncode == 0


def find_latest_checkpoint(ckpt_dir, prefix):
    """Find the checkpoint with the highest step number."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    ckpts = list(ckpt_dir.glob(f"{prefix}*.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.stem.split('step')[-1]))
    return str(ckpts[-1])


def main():
    start_time = time.time()
    print(f"Overnight v2 pipeline started at {datetime.now()}")
    print(f"Project root: {PROJECT_ROOT}")

    # Archive v1 checkpoints
    v1_vqvae = PROJECT_ROOT / "checkpoints" / "vqvae_v1"
    v1_ar = PROJECT_ROOT / "checkpoints" / "ar_v1"
    vqvae_dir = PROJECT_ROOT / "checkpoints" / "vqvae"
    ar_dir = PROJECT_ROOT / "checkpoints" / "ar"

    if vqvae_dir.exists() and not v1_vqvae.exists():
        print(f"Archiving v1 VQ-VAE checkpoints to {v1_vqvae}")
        shutil.copytree(vqvae_dir, v1_vqvae)
        # Clear vqvae dir for v2
        for f in vqvae_dir.glob("*.pt"):
            f.unlink()
        for f in vqvae_dir.glob("*.json"):
            f.unlink()

    if ar_dir.exists() and not v1_ar.exists():
        print(f"Archiving v1 AR checkpoints to {v1_ar}")
        shutil.copytree(ar_dir, v1_ar)
        for f in ar_dir.glob("*.pt"):
            f.unlink()

    # ================================================================
    # Phase A: Train VQ-VAE v2 (8³ latent, ~2h)
    # ================================================================
    vqvae_resume = find_latest_checkpoint(vqvae_dir, "vqvae_step")
    vqvae_cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "train_vqvae.py"),
        "--steps", "100000",
        "--batch_size", "8",
        "--lr", "3e-4",
        "--num_codes", "2048",
        "--code_dim", "256",
        "--hidden_dim", "256",      # v2: doubled from 128
        "--embed_dim", "32",
        "--save_every", "10000",
        "--air_weight", "0.05",
        "--n_downsample", "2",      # v2: 8³ latent (was 3 for 4³)
    ]
    if vqvae_resume:
        vqvae_cmd.extend(["--resume", vqvae_resume])
        print(f"Resuming VQ-VAE from {vqvae_resume}")

    ok = run_phase("vqvae_v2_train", vqvae_cmd, timeout_hours=3.0)

    if not ok:
        print("*** VQ-VAE v2 training failed! Aborting. ***")
        return

    # ================================================================
    # Phase B: Train AR Transformer v2 (512 tokens, ~3.5h)
    # ================================================================
    ok = run_phase("ar_v2_train", [
        PYTHON, str(PROJECT_ROOT / "scripts" / "train_ar.py"),
        "--steps", "60000",
        "--batch_size", "32",       # v2: smaller batch for 512 tokens
        "--lr", "2e-4",             # v2: slightly lower lr for larger model
        "--dim", "512",             # v2: doubled from 256
        "--n_layers", "12",         # v2: 12 from 8
        "--n_heads", "8",
        "--dropout", "0.1",
        "--save_every", "5000",
    ], timeout_hours=5.0)

    if not ok:
        print("*** AR v2 training failed! Attempting generation with best checkpoint. ***")

    # ================================================================
    # Phase C: Generate samples
    # ================================================================
    vqvae_ckpt = find_latest_checkpoint(vqvae_dir, "vqvae_step")
    ar_ckpt = find_latest_checkpoint(ar_dir, "ar_step")

    if vqvae_ckpt and ar_ckpt:
        run_phase("generate_v2", [
            PYTHON, str(PROJECT_ROOT / "scripts" / "generate.py"),
            "--vqvae_ckpt", vqvae_ckpt,
            "--ar_ckpt", ar_ckpt,
            "--n_samples", "50",
            "--temperature", "0.9",
            "--top_k", "100",
        ], timeout_hours=1.0)
    else:
        print(f"*** Cannot generate: vqvae={vqvae_ckpt}, ar={ar_ckpt} ***")

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"V2 PIPELINE COMPLETE")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*80}")

    summary = {
        'version': 'v2',
        'total_time_hours': total_time / 3600,
        'finished_at': datetime.now().isoformat(),
        'vqvae_checkpoint': vqvae_ckpt,
        'ar_checkpoint': ar_ckpt,
        'architecture': {
            'vqvae': 'n_downsample=2, hidden=256, 8³ latent',
            'ar': 'dim=512, 12 layers, 512 tokens',
        },
        'generated_samples': str(PROJECT_ROOT / "outputs" / "generated"),
    }
    with open(PROJECT_ROOT / "logs" / "overnight_v2_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
