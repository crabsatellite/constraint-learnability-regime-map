"""
Overnight pipeline v3 — Structural constraint conditioning.

Reuses VQ-VAE v2 (already trained). Focuses on:
  Phase A: Extract structural features from training data
  Phase B: Train conditioned AR Transformer (with CFG dropout)
  Phase C: Generate constraint experiments (16 configurations)
  Phase D: Run structural evaluation metrics

Total estimated time: ~5h
"""

import os
import subprocess
import sys
import time
import json
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
    print(f"Overnight v3 pipeline started at {datetime.now()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Strategy: Structural constraint conditioning + evaluation protocol")

    # Verify VQ-VAE v2 exists
    vqvae_ckpt = find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "vqvae", "vqvae_step"
    )
    if not vqvae_ckpt:
        print("*** ERROR: No VQ-VAE checkpoint found! Run v2 pipeline first. ***")
        return
    print(f"Using VQ-VAE: {vqvae_ckpt}")

    latent_path = PROJECT_ROOT / "data" / "processed" / "latent_codes.npz"
    if not latent_path.exists():
        print("*** ERROR: No encoded latent codes found! Run v2 pipeline first. ***")
        return
    print(f"Using latent codes: {latent_path}")

    # ================================================================
    # Phase A: Extract structural features (~10 min)
    # ================================================================
    features_path = PROJECT_ROOT / "data" / "processed" / "structural_features.json"
    if features_path.exists():
        print(f"\nStructural features already exist: {features_path}")
        with open(features_path, 'r') as f:
            feat_data = json.load(f)
        print(f"  {feat_data['total_builds']} builds with features")
    else:
        ok = run_phase("extract_features", [
            PYTHON, str(PROJECT_ROOT / "scripts" / "extract_structural_features.py"),
        ], timeout_hours=0.5)
        if not ok:
            print("*** Feature extraction failed! ***")
            return

    # ================================================================
    # Phase B: Train conditioned AR Transformer (~4h)
    # ================================================================
    # Transfer learning from v2 AR checkpoint
    v2_ar_ckpt = find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "ar", "ar_step"
    )
    ar_cond_cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "train_ar_conditioned.py"),
        "--steps", "80000",
        "--batch_size", "32",
        "--lr", "2e-4",
        "--dim", "512",
        "--n_layers", "12",
        "--n_heads", "8",
        "--dropout", "0.1",
        "--save_every", "5000",
        "--uncond_prob", "0.1",
    ]
    if v2_ar_ckpt:
        ar_cond_cmd.extend(["--resume", v2_ar_ckpt])
        print(f"Transfer learning from v2 AR: {v2_ar_ckpt}")

    # Check for existing conditioned checkpoints to resume
    cond_resume = find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "ar_cond", "ar_cond_step"
    )
    if cond_resume:
        # Replace resume arg with conditioned checkpoint
        ar_cond_cmd = [
            PYTHON, str(PROJECT_ROOT / "scripts" / "train_ar_conditioned.py"),
            "--steps", "80000",
            "--batch_size", "32",
            "--lr", "2e-4",
            "--dim", "512",
            "--n_layers", "12",
            "--n_heads", "8",
            "--dropout", "0.1",
            "--save_every", "5000",
            "--uncond_prob", "0.1",
            "--resume", cond_resume,
            "--resume_step", str(int(Path(cond_resume).stem.split('step')[-1])),
        ]
        print(f"Resuming conditioned AR from {cond_resume}")

    ok = run_phase("ar_conditioned_train", ar_cond_cmd, timeout_hours=6.0)
    if not ok:
        print("*** Conditioned AR training failed! Attempting with best checkpoint. ***")

    # ================================================================
    # Phase C: Generate constraint experiments (~30 min)
    # ================================================================
    ar_cond_ckpt = find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "ar_cond", "ar_cond_step"
    )

    if vqvae_ckpt and ar_cond_ckpt:
        ok = run_phase("generate_conditioned", [
            PYTHON, str(PROJECT_ROOT / "scripts" / "generate_conditioned.py"),
            "--vqvae_ckpt", vqvae_ckpt,
            "--ar_ckpt", ar_cond_ckpt,
            "--n_samples", "8",
            "--temperature", "0.9",
            "--top_k", "100",
            "--cfg_scale", "2.0",
        ], timeout_hours=1.0)
    else:
        print(f"*** Cannot generate: vqvae={vqvae_ckpt}, ar_cond={ar_cond_ckpt} ***")

    # ================================================================
    # Phase D: Structural evaluation (~10 min)
    # ================================================================
    eval_script = PROJECT_ROOT / "scripts" / "eval_structural.py"
    if eval_script.exists():
        # Evaluate both v2 unconditioned and v3 conditioned
        for gen_name, gen_dir in [
            ("v2_uncond", PROJECT_ROOT / "outputs" / "generated"),
            ("v3_cond_uncond", PROJECT_ROOT / "outputs" / "conditioned" / "uncond"),
            ("v3_cond_tall_sym_enc", PROJECT_ROOT / "outputs" / "conditioned" / "tall_symmetric_enclosed"),
            ("v3_cond_large_sym_det", PROJECT_ROOT / "outputs" / "conditioned" / "large_symmetric_detailed"),
        ]:
            if gen_dir.exists() and list(gen_dir.glob("*.npz")):
                run_phase(f"eval_{gen_name}", [
                    PYTHON, str(eval_script),
                    "--generated_dir", str(gen_dir),
                    "--output_dir", str(PROJECT_ROOT / "outputs" / "eval" / gen_name),
                    "--max_training_samples", "500",
                ], timeout_hours=0.5)
    else:
        print("*** eval_structural.py not found, skipping evaluation ***")

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"V3 PIPELINE COMPLETE")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*80}")

    summary = {
        'version': 'v3',
        'total_time_hours': total_time / 3600,
        'finished_at': datetime.now().isoformat(),
        'vqvae_checkpoint': vqvae_ckpt,
        'ar_cond_checkpoint': ar_cond_ckpt,
        'architecture': {
            'vqvae': 'n_downsample=2, hidden=256, 8³ latent (reused from v2)',
            'ar': 'dim=512, 12 layers, 512 tokens + 6 structural prefix tokens',
            'conditioning': '6 features: height, size, footprint, symmetry, enclosure, complexity',
            'cfg_dropout': '10%',
        },
        'outputs': {
            'conditioned_samples': str(PROJECT_ROOT / "outputs" / "conditioned"),
            'evaluation': str(PROJECT_ROOT / "outputs" / "eval"),
        },
    }
    with open(PROJECT_ROOT / "logs" / "overnight_v3_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
