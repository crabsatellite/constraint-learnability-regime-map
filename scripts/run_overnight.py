"""
Overnight training orchestrator.

Runs the full pipeline:
  Phase A: Train VQ-VAE (~2.5h) + encode all builds
  Phase B: Train AR Transformer (~5h)
  Phase C: Generate samples (~30min)

Total: ~8 hours, fits in a 9-hour sleep window.
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

                # Check timeout
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
    # Sort by step number
    ckpts.sort(key=lambda p: int(p.stem.split('step')[-1]))
    return str(ckpts[-1])


def main():
    start_time = time.time()
    print(f"Overnight training pipeline started at {datetime.now()}")
    print(f"Project root: {PROJECT_ROOT}")

    # Phase A: Train VQ-VAE (4³ latent)
    # Check for existing checkpoint to resume from
    vqvae_resume = find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "vqvae", "vqvae_step"
    )
    vqvae_cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "train_vqvae.py"),
        "--steps", "100000",
        "--batch_size", "8",
        "--lr", "3e-4",
        "--num_codes", "2048",
        "--code_dim", "256",
        "--hidden_dim", "128",
        "--embed_dim", "32",
        "--save_every", "10000",
        "--air_weight", "0.05",
    ]
    if vqvae_resume:
        vqvae_cmd.extend(["--resume", vqvae_resume])
        print(f"Resuming VQ-VAE from {vqvae_resume}")
    ok = run_phase("vqvae_train", vqvae_cmd, timeout_hours=4.0)

    if not ok:
        print("*** VQ-VAE training failed! Aborting pipeline. ***")
        return

    # Phase B: Train AR Transformer (64 tokens, ~4.6h at 2.4 it/s)
    ok = run_phase("ar_train", [
        PYTHON, str(PROJECT_ROOT / "scripts" / "train_ar.py"),
        "--steps", "40000",
        "--batch_size", "128",
        "--lr", "3e-4",
        "--dim", "256",
        "--n_layers", "8",
        "--n_heads", "8",
        "--dropout", "0.1",
        "--save_every", "5000",
    ], timeout_hours=5.5)

    if not ok:
        print("*** AR training failed! Attempting generation with best checkpoint. ***")

    # Phase C: Generate samples
    vqvae_ckpt = find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "vqvae", "vqvae_step"
    )
    ar_ckpt = find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "ar", "ar_step"
    )

    if vqvae_ckpt and ar_ckpt:
        run_phase("generate", [
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
    print(f"PIPELINE COMPLETE")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*80}")

    # Write summary file
    summary = {
        'total_time_hours': total_time / 3600,
        'finished_at': datetime.now().isoformat(),
        'vqvae_checkpoint': vqvae_ckpt,
        'ar_checkpoint': ar_ckpt,
        'generated_samples': str(PROJECT_ROOT / "outputs" / "generated"),
    }
    with open(PROJECT_ROOT / "logs" / "overnight_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
