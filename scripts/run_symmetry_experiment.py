"""
Symmetry augmentation experiment — minimal intervention test.

Hypothesis: symmetry controllability failure is FREQUENCY-limited, not representation-limited.
Intervention: augment training data with symmetrized copies (50% symmetric vs original 4%).
Control: enclosure remains at 2% — if it's still 0%, that's a representation limit.

Phases:
  A: Create symmetry-augmented latent codes (CPU, ~1 min)
  B: Train conditioned AR on augmented data (GPU, ~4h)
  C: Run composability + OOD tests (GPU, ~5 min)
  D: Compare with non-augmented baseline
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
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print(f"\n{'='*80}")
    print(f"PHASE: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    start = time.time()
    timeout_sec = int(timeout_hours * 3600) if timeout_hours else None

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    with open(log_path, 'w', encoding='utf-8') as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace',
            cwd=str(PROJECT_ROOT), env=env,
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
    print(f"Symmetry augmentation experiment started at {datetime.now()}")

    # ================================================================
    # Phase A: Create augmented data
    # ================================================================
    aug_latent = PROJECT_ROOT / "data" / "processed" / "latent_codes_sym_aug.npz"
    aug_features = PROJECT_ROOT / "data" / "processed" / "structural_features_sym_aug.json"

    if aug_latent.exists() and aug_features.exists():
        print(f"Augmented data already exists, skipping Phase A")
    else:
        ok = run_phase("augment_symmetry", [
            PYTHON, str(PROJECT_ROOT / "scripts" / "augment_symmetry.py"),
        ], timeout_hours=0.1)
        if not ok:
            print("*** Augmentation failed! ***")
            return

    # ================================================================
    # Phase B: Train conditioned AR on augmented data
    # Same hyperparams as v3, same starting point (v2 AR checkpoint)
    # ONLY difference: augmented training data
    # ================================================================
    v2_ar_ckpt = find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "ar", "ar_step"
    )

    # Check for existing sym_aug checkpoints to resume
    sym_resume = find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "ar_cond_sym_aug", "ar_cond_step"
    )

    train_cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "train_ar_conditioned.py"),
        "--steps", "80000",
        "--batch_size", "32",
        "--lr", "2e-4",
        "--dim", "512",
        "--n_layers", "12",
        "--n_heads", "8",
        "--dropout", "0.1",
        "--save_every", "10000",
        "--uncond_prob", "0.1",
        "--latent_path", str(aug_latent),
        "--features_path", str(aug_features),
        "--ckpt_dir", "ar_cond_sym_aug",
    ]

    if sym_resume:
        train_cmd.extend(["--resume", sym_resume,
                          "--resume_step", str(int(Path(sym_resume).stem.split('step')[-1]))])
        print(f"Resuming sym_aug training from {sym_resume}")
    elif v2_ar_ckpt:
        train_cmd.extend(["--resume", v2_ar_ckpt])
        print(f"Transfer learning from v2 AR: {v2_ar_ckpt}")

    ok = run_phase("ar_sym_aug_train", train_cmd, timeout_hours=6.0)

    # ================================================================
    # Phase C: Run diagnostic tests
    # ================================================================
    sym_ckpt = find_latest_checkpoint(
        PROJECT_ROOT / "checkpoints" / "ar_cond_sym_aug", "ar_cond_step"
    )
    if sym_ckpt:
        # Temporarily patch the test script's checkpoint discovery
        # by using the --ar_ckpt override (need to pass it differently)
        # For now, run a quick inline test
        run_phase("test_sym_aug", [
            PYTHON, "-c", f"""
import sys, json
sys.path.insert(0, '{PROJECT_ROOT}')
from scripts.test_composability_ood import *

device = 'cuda'
vqvae_ckpt = '{PROJECT_ROOT / "checkpoints" / "vqvae" / "vqvae_step100000.pt"}'
ar_ckpt = '{sym_ckpt}'
print(f"Testing sym_aug checkpoint: {{ar_ckpt}}")
vqvae, ar, grid_size = load_models(vqvae_ckpt, ar_ckpt, device)
results_1 = experiment_1_composability(ar, vqvae, grid_size, device)
results_2 = experiment_2_ood_generalization(ar, vqvae, grid_size, device)
output = {{
    'checkpoint': ar_ckpt,
    'experiment': 'symmetry_augmentation',
    'composability': results_1,
    'ood_generalization': [
        {{k: v for k, v in e.items() if k != 'measurements'}}
        for e in results_2
    ],
}}
out_path = '{PROJECT_ROOT / "outputs" / "sym_aug_composability_ood.json"}'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"Results saved to {{out_path}}")
""",
        ], timeout_hours=0.5)

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"SYMMETRY EXPERIMENT COMPLETE")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"{'='*80}")

    # Load and compare results if both exist
    baseline_path = PROJECT_ROOT / "outputs" / "composability_ood_results.json"
    aug_path = PROJECT_ROOT / "outputs" / "sym_aug_composability_ood.json"
    if baseline_path.exists() and aug_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        with open(aug_path) as f:
            aug = json.load(f)
        print(f"\n{'='*80}")
        print(f"COMPARISON: Baseline vs Symmetry-Augmented")
        print(f"{'='*80}")
        # Extract symmetry satisfaction from OOD results
        for exp_set, label in [(baseline, "Baseline"), (aug, "Sym-Aug")]:
            print(f"\n  {label}:")
            for e in exp_set.get('ood_generalization', []):
                print(f"    {e['name']}: satisfaction={e.get('avg_satisfaction', 'N/A')}")


if __name__ == "__main__":
    main()
