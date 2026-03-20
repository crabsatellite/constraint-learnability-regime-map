"""
Master script: re-run regime map with cfg_scale=2.0, bootstrap CIs, visualizations.

Fixes the cfg_scale=0.0 bug (enclosed_ratio discrepancy), adds confidence
intervals, and generates paper figures. Run on GPU machine with checkpoints.

Usage:
    python scripts/run_fixes.py                  # all steps
    python scripts/run_fixes.py --step regime     # regime map only
    python scripts/run_fixes.py --step bootstrap  # bootstrap only (uses existing results)
    python scripts/run_fixes.py --step figures     # figures only
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def check_checkpoints():
    vqvae = PROJECT_ROOT / "checkpoints" / "vqvae" / "vqvae_step100000.pt"
    ar = PROJECT_ROOT / "checkpoints" / "ar_cond" / "ar_cond_step80000.pt"
    if not vqvae.exists() or not ar.exists():
        print("ERROR: Checkpoints not found.")
        print(f"  Expected: {vqvae}")
        print(f"  Expected: {ar}")
        print("\nCopy checkpoints from training machine first.")
        return False
    return True


def run_step(script_name, description):
    print(f"\n{'=' * 60}")
    print(f"STEP: {description}")
    print(f"{'=' * 60}")
    script = PROJECT_ROOT / "scripts" / script_name
    result = subprocess.run([sys.executable, str(script)],
                            cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"FAILED: {script_name} (exit code {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['regime', 'bootstrap', 'figures', 'predictor'],
                        help='Run only this step')
    args = parser.parse_args()

    steps = {
        'regime': ('test_regime_map.py', 'Re-run regime map with cfg_scale=2.0'),
        'predictor': ('regime_predictor.py', 'Re-fit predictor with updated regime data'),
        'bootstrap': ('bootstrap_regime_map.py', 'Compute bootstrap 95% CIs'),
        'figures': ('visualize_structures.py', 'Generate paper figures'),
    }

    if args.step:
        if args.step in ('regime', 'figures') and not check_checkpoints():
            return
        run_step(*steps[args.step])
    else:
        if not check_checkpoints():
            return
        for step_name, (script, desc) in steps.items():
            if not run_step(script, desc):
                print(f"\nStopped at step: {step_name}")
                return

    print(f"\n{'=' * 60}")
    print("ALL DONE")
    print(f"{'=' * 60}")
    print("\nOutputs:")
    print(f"  Regime map:  {PROJECT_ROOT / 'outputs' / 'regime_map_results.json'}")
    print(f"  Predictor:   {PROJECT_ROOT / 'outputs' / 'regime_predictor_results.json'}")
    print(f"  Bootstrap:   {PROJECT_ROOT / 'outputs' / 'bootstrap_results.json'}")
    print(f"  Figures:     {PROJECT_ROOT / 'figures' / '*.png'}")


if __name__ == "__main__":
    main()
