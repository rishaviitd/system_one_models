#!/usr/bin/env python3
"""Run the vendored Kev Banking77 LoRA trainer from finetune.yaml."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "finetune.yaml"
VENDOR = ROOT / "vendor"


def resolve(value):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", help="override model.init_from or KEV_INIT_FROM")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths, model = cfg["paths"], cfg["model"]
    env = os.environ.copy()
    env["HF_HUB_CACHE"] = str(resolve(paths["hf_hub_cache"]))
    env["PYTHONPATH"] = str(VENDOR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    train = cfg["training"]
    if train["method"].lower() != "lora":
        parser.error("This runner currently supports LoRA training only.")
    init_from = resolve(args.init_from or env.get("KEV_INIT_FROM") or model["init_from"])
    command = [
        sys.executable, "-m", "kev.train",
        "--data", str(resolve(paths["train_data"])),
        "--base", model["base"], "--base_revision", str(model["base_revision"]),
        "--init_from", str(init_from), "--epochs", str(train["epochs"]),
        "--lr", str(train["learning_rate"]), "--lora", str(train["lora_rank"]),
        "--batch", str(train["batch_size"]),
        "--accum", str(train["gradient_accumulation_steps"]),
        "--dtype", train["precision"], "--weights_dtype", train["weights_precision"],
        "--checkpointing", str(int(train["gradient_checkpointing"])),
        "--device", train["device"], "--full_ft", "0", "--seed", str(train["seed"]),
        "--epoch_checkpoint_dir", str(resolve(paths["epoch_checkpoint_dir"])),
        "--out", str(resolve(paths["run_output"])),
    ]
    required = [resolve(paths["train_data"]), init_from]

    if not args.dry_run:
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Required input not found: " + ", ".join(missing))
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
