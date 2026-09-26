#!/usr/bin/env python3
"""Benchmark a Banking77 Kev checkpoint on the full test split."""
import argparse
import json
import os
import shlex
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
    parser.add_argument("--checkpoint", help="checkpoint to evaluate; defaults to the final training output")
    parser.add_argument("--output", help="evaluation output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths, settings = cfg["paths"], cfg["benchmark"]
    test_data = resolve(paths["test_data"])
    output_root = resolve(args.output or paths["benchmark_output"])

    if args.checkpoint:
        checkpoint = resolve(args.checkpoint)
        targets = [(checkpoint.name, checkpoint, output_root)]
    else:
        epochs = resolve(paths["epoch_checkpoint_dir"])
        targets = [("base", resolve(cfg["model"]["init_from"]), output_root / "base")]
        targets.extend(
            (f"epoch_{epoch:02d}", epochs / f"epoch_{epoch:02d}", output_root / f"epoch_{epoch:02d}")
            for epoch in range(1, int(cfg["training"]["epochs"]) + 1)
        )

    if not args.dry_run:
        missing = [str(path) for _, path, _ in targets if not path.exists()]
        if not test_data.exists():
            missing.append(str(test_data))
        if missing:
            raise FileNotFoundError("Required input not found: " + ", ".join(missing))

    env = os.environ.copy()
    env["HF_HUB_CACHE"] = str(resolve(paths["hf_hub_cache"]))
    env["PYTHONPATH"] = str(VENDOR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    summary = []
    for name, checkpoint, output in targets:
        command = [
            sys.executable, "-m", "kev.benchmark",
            "--run", str(checkpoint),
            "--data", str(test_data),
            "--out", str(output),
            "--device", settings["device"],
            "--rotations", str(settings["rotations"]),
        ]
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        result = {
            "name": name,
            "checkpoint": str(checkpoint),
            "accuracy": report["clean"]["acc"],
            "nll": report["clean"]["nll"],
            "median_latency_ms": report["latency_ms"]["median"],
            "p95_latency_ms": report["latency_ms"]["p95"],
        }
        summary.append(result)
        print(json.dumps(result, indent=2), flush=True)

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
