#!/usr/bin/env python3
"""Download the pinned base model and prepare Kev's LoRA initialization."""
import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from peft.utils.save_and_load import load_peft_weights
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "finetune.yaml"


def resolve(value):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else ROOT / path


def expand_adapter(source, destination, target_rank, seed, source_label):
    config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    source_rank = int(config["r"])
    if target_rank < source_rank:
        raise ValueError(f"target rank {target_rank} is below source rank {source_rank}")

    weights = load_peft_weights(str(source), device="cpu")
    torch.manual_seed(seed)
    expanded = {}
    count_a = count_b = 0
    for name, tensor in weights.items():
        if name.endswith("lora_A.weight"):
            value = torch.empty((target_rank, *tensor.shape[1:]), dtype=tensor.dtype)
            value[:source_rank].copy_(tensor)
            torch.nn.init.kaiming_uniform_(value[source_rank:], a=math.sqrt(5))
            expanded[name] = value.contiguous()
            count_a += 1
        elif name.endswith("lora_B.weight"):
            value = torch.zeros((tensor.shape[0], target_rank), dtype=tensor.dtype)
            value[:, :source_rank].copy_(tensor)
            expanded[name] = value.contiguous()
            count_b += 1
        else:
            expanded[name] = tensor.contiguous()
    if count_a == 0 or count_a != count_b:
        raise ValueError(f"unpaired LoRA tensors: A={count_a}, B={count_b}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp_dir:
        prepared = Path(temp_dir) / destination.name
        shutil.copytree(source, prepared, ignore=shutil.ignore_patterns(".cache"))
        save_file(expanded, str(prepared / "adapter_model.safetensors"), metadata={"format": "pt"})
        config["r"] = target_rank
        config["lora_alpha"] = int(config["lora_alpha"] * target_rank / source_rank)
        (prepared / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        head_path = prepared / "head.pt"
        head = torch.load(head_path, map_location="cpu", weights_only=False)
        if int(head.get("lora", -1)) != source_rank:
            raise ValueError(f"head metadata reports rank {head.get('lora')}, expected {source_rank}")
        expansion = {
            "source_checkpoint": source_label,
            "from": source_rank,
            "to": target_rank,
            "method": "preserve existing A/B; initialize new A; zero-initialize new B",
        }
        head["lora"] = target_rank
        head["rank_expansion"] = expansion
        torch.save(head, head_path)
        (prepared / "rank_expansion.json").write_text(json.dumps(expansion, indent=2) + "\n", encoding="utf-8")
        prepared.rename(destination)

    print(f"Prepared {destination} (LoRA rank {source_rank} -> {target_rank})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths, model, training = cfg["paths"], cfg["model"], cfg["training"]
    cache = resolve(paths["hf_hub_cache"])
    destination = resolve(model["init_from"])
    source = ROOT / "models" / "kev-4b-r16"

    if args.dry_run:
        print(f"Base: {model['base']}@{model['base_revision']} -> {cache}")
        print(f"Initialization: {model['init_repo']}@{model['init_revision']} -> {destination}")
        return

    cache.mkdir(parents=True, exist_ok=True)
    base_snapshot = snapshot_download(
        repo_id=model["base"], revision=str(model["base_revision"]), cache_dir=cache
    )
    print(f"Downloaded base snapshot: {base_snapshot}")

    snapshot_download(
        repo_id=model["init_repo"], revision=str(model["init_revision"]), local_dir=source
    )
    if destination.exists():
        existing = json.loads((destination / "adapter_config.json").read_text(encoding="utf-8"))
        if int(existing.get("r", -1)) != int(training["lora_rank"]):
            raise ValueError(f"{destination} exists with unexpected rank {existing.get('r')}")
        print(f"Initialization already prepared: {destination}")
        return

    label = f"{model['init_repo']}@{model['init_revision']}"
    expand_adapter(source, destination, int(training["lora_rank"]), int(model["initialization_seed"]), label)


if __name__ == "__main__":
    main()
