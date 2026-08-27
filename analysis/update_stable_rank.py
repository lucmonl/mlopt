"""
Compute the stable rank of the overall parameter update for three methods:
  - LoRA:     stable_rank(B @ A)
  - DION:     stable_rank(new_param - old_param)
  - MuonLoRA: stable_rank(new_param + (lora_alpha/lora_r) * B @ A - old_param)

Usage:
  python analysis/update_stable_rank.py --method lora --checkpoint_path <path_to_checkpoint>
  python analysis/update_stable_rank.py --method dion --checkpoint_path <path_to_checkpoint>
  python analysis/update_stable_rank.py --method muonlora --checkpoint_path <path_to_checkpoint>

  Optional:
    --base_model   : HF model name (default: meta-llama/Llama-3.2-3B)
    --dtype        : torch dtype for loading (default: bf16)
    --lora_rank    : LoRA rank (default: 16)
    --lora_alpha   : LoRA alpha (default: 16)
    --target_modules : comma-separated target modules (default: q_proj,v_proj,k_proj,o_proj)
    --server_name  : adapter name used during training (default: server)
"""

import argparse
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.rank import compute_stable_rank


def get_dtype(dtype_str):
    if dtype_str in ["bf16", "default"]:
        return torch.bfloat16
    elif dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def compute_update_stable_rank_lora(checkpoint_path, base_model, dtype, lora_rank, lora_alpha, target_modules, server_name):
    """LoRA: stable_rank(B @ A) for each target layer."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, LoraModel

    torch_dtype = get_dtype(dtype)

    # Load checkpoint state dict
    state_dict = torch.load(os.path.join(checkpoint_path, "model.ckpt"), map_location="cpu", weights_only=True)

    # Extract lora_A and lora_B pairs
    lora_A_keys = sorted([k for k in state_dict if f"lora_A.{server_name}" in k])
    results = []

    for a_key in lora_A_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in state_dict:
            continue
        A = state_dict[a_key].float()  # [r, in_features]
        B = state_dict[b_key].float()  # [out_features, r]
        W_update = (lora_alpha / lora_rank) * (B @ A)
        sr = compute_stable_rank(W_update)
        layer_name = a_key.replace(f".lora_A.{server_name}.weight", "")
        results.append((layer_name, sr, W_update.shape))
        print(f"  {layer_name:60s}  shape={tuple(W_update.shape)}  stable_rank={sr:.4f}")

    avg_sr = sum(r[1] for r in results) / len(results) if results else 0
    print(f"\n  Average stable rank: {avg_sr:.4f}")
    return results


def compute_update_stable_rank_dion(checkpoint_path, base_model, dtype, target_modules):
    """DION: stable_rank(new_param - old_param) for each target layer."""
    from transformers import AutoModelForCausalLM

    torch_dtype = get_dtype(dtype)

    # Load base model (original weights)
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map="cpu")
    base_state = base.state_dict()

    # Load checkpoint state dict
    ckpt_state = torch.load(os.path.join(checkpoint_path, "model.ckpt"), map_location="cpu", weights_only=True)

    results = []
    for name in sorted(base_state.keys()):
        # Only look at 2D weight matrices in target modules
        if not any(t in name for t in target_modules):
            continue
        if "weight" not in name or base_state[name].dim() != 2:
            continue
        if name not in ckpt_state:
            continue

        old_W = base_state[name].float()
        new_W = ckpt_state[name].float()
        delta = new_W - old_W

        if delta.norm() < 1e-10:
            continue

        sr = compute_stable_rank(delta)
        results.append((name, sr, delta.shape))
        print(f"  {name:60s}  shape={tuple(delta.shape)}  stable_rank={sr:.4f}")

    avg_sr = sum(r[1] for r in results) / len(results) if results else 0
    print(f"\n  Average stable rank: {avg_sr:.4f}")
    return results


def compute_update_stable_rank_muonlora(checkpoint_path, base_model, dtype, lora_rank, lora_alpha, target_modules, server_name):
    """MuonLoRA: stable_rank(new_base_param + (alpha/r)*B@A - old_base_param) for each target layer."""
    from transformers import AutoModelForCausalLM

    torch_dtype = get_dtype(dtype)

    # Load base model (original weights)
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype, device_map="cpu")
    base_state = base.state_dict()

    # Load checkpoint state dict
    ckpt_state = torch.load(os.path.join(checkpoint_path, "model.ckpt"), map_location="cpu", weights_only=True)

    # Build mapping: base_layer weight name -> (lora_A_key, lora_B_key)
    lora_A_keys = sorted([k for k in ckpt_state if f"lora_A.{server_name}" in k])

    results = []
    for a_key in lora_A_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in ckpt_state:
            continue

        # Derive the base_layer weight name in the checkpoint
        base_layer_key = a_key.replace(f"lora_A.{server_name}.weight", "base_layer.weight")
        if base_layer_key not in ckpt_state:
            continue

        # Derive the original model weight name (without peft wrapper)
        # The checkpoint key may look like:
        #   "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight" (PeftModel)
        #   "model.model.layers.0.self_attn.q_proj.base_layer.weight" (LoraModel)
        # The base model key is:
        #   "model.layers.0.self_attn.q_proj.weight"
        # Strategy: remove ".base_layer", then strip known prefixes until we find a match
        orig_name = base_layer_key.replace(".base_layer.", ".")
        candidates = [
            orig_name,
            orig_name.removeprefix("base_model.model."),
            orig_name.removeprefix("base_model."),
            orig_name.removeprefix("model."),
        ]
        matched = False
        for candidate in candidates:
            if candidate in base_state:
                orig_name = candidate
                matched = True
                break
        if not matched:
            print(f"  WARNING: cannot find original weight for {base_layer_key}")
            continue

        old_W = base_state[orig_name].float()
        new_base_W = ckpt_state[base_layer_key].float()

        A = ckpt_state[a_key].float()  # [r, in_features]
        B = ckpt_state[b_key].float()  # [out_features, r]
        lora_update = (lora_alpha / lora_rank) * (B @ A)

        total_update = new_base_W + lora_update - old_W
        if total_update.norm() < 1e-10:
            continue

        sr = compute_stable_rank(total_update)
        layer_name = a_key.replace(f".lora_A.{server_name}.weight", "")
        results.append((layer_name, sr, total_update.shape))
        print(f"  {layer_name:60s}  shape={tuple(total_update.shape)}  stable_rank={sr:.4f}")

    avg_sr = sum(r[1] for r in results) / len(results) if results else 0
    print(f"\n  Average stable rank: {avg_sr:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute stable rank of parameter updates")
    parser.add_argument("--method", type=str, required=True, choices=["lora", "dion", "muonlora"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj")
    parser.add_argument("--server_name", type=str, default="server")
    args = parser.parse_args()

    target_modules = args.target_modules.split(",")

    print(f"Method: {args.method}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print()

    if args.method == "lora":
        compute_update_stable_rank_lora(
            args.checkpoint_path, args.base_model, args.dtype,
            args.lora_rank, args.lora_alpha, target_modules, args.server_name
        )
    elif args.method == "dion":
        compute_update_stable_rank_dion(
            args.checkpoint_path, args.base_model, args.dtype, target_modules
        )
    elif args.method == "muonlora":
        compute_update_stable_rank_muonlora(
            args.checkpoint_path, args.base_model, args.dtype,
            args.lora_rank, args.lora_alpha, target_modules, args.server_name
        )


if __name__ == "__main__":
    main()
