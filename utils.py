import json
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten
from huggingface_hub import snapshot_download

from lora_linear import LoRALinear


def apply_lora_to_all_layers(model):
    linear_replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.QuantizedLinear):
            replacement_module = LoRALinear.from_linear(module)
            linear_replacements[name] = replacement_module

    model.update_modules(tree_unflatten(list(linear_replacements.items())))


def merge_lora(model):
    linear_replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            linear_replacements[name] = module.merge()

    model.update_modules(tree_unflatten(list(linear_replacements.items())))


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        estimated_size = v.size * v.dtype.size
        if shard_size + estimated_size > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += estimated_size
    shards.append(shard)
    return shards

def get_model_path(model_identifier, allow_patterns=None):
    model_path = Path(model_identifier)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=model_identifier,
                allow_patterns=allow_patterns if allow_patterns else ["*.json", "*.safetensors", "tokenizer.model"],
            )
        )
    return model_path


def prepare_model_for_export(
    model,
    model_path,
    tokenizer,
    export_path,
):
    mlx_path = Path(export_path)
    mlx_path.mkdir(parents=True, exist_ok=True)
    model_path =get_model_path(model_path)
    merge_lora(model)

    shards = make_shards(dict(tree_flatten(model.parameters())))
    for i, shard in enumerate(shards):
        mx.save_safetensors(str(mlx_path / f"weights.{i:02d}.safetensors"), shard)

    tokenizer.save_pretrained(mlx_path)

    with open(Path(model_path) / "config.json", "r") as f:
        config = json.loads(f.read())
        config.pop("quantization", None)  # merged model is not quantized
    with open(mlx_path / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

