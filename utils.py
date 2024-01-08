import json
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten

from models import LoRALinear


def apply_lora_to_all_layers(model):
    linear_replacements = {}
    for name, module in model.named_modules():
        if name == "lm_head":
            continue
        if isinstance(module, nn.Linear) or isinstance(module, nn.QuantizedLinear):
            replacement_module = LoRALinear.from_linear(module)
            linear_replacements[name] = replacement_module

    model.update_modules(tree_unflatten(list(linear_replacements.items())))


def merge_lora(model):
    linear_replacements = {}
    for name, module in model.named_modules():
        # dequantize the lm_head layer seems cause a lot of performance degradation, should avoid quantizing lm_head layer
        if name == "lm_head":
            if isinstance(module, nn.QuantizedLinear):
                weight = mx.dequantize(
                    module.weight,
                    module.scales,
                    module.biases,
                    module.group_size,
                    module.bits,
                )
                output_dims, input_dims = weight.shape
                new_linear = nn.Linear(input_dims, output_dims, bias=False)
                new_linear.weight = weight
                mx.eval(new_linear.weight)
                linear_replacements[name] = new_linear 

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


def prepare_model_for_export(
    model,
    model_path,
    tokenizer,
    export_path,
):
    mlx_path = Path(export_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

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
