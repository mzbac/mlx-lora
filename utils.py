import mlx.nn as nn
from mlx.utils import tree_unflatten

from models import LoRALinear


def apply_lora_to_all_layers(model):
    linear_replacements = {}
    for name, module in model.named_modules():
        if name == "lm_head":
            continue
        if isinstance(module, nn.Linear):
            replacement_module = LoRALinear.from_linear(module)
            linear_replacements[name] = replacement_module

    model.update_modules(tree_unflatten(list(linear_replacements.items())))
