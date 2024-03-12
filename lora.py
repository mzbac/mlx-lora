import random
from typing import Tuple
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import TrainingArgs, train
import mlx.optimizers as optim

import json
from pathlib import Path


MAX_SEQ_LENGTH = 2048
ITERS = 10000
class Dataset:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def load_dataset(
    path: str, tokenizer, train_split: float = 0.8
) -> Tuple[Dataset, Dataset]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)

    random.shuffle(data)

    filtered_data = []
    for item in data:
        combined_text = tokenizer.apply_chat_template(item['messages'], tokenize=False)
        token_count = len(tokenizer.tokenize(combined_text))
        if token_count <= MAX_SEQ_LENGTH:
            filtered_data.append(combined_text)
            
    print(f"total data: {len(data)}, filtered data: {len(filtered_data)}")
    combined_data = filtered_data[:ITERS]

    random.shuffle(combined_data)

    split_idx = int(len(combined_data) * train_split)
    train_data = combined_data[:split_idx]
    val_data = combined_data[split_idx:]

    train_dataset = Dataset(train_data)
    val_dataset = Dataset(val_data)

    return train_dataset, val_dataset


def main():
    train_dataset_path = (
        "./data/m-a-p/Code-Feedback/Code-Feedback.jsonl"
    )

    model_path = "mlx-community/Mixtral-8x7B-Instruct-v0.1-hf-4bit-mlx"

    model, tokenizer = load(model_path)

    train_dst, valid_dst = load_dataset(train_dataset_path, tokenizer)

    model.freeze()
    for l in model.model.layers:
        l.self_attn.q_proj = LoRALinear.from_linear(
            l.self_attn.q_proj, r=128, lora_alpha=256
        )
        l.self_attn.k_proj = LoRALinear.from_linear(
            l.self_attn.k_proj, r=128, lora_alpha=256
        )

        l.block_sparse_moe.gate = LoRALinear.from_linear(
            l.block_sparse_moe.gate, r=128, lora_alpha=256
        )

        for e in l.block_sparse_moe.experts:
            e.w1 = LoRALinear.from_linear(e.w1, r=128, lora_alpha=256)
            e.w2 = LoRALinear.from_linear(e.w2, r=128, lora_alpha=256)
            e.w3 = LoRALinear.from_linear(e.w3, r=128, lora_alpha=256)

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")
   
    trainingArgs = TrainingArgs(
        batch_size=1,
        iters=ITERS,
        val_batches=25,
        steps_per_report=10,
        steps_per_eval=200,
        steps_per_save=200,
        adapter_file="adapters.npz",
        max_seq_length=MAX_SEQ_LENGTH,
    )

    model.train()
    lr_schedule = optim.cosine_decay(1e-5, ITERS)
    opt = optim.AdamW(learning_rate=lr_schedule)

    train(
        model=model,
        tokenizer=tokenizer,
        args=trainingArgs,
        optimizer=opt,
        train_dataset=train_dst,
        val_dataset=valid_dst,
    )


main()
