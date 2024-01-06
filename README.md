# mlx-lora

## Setup

Install the dependencies:
```
pip install -r requirements.txt
```

## Training the Model
Training the Model

```
python lora.py --model <path_to_model> \
               --train \
               --iters 600
```

Here, <path_to_your_model> should be replaced with the path to your model.

For instance, to train the hf model directly, use:

```
python lora.py --model mistralai/Mistral-7B-v0.1 --train --iters 600
```

Or, to train the hf model with qlora, use:

```
python lora.py --model mlx-community/deepseek-coder-6.7b-instruct-hf-4bit-mlx --train --iters 600
```

## Preparing Your Data

The dataset for fine-tuning is located in the `data` folder. The file is named `data.jsonl``. The data format should be as follows:

```
{"text": "This is an example for the model."}
```
Please note that you need to prepare your own Q&A dataset. The dataset should be a single text sentence without any beginning of sentence (bos) or end of sentence (eos) characters.
