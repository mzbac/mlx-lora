# mlx-lora

## Setup

Install the dependencies:
```
pip install -r requirements.txt
```

## Training the Model

To train the model, you'll need to use the lora.py script with the appropriate command-line arguments. Here's an example of how to do this:

```
python lora.py --model <path_to_model> \
               --train \
               --iters 600
```

In the above command, replace <path_to_your_model> with the path to the model you want to train.

For instance, if you want to train the hf model directly, you would use:

```
python lora.py --model mistralai/Mistral-7B-v0.1 --train --iters 600
```

On the other hand, if you wish to train the hf model with qlora, the command would be:


```
python lora.py --model mlx-community/deepseek-coder-6.7b-instruct-hf-4bit-mlx --train --iters 600
```
For fine-tuning with all linear layers, use the following command:


```
python lora.py --model mistralai/Mistral-7B-v0.1 --train --iters 600 --all-layers
```

## Preparing Your Data

The dataset for fine-tuning is located in the `data` folder. The file is named `data.jsonl``. The data format should be as follows:

```
{"text": "This is an example for the model."}
```
Please note that you need to prepare your own Q&A dataset. The dataset should be a single text sentence without any beginning of sentence (bos) or end of sentence (eos) characters.


## Merge lora back to original model

Merge the lora model back to the original model. It uses the `lora.py` script with the following command-line arguments:
## Merge lora back to original model

```
python lora.py --model mistralai/Mistral-7B-v0.1 --adapter-file adapters.npz --merge-lora  
```

To run inference only, use the following command:

```
python inference.py --model merged_model --prompt "table: 1-10015132-16
columns: Player, No., Nationality, Position, Years in Toronto, School/Club Team
Q: What is terrence ross' nationality
A: "
```
