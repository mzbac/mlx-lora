# mlx-lora

## Setup

Install the dependencies:
```
pip install -r requirements.txt
```

## Training the Model

You can customize the lora.py script as per your requirements. To run the script, use:

```
python lora.py
```

## Preparing Your Data

The dataset for fine-tuning is located in the `data` folder. The data format should be as follows:

```
{"text": "This is an example for the model."}
```
Please note that you need to preprocess your own Q&A dataset to construct each Q&A pair into a single sentence. For example, if you have a question "What is the capital of France?" and its answer "The capital of France is Paris." you should concatenate these into a single sentence. A possible concatenated sentence could be: "Q:What is the capital of France?\nA:The capital of France is Paris," and in your data.json it should be like:
```
{"text": "Q:What is the capital of France?\nA:The capital of France is Paris."}
```


## Merge lora back to original model

Merge the lora model back to the original model. It uses the mlx-lm `fuse.py` script with the following command-line arguments:
## Merge lora back to original model

```
python -m mlx_lm.fuse --model <path_to_model> --adapter-file <path_to_adapter>
```

To run inference only, use the following command:

```
python -m mlx_lm.generate --model <path_to_model> --adapter-file <path_to_adapter>
```
