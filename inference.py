import argparse
import json
from pathlib import Path
import time
import mlx.core as mx
import llama
import phi2
from utils import get_model_path
import transformers

def generate(
    model,
    tokenizer: transformers.AutoTokenizer,
    prompt: str,
    max_tokens: int,
    temp: float = 0.0,
):
    prompt = tokenizer.encode(prompt)

    prompt = mx.array(prompt)

    tic = time.time()
    tokens = []
    skip = 0
    for token, n in zip(
        model.generate(prompt, args.temp),
        range(args.num_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        if n == 0:
            prompt_time = time.time() - tic
            tic = time.time()

        tokens.append(token.item())
        # if (n + 1) % 10 == 0:
        s = tokenizer.decode(tokens)
        print(s[skip:], end="", flush=True)
        skip = len(s)
    print(tokenizer.decode(tokens)[skip:], flush=True)
    gen_time = time.time() - tic
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return
    prompt_tps = prompt.size / prompt_time
    gen_tps = (len(tokens) - 1) / gen_time
    print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
    print(f"Generation: {gen_tps:.3f} tokens-per-sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm inference script")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_model",
        help="The path to the mlx model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="hello",
    )
    parser.add_argument(
        "--num-tokens",
        "-n",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.6,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    print("Loading pretrained model")
    model_path = get_model_path(args.model)
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        
    if config["model_type"] == "phi-msft":
        model, tokenizer = phi2.load(args.model)
    else:
        model, tokenizer = llama.load(args.model)
    if args.prompt is not None:
        print("Generating")
        model.eval()
        generate(model=model, prompt= args.prompt, tokenizer= tokenizer, max_tokens=args.num_tokens, temp=args.temp)