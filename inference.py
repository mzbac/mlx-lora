import argparse
from pathlib import Path
import mlx.core as mx
import models

def generate(model, prompt, tokenizer, args):
    print(args.prompt, end="", flush=True)
    prompt = mx.array(tokenizer.encode(args.prompt))

    def generate_step():
        temp = args.temp

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        logits, cache = model(prompt[None])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, cache = model(y[:, None], cache)
            y = sample(logits.squeeze(1))
            yield y

    tokens = []
    for token, _ in zip(generate_step(), range(args.num_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)

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
    model, tokenizer = models.load(args.model)
    if args.prompt is not None:
        print("Generating")
        generate(model, args.prompt, tokenizer, args)