import math
import mlx.nn as nn
import mlx.core as mx


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 16):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape

        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims, output_dims, rank
        )
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self, input_dims: int, output_dims: int, lora_rank: int = 8, bias: bool = False
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Low rank lora weights
        scale = 1 / math.sqrt(6/input_dims) # change to kaiming uniform
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, lora_rank),
        )
        self.lora_b = mx.zeros(shape=(lora_rank, output_dims))

    def __call__(self, x):
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype
        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b
        return y + 2.0 * z

    def merge(self):
        """
        Merge the base linear layer with lora weights from LoRA layers.

        Returns:
            nn.Linear: A new linear layer with modified weights.
        """
        if isinstance(self.linear, nn.QuantizedLinear):
            self.linear.weight = mx.dequantize(
                self.linear.weight,
                self.linear.scales,
                self.linear.biases,
                self.linear.group_size,
                self.linear.bits,
            )
        output_dims, input_dims = self.linear.weight.shape
        self.linear.weight += (self.lora_a @ self.lora_b).T * 2.0
        new_linear = nn.Linear(
            input_dims,
            output_dims,
            bias=self.linear.bias is not None,
        )
        new_linear.weight = self.linear.weight
        new_linear.bias = self.linear.bias
        return new_linear
