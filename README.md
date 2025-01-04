
## Overview
Llama3 is a Transformer-based language model that includes several advanced features like Rotary Position Embeddings (RoPE), multi-head grouped query attention, and key-value caching for efficient inference. The model is designed to be highly customizable, supporting options for flash attention, grouping, and adaptive feed-forward networks.

<p align="center">
  <img src="assets/llama3.webp" width="400" height="400" alt="Cute Llama">
</p>

## Key Features
- **Rotary Position Embedding (RoPE)**: The model utilizes RoPE to apply rotary positional encoding to input tensors, enhancing its capability to capture positional relationships.
- **Grouped Query Attention (GQAttention)**: Uses grouped query multi-head attention, which allows splitting of heads for more computational efficiency and better generalization.
- **Key-Value Caching**: Supports caching of key-value pairs to speed up sequential generation, particularly useful during inference.
- **Flash Attention**: Optionally uses Flash Attention to accelerate attention calculations during inference.

Each `DecoderLayer` follows the general Transformer decoder architecture, but with enhancements that improve efficiency and adaptability.

## Code Structure
- **RoPE Class**: Implements Rotary Position Embedding, computing frequencies and applying them to tensors.
- **FFN Class**: Defines a feed-forward network used in each decoder layer, with customized hidden layer scaling.
- **KV_Cache Class**: Implements caching for key-value pairs for faster sequential generation.
- **MultiHeadGQAttention Class**: Implements multi-head grouped query attention, with support for Flash Attention.
- **DecoderLayer Class**: Represents a single Transformer decoder layer, combining attention and feed-forward networks.
- **Llama3 Class**: The main model that supports both training and generation functionalities.

## Usage
### Training
The model can be trained using the standard PyTorch training loop. The following parameters are required:
- **Target Sequence (`tgt`)**: The input sequence of tokens.
- **Attention Mask (`attention_mask`)**: An optional mask to handle padded positions causal attention.
- **Labels (`labels`)**: Token labels for loss calculation.

The `forward()` method calculates the cross-entropy loss given the target and labels.

## Example
To use Llama3 for text generation, you can instantiate the model and use the `generate()` method:

```python
# Example instantiation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
max_seq_len = 256
args = ModelArgs(
    vocab_size=len(tokenizer),
    tokenizer=tokenizer,
    d_model=256,
    heads=4,
    group_size=2,
    num_layers=32,
    max_seq_len=max_seq_len,
    use_flash=True,
)
model = Llama3.from_pretrained("tiny_stories_50M.pth", args).to(device)
model.eval()

generated_text = model.generate_kv(
    "There once was a strong man called Bene who liked to play on the computer.",
    tokenizer=tokenizer,
    top_p=0.8,
)
print(generated_text)
```

## Installation
The model relies on PyTorch for deep learning capabilities. Install the necessary dependencies:
```sh
pip install torch
```

## Notes
- Ensure that you use a compatible tokenizer when working with the model.
- The `generate()` and `forward()` methods allow easy integration into existing Transformer pipelines.

