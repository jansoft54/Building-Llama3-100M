import torch
from model import Llama3, ModelArgs
from transformers import GPT2TokenizerFast
from colorama import Fore, Back, Style, init

init()

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

res = model.generate_kv(
    "There once was a strong man called Bene who liked to play on the computer.",
    tokenizer=tokenizer,
    top_p=0.8,
)

print(Fore.GREEN + res)
