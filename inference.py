import torch
from model import Llama3, ModelArgs
from transformers import GPT2TokenizerFast
from colorama import Fore, Back, Style, init

init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

d_model = 512
heads = 8
num_layers = 8
group_size = 1
max_seq_len = 256


args = ModelArgs(
    vocab_size=len(tokenizer),
    tokenizer=tokenizer,
    d_model=d_model,
    heads=heads,
    group_size=group_size,
    num_layers=num_layers,
    max_seq_len=max_seq_len,
    use_flash=True,
)

device = "cuda"
model = Llama3.from_pretrained("tiny_stories_2.pth").to(device)
model.eval()

res = model.generate_kv(
    "This little girl was me,", tokenizer=tokenizer, temp=0.25, top_k=10
)

print(Fore.GREEN + res)
