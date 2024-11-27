import torch
from model import Llama3
from transformers import GPT2TokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

d_model=256
heads=16
num_layers=16
group_size=4
max_seq_len=256

warmup = 100
lr = 2.5e-4
min_lr = 1e-6

device = 'cuda'
model = Llama3(vocab_size=len(tokenizer),
               d_model=d_model,
               heads=heads,
               group_size=group_size,
               num_layers=num_layers,
               max_seq_len=max_seq_len).to(device)

model.load_state_dict(torch.load('model.pth'))

res = model.generate("The game began development in",tokenizer=tokenizer,temp=1)

print(res)