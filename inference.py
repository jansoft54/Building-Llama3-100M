import torch
from model import Llama3
from transformers import GPT2TokenizerFast
from colorama import Fore, Back, Style, init
init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

d_model=512
heads=8
num_layers=8
group_size=1
max_seq_len=256


device = 'cuda'
model = Llama3(vocab_size=len(tokenizer),
               tokenizer=tokenizer,
               d_model=d_model,
               heads=heads,
               group_size=group_size,
               num_layers=num_layers,
               max_seq_len=max_seq_len,
               use_flash=True).to(device)

model.load_state_dict(torch.load('tiny_stories_2.pth'))
model.eval()
res = model.generate_k_v("The scary wizard",
                      tokenizer=tokenizer,temp=0.5,top_k=20)

print(Fore.GREEN + res)
