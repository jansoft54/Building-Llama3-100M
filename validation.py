import torch
from model import Llama3
from transformers import GPT2TokenizerFast
from colorama import Fore, Back, Style, init
import torch
from datasets import load_dataset
import tiktoken
import math

from datasets import load_dataset
from transformers import GPT2TokenizerFast
from model import Llama3
import logging
init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

d_model=512
heads=16
num_layers=16
group_size=4
max_seq_len=512

warmup = 100
lr = 2.5e-4
min_lr = 1e-6

device = 'cuda'
model = Llama3(vocab_size=len(tokenizer),
               tokenizer=tokenizer,
               d_model=d_model,
               heads=heads,
               group_size=group_size,
               num_layers=num_layers,
               max_seq_len=max_seq_len).to(device)

model.load_state_dict(torch.load('best_model_2.pth'))



batch_size = 4
n_batches = 2000
# Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='test')

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples['text'], max_length=max_seq_len,truncation=True)

dataset = dataset.filter(lambda example: example['text'] is not None and example['text'].strip() != '')

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)#.select(range(batch_size*n_batches))
print(len(tokenized_dataset))

from transformers import DataCollatorForLanguageModeling

# Initialize the data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)
from torch.utils.data import DataLoader

dataloader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,  # Adjust based on your resources
    shuffle=True,
    collate_fn=lambda feature : Llama3.gen_labels(feature,tokenizer,data_collator)
)
model.train()
for i,batch in enumerate(dataloader):
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    loss = model(inputs,attention_mask,labels)
    print(loss)