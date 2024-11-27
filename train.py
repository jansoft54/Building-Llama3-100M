import torch
from datasets import load_dataset
import tiktoken
import math

from datasets import load_dataset
from transformers import GPT2TokenizerFast
from model import Llama3
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to stdout
    ]
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = 256
batch_size = 12
n_batches = 2000
# Load the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples['text'], max_length=max_len,truncation=True)

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
    collate_fn=data_collator
)



d_model=512
heads=16
num_layers=16
group_size=4
max_seq_len=256

warmup = 100
lr = 1e-4
min_lr = 1e-6

class InverseSquareRootLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, init_lr, min_lr=1e-9, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        super(InverseSquareRootLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            lr = self.init_lr * (step / self.warmup_steps)
        else:
            lr = self.init_lr * math.sqrt(self.warmup_steps / step)

        lr = max(lr, self.min_lr)

        return [lr for _ in self.base_lrs]

epochs = 50

model = Llama3(vocab_size=len(tokenizer),
               tokenizer=tokenizer,
               d_model=d_model,
               heads=heads,
               group_size=group_size,
               num_layers=num_layers,
               max_seq_len=max_seq_len).to(device)
#model.load_state_dict(torch.load('model.pth'))

optim = torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9, 0.98))
lr_scheduler = InverseSquareRootLR(optim,warmup,lr,min_lr=min_lr)

#train_set = tokenized_dataset['train']
logger.info(f"Param. count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
for epoch in range(epochs):
    for i,batch in enumerate(dataloader):
       
       # print(batch['attention_mask'])
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        loss = model(inputs,attention_mask,labels)
        loss.backward()
      #  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()
        lr_scheduler.step()
        optim.zero_grad()

    logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

   

torch.save(model.state_dict(), 'best_model.pth')






