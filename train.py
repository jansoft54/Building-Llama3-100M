import torch
from datasets import load_dataset
import tiktoken
import math

torch.set_printoptions(profile="full")
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from model import Llama3, ModelArgs
from transformers import DataCollatorForLanguageModeling
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],  # Logs to stdout
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_seq_len = 256
batch_size = 40
dataset = load_dataset("roneneldan/TinyStories", split="train")  # .select(range(1000))
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=max_seq_len, truncation=True)


dataset = dataset.filter(
    lambda example: example["text"] is not None and example["text"].strip() != ""
)

tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
from torch.utils.data import DataLoader


dataloader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda feature: Llama3.gen_labels(feature, tokenizer, data_collator),
)


warmup = 100
lr = 5e-5
min_lr = 7e-5

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


epochs = 2

model = Llama3(args).to(device)
model.load_state_dict(torch.load("tiny_stories_50M.pth"))
model = torch.compile(model)
model.train()
optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
# lr_scheduler = InverseSquareRootLR(optim, warmup, lr, min_lr=min_lr)
logger.info(
    f"Param. count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)
scaler = torch.GradScaler()

for epoch in range(epochs):
    for i, batch in enumerate(dataloader):

        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optim.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(inputs, attention_mask, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optim)
        scaler.update()
        # lr_scheduler.step()

        logger.info(
            f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
        )
        if (i + 1) % 1000 == 0:
            torch.save(model._orig_mod.state_dict(), "tiny_stories_50M.pth")
