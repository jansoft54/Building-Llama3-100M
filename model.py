import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

device = "cuda"


class RoPE:
    @staticmethod
    def compute_freq(head_dim: int, seq_len: int, base: int = 10000):

        exp = -2 * torch.arange(0, head_dim, 2).float() / head_dim
        thetas = torch.pow(base, exp)
        m = torch.arange(0, seq_len).float()
        freq = torch.outer(m, thetas).float()
        freq_comp = torch.polar(torch.ones_like(freq), freq)
        return freq_comp

    @staticmethod
    def apply_rotary_embedding(x: torch.Tensor, freq_cis):
        # batch,seq_len,heads,d_k
        x_comp = torch.view_as_complex(x.float().reshape(*(x.shape[:-1]), -1, 2))
        freq_com = freq_cis.unsqueeze(0).unsqueeze(2)

        x_out = torch.view_as_real(x_comp * freq_com).reshape(*x.shape)
        return x_out.float()


class FFN(nn.Module):
    def __init__(self, d_model=256, multiple_of=256):
        super(FFN, self).__init__()
        hidden = 4 * d_model
        hidden = int(2 * hidden / 3)

        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.v = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.v(x))


import math


class KV_Cache(nn.Module):
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim):
        super(KV_Cache, self).__init__()
        device = "cuda"
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.cache_k = torch.zeros(cache_shape, device=device)
        self.cache_v = torch.zeros(cache_shape, device=device)

    def update(self, xk, xv, pos):
        bx, seq_len = xk.shape[:2]
        self.cache_k[:bx, pos : pos + seq_len] = xk
        self.cache_v[:bx, pos : pos + seq_len] = xv
        return self.cache_k[:bx, : pos + seq_len], self.cache_v[:bx, : pos + seq_len]


class MultiHeadGQAttention(nn.Module):
    flash = False

    def __init__(self, heads=4, d_model=256, group_size=2, max_seq_len=256):
        super(MultiHeadGQAttention, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.group_size = group_size

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model // group_size, bias=False)
        self.W_v = nn.Linear(d_model, d_model // group_size, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.kv_cache = KV_Cache(
            batch_size=4,
            seq_length=max_seq_len,
            n_kv_heads=self.heads // self.group_size,
            head_dim=d_model // heads,
        )

    def __repeat_kv(self, x):
        bs, slen, n_kv_heads, head_dim = x.shape

        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, self.group_size, head_dim)
            .reshape(bs, slen, n_kv_heads * self.group_size, head_dim)
        )

    def forward(
        self,
        q,
        k,
        v,
        mask,
        freq_cis,
        position=-1,
    ):
        d_k = self.d_model // self.heads
        bs, seq_len = q.shape[:2]
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)

        q = q.view(
            q.shape[0], q.shape[1], self.heads, -1
        )  # (batch,seq_len,heads,d_k) or (batch,1,heads,d_k)
        k = k.view(
            k.shape[0], k.shape[1], self.heads // self.group_size, -1
        )  # (batch,seq_len,heads,d_k) or (batch,1,heads,d_k)
        v = v.view(
            v.shape[0], v.shape[1], self.heads // self.group_size, -1
        )  # (batch,seq_len,heads,d_k) or (batch,1,heads,d_k)
        q = RoPE.apply_rotary_embedding(q, freq_cis)
        k = RoPE.apply_rotary_embedding(k, freq_cis)
        if not self.training:
            k, v = self.kv_cache.update(k, v, position)

        k = self.__repeat_kv(k)
        v = self.__repeat_kv(v)

        q, k, v = (x.transpose(1, 2) for x in (q, k, v))

        if MultiHeadGQAttention.flash:
            q, k, v = (x.contiguous() for x in (q, k, v))
            if mask is not None:
                mask = (mask == 1).unsqueeze(1)
            output = (
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                .transpose(1, 2)
                .reshape(bs, seq_len, -1)
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                mask = mask.unsqueeze(1)
                scores = torch.masked_fill(scores, mask == 0, float("-inf"))
            attention = nn.functional.softmax(scores, dim=-1)
            output = (
                torch.matmul(attention, v)
                .transpose(1, 2)
                .contiguous()
                .view(bs, seq_len, -1)
            )

        output = self.W_o(output)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, heads=4, group_size=2, max_seq_len=256):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-6)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-6)
        self.ffn = FFN(d_model=d_model)
        self.attention = MultiHeadGQAttention(
            heads=heads, d_model=d_model, group_size=group_size, max_seq_len=max_seq_len
        )

    def forward(self, x, tgt_causal_mask, pos, freqs_cis):
        x_norm = self.norm1(x)
        x = x + self.attention(
            x_norm, x_norm, x_norm, tgt_causal_mask, freqs_cis, position=pos
        )
        return x + self.ffn(self.norm2(x))


@dataclass
class ModelArgs:
    vocab_size: int
    d_model: int = 256
    heads: int = 4
    group_size: int = 2
    num_layers: int = 8
    max_seq_len: int = 256
    tokenizer: any = None
    ignore_index: int = -100
    use_flash: bool = False


class Llama3(nn.Module):
    def __init__(self, params: ModelArgs):
        super(Llama3, self).__init__()
        self.tokenizer = params.tokenizer
        self.max_seq_len = params.max_seq_len
        self.ignore_index = params.ignore_index
        self.num_layers = params.num_layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=params.d_model,
                    heads=params.heads,
                    group_size=params.group_size,
                    max_seq_len=params.max_seq_len,
                )
                for _ in range(params.num_layers)
            ]
        )
        self.embedding = nn.Embedding(params.vocab_size, params.d_model)
        self.norm = nn.RMSNorm(params.d_model, eps=1e-6)
        self.ffn = nn.Linear(params.d_model, params.vocab_size, bias=False)
        self.group_size = params.group_size
        self.d_model = params.d_model
        self.heads = params.heads
        self.d_k = params.d_model // params.heads
        self.freqs_cis = RoPE.compute_freq(
            head_dim=params.d_model // params.heads, seq_len=params.max_seq_len
        )
        self.d_model = params.d_model

        MultiHeadGQAttention.flash = params.use_flash

    @staticmethod
    def from_pretrained(checkpoint, params: ModelArgs) -> "Llama3":

        model = Llama3(params).to(device)
        model.load_state_dict(torch.load(checkpoint))
        return model

    @staticmethod
    def build_masks(seq_len, attention_mask, device, position=0, training=False):
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0)

        if not training:
            i = torch.arange(seq_len).unsqueeze(1)  # Shape: (seqlen, 1)
            j = torch.arange(position + seq_len).unsqueeze(
                0
            )  # Shape: (1, cache_len + seqlen)
            return (j <= (position + i)).int().unsqueeze(0).to(device)

        if attention_mask == None:
            return causal
        attention_mask = attention_mask.unsqueeze(1).repeat(1, seq_len, 1).int()

        return (causal & attention_mask).int()

    @staticmethod
    def gen_labels(labels, tokenizer, data_collator, ignore_index=-100):
        batch = data_collator(labels)
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        for i in range(labels.shape[0]):
            l = torch.roll(labels[i], -1)
            target_indices = (l == ignore_index).nonzero(as_tuple=True)[0]
            if len(target_indices) == 0:
                l[-1] = tokenizer.eos_token_id
            else:
                seq_len = len(l) - len(target_indices)
                l[-1] = ignore_index
                l[seq_len - 1] = tokenizer.eos_token_id

            labels[i] = l

        batch["labels"] = labels
        batch["attention_mask"] = Llama3.build_masks(
            labels.shape[1], attention_mask, device, training=True
        )

        return batch

    def calc_loss(self, logits, labels):
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=self.ignore_index,
        )
        return loss

    @torch.inference_mode
    def generate_kv(self, prompt, tokenizer, temp=1.0, top_k=None, top_p=None):
        device = "cuda"
        tokenized = tokenizer(prompt, max_length=self.max_seq_len, truncation=True)
        tokens = torch.tensor(tokenized["input_ids"]).unsqueeze(0).to(device)
        sampled_token = None
        sampled_tokens = tokens.squeeze(0).tolist()

        token_len = tokens.shape[1]

        mask = (
            torch.tril(torch.ones(token_len, token_len, dtype=torch.bool))
            .unsqueeze(0)
            .to(device)
        )

        i = 0
        for block in self.layers:
            block.attention.cache = KV_Cache(
                batch_size=1,
                seq_length=self.max_seq_len,
                n_kv_heads=self.heads // self.group_size,
                head_dim=self.d_model // self.heads,
            )

        while i < self.max_seq_len:

            logits = self.__run_model(tokens, mask, position=i)[:, -1, :] / temp
            probabilities = F.softmax(logits.float(), dim=-1).squeeze()
            if top_k is not None:
                topk_probs, _ = torch.topk(probabilities, top_k)
                probabilities = torch.where(
                    probabilities < topk_probs.squeeze(0)[-1], 0, probabilities
                )
                sampled_token = torch.multinomial(
                    probabilities / probabilities.sum(), num_samples=1
                )
            elif top_p is not None:
                sorted_probs, sorted_indices = torch.sort(
                    probabilities, descending=True
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=True)

                sorted_probs[cutoff_index + 1 :] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                probabilities = torch.zeros_like(probabilities).scatter(
                    0, sorted_indices, sorted_probs
                )

                sampled_token = torch.multinomial(probabilities, num_samples=1)
            else:
                sampled_token = torch.multinomial(probabilities, num_samples=1)

            tokens = sampled_token.unsqueeze(0)
            if sampled_token.item() != tokenizer.eos_token_id:
                sampled_tokens.append(sampled_token.item())
            else:
                break
            i = len(sampled_tokens)
            mask = None

        return tokenizer.decode(sampled_tokens)

    def __run_model(self, tgt, attention_mask, position=0):

        tgt_embed = self.embedding(tgt)
        freqs_cis = self.freqs_cis[position : position + tgt_embed.shape[1]].to(
            tgt.device
        )

        for i in range(self.num_layers):
            tgt_embed = self.layers[i](tgt_embed, attention_mask, position, freqs_cis)
        return self.ffn(self.norm(tgt_embed))

    def forward(self, tgt, attention_mask, labels):
        logits = self.__run_model(tgt, attention_mask)
        return self.calc_loss(logits, labels)
