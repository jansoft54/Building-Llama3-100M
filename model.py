
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'
class RoPE:
  @staticmethod
  def compute_freq(head_dim: int,seq_len: int,base:int = 10000):

    exp = -2 * torch.arange(0,head_dim,2).float() / head_dim
    thetas = torch.pow(base,exp)
    m = torch.arange(0,seq_len).float()
    freq = torch.outer(m,thetas).float()
    freq_comp = torch.polar(torch.ones_like(freq),freq)
    return freq_comp
  @staticmethod
  def apply_rotary_embedding(x: torch.Tensor,freq_cis):
    # batch,seq_len,heads,d_k
    x_comp = torch.view_as_complex(x.float().reshape(*(x.shape[:-1]),-1,2))
    freq_com = freq_cis.unsqueeze(0).unsqueeze(2)

    x_out = torch.view_as_real(x_comp * freq_com).reshape(*x.shape)
    return x_out.float()

class FFN(nn.Module):
  def __init__(self,d_model=256,multiple_of=2048):
    super(FFN, self).__init__()
    hidden = 4*d_model
    hidden = int(2*hidden/3)

    hidden = multiple_of*((hidden + multiple_of -1)//multiple_of)

    self.w1 = nn.Linear(d_model,hidden,bias=False)
    self.v = nn.Linear(d_model,hidden,bias=False)
    self.w2 = nn.Linear(hidden,d_model,bias=False)

  def forward(self,x):
    return  self.w2(nn.functional.silu(self.w1(x)) * self.v(x))

import math

class KV_Cache(nn.Module):
  def __init__(self, batch_size, seq_length, n_kv_heads, head_dim,dtype):
    super(KV_Cache, self).__init__()
    device = 'cuda' 
    cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
    self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
    self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

  def update(self,xk,xv,pos):
    bx,seq_len = xk.shape[:2]
    self.cache_k[:bx,pos:pos + seq_len] = xk
    self.cache_v[:bx,pos:pos + seq_len] = xv
    return self.cache_k[:bx,:pos+seq_len],self.cache_v[:bx,:pos+seq_len]

class MultiHeadGQAttention(nn.Module):
  flash = False
  def __init__(self, heads=4,
               d_model=256,
               group_size=2,
               max_seq_len=256):
    super(MultiHeadGQAttention, self).__init__()
    self.heads = heads
    self.d_model = d_model
    self.group_size = group_size

    self.W_q = nn.Linear(d_model,d_model, bias=False)
    self.W_k = nn.Linear(d_model,d_model//group_size, bias=False)
    self.W_v = nn.Linear(d_model,d_model//group_size, bias=False)
    self.W_o = nn.Linear(d_model,d_model, bias=False)
    self.kv_cache = KV_Cache(batch_size=4,
                             seq_length=max_seq_len,
                             n_kv_heads=self.heads//self.group_size,
                             head_dim=d_model // heads,
                             dtype=torch.float32)


  def __repeat_kv(self,x):
    bs, slen, n_kv_heads, head_dim = x.shape
   
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, self.group_size, head_dim)
        .reshape(bs, slen, n_kv_heads * self.group_size, head_dim)
    )
  def forward(self, q,k,v,mask,freq_cis,position=-1,):
    d_k = self.d_model // self.heads
    q,k,v = self.W_q(q),self.W_k(k),self.W_v(v)

    q = q.view(q.shape[0],q.shape[1],self.heads,-1) # (batch,seq_len,heads,d_k) or (batch,1,heads,d_k)
    k = k.view(k.shape[0],k.shape[1],self.heads//self.group_size,-1) # (batch,seq_len,heads,d_k) or (batch,1,heads,d_k)
    v = v.view(v.shape[0],v.shape[1],self.heads//self.group_size,-1)  # (batch,seq_len,heads,d_k) or (batch,1,heads,d_k)
    q = RoPE.apply_rotary_embedding(q,freq_cis)
    k = RoPE.apply_rotary_embedding(k,freq_cis)
    if not self.training:
      k,v = self.kv_cache.update(k,v,position)

    k = self.__repeat_kv(k)
    v = self.__repeat_kv(v)

    q = q.transpose(1,2) # (batch,heads,seq_len,d_k)
    k = k.transpose(1,2) # (batch,heads,seq_len,d_k)
    v = v.transpose(1,2) # (batch,heads,seq_len,d_k)
   
  

    if MultiHeadGQAttention.flash:
      q = q.contiguous()
      k = k.contiguous()
      v = v.contiguous()
      output = F.scaled_dot_product_attention(
        q, k, v, None if mask is None else (mask==1).unsqueeze(1)
        ).reshape(q.shape[0],-1,self.d_model)
    else:
      res = torch.matmul(q,k.transpose(-2,-1))  /  math.sqrt(d_k)

      if mask is not None:
        mask = mask.unsqueeze(1)
        res = torch.masked_fill(res,mask==0,float('-inf'))

        attention = nn.functional.softmax(res,dim=-1)

        output = torch.matmul(attention,v).transpose(1,2).contiguous().view(q.shape[0],-1,self.d_model)
    
    output = self.W_o(output)


    return output

class DecoderLayer(nn.Module):
  def __init__(self,
               d_model=256,
               heads=4,
               group_size=2,
               max_seq_len=256):
    super(DecoderLayer, self).__init__()
    self.norm1 = nn.RMSNorm(d_model,eps=1e-6)
    self.norm2 = nn.RMSNorm(d_model,eps=1e-6)
    self.ffn = FFN(d_model=d_model)
    self.attention = MultiHeadGQAttention(heads=heads,
                                          d_model=d_model,
                                          group_size=group_size,
                                          max_seq_len=max_seq_len)

  def forward(self,x,tgt_causal_mask,pos,freqs_cis):
    x_norm = self.norm1(x)
    x = x + self.attention(x_norm,x_norm,x_norm,tgt_causal_mask,freqs_cis,position=pos)
    return x + self.ffn(self.norm2(x))

class Llama3(nn.Module):
  def __init__(self,vocab_size,
               d_model,
               heads,
               group_size=2,
               num_layers=8,
               max_seq_len=256,
               tokenizer=None,
               ignore_index=-100,
               use_flash = False
  ):
    super(Llama3, self).__init__()
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.ignore_index=ignore_index
    self.num_layers = num_layers
    self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                              heads=heads,
                                              group_size=group_size,
                                              max_seq_len=max_seq_len) for i in range(num_layers)])

    self.embedding = nn.Embedding(vocab_size,d_model)
    self.norm = nn.RMSNorm(d_model,eps=1e-6)
    self.ffn = nn.Linear(d_model,vocab_size,bias=False)

    self.d_k = d_model//heads
    self.freqs_cis = RoPE.compute_freq(head_dim=d_model//heads,seq_len=max_seq_len)
    MultiHeadGQAttention.flash = use_flash

  @staticmethod
  def _build_masks(seq_len,attention_mask,device):
    causal = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool)).to(device).unsqueeze(0)
    if attention_mask == None:
      return causal
    attention_mask = attention_mask.unsqueeze(1).repeat(1,seq_len,1).int()
    return (causal & attention_mask).int()
  @staticmethod
  def gen_labels(labels,tokenizer,data_collator,pad_token_id=-100):
    batch = data_collator(labels)
    labels = batch['labels']
    for i in range(labels.shape[0]):
      l = torch.roll(labels[i],-1)
      target_indices = (l == pad_token_id).nonzero(as_tuple=True)[0]
      if len(target_indices) == 0:
        l[-1] = tokenizer.eos_token_id
      else:
        seq_len = len(l) - len(target_indices) 
        l[-1] = pad_token_id
        l[seq_len-1] = tokenizer.eos_token_id
      labels[i] = l
    batch['labels'] = labels
    return batch
  def calc_loss(self,logits,labels):
    loss = nn.functional.cross_entropy(logits.view(-1,logits.shape[-1]),labels.view(-1),ignore_index=self.ignore_index)
    return loss

  def generate(self,prompt,tokenizer,temp=1.0,top_k=None):
    device = 'cuda' 
    tokenized = tokenizer(prompt, max_length=self.max_seq_len,truncation=True)
    tokens = torch.tensor(tokenized['input_ids']).unsqueeze(0).to(device)
    sampled_token = None
    sampled_token_list = tokens[0].tolist()
    i = 0
    while sampled_token != tokenizer.eos_token_id and i < self.max_seq_len:
      logits = self.__run_model(tokens,None,position=i)[:,-1,:] / temp
      if top_k == None:
        probabilities = F.softmax(logits, dim=-1)
        new_token = torch.argmax(probabilities,dim=-1).unsqueeze(0)
      else:
        indices_to_remove = logits < torch.topk(logits,k=top_k,dim=1)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
        probabilities=F.softmax(logits/temp,dim=-1).squeeze(0)
        new_token = torch.argmax(probabilities,dim=-1).unsqueeze(0)#torch.multinomial(probabilities,num_samples=1).unsqueeze(0)
      
      i = i + tokens.shape[1]
      tokens = new_token
      sampled_token = new_token.squeeze().item()
      sampled_token_list.append(sampled_token)


    sampled_token_list = sampled_token_list[:-1] if sampled_token == tokenizer.eos_token_id else sampled_token_list
    return tokenizer.decode(sampled_token_list)
  def generate2(self,prompt,tokenizer,temp=1.0,top_k=None):
    device = 'cuda'
    tokenized = tokenizer(prompt, max_length=self.max_seq_len,truncation=True)
    tokens = torch.tensor(tokenized['input_ids']).unsqueeze(0).to(device)
    sampled_token = None
    i = 0
    while sampled_token != tokenizer.eos_token_id and i < 128:
     
      logits = self.__run_model(tokens,None)[:,-1,:] / temp

      probabilities = F.softmax(logits, dim=-1)
      new_token = torch.argmax(probabilities,dim=-1).unsqueeze(0)
      tokens = torch.cat((tokens, new_token), dim=1)
      sampled_token = new_token.squeeze().item()


    tokens = tokens.squeeze().tolist()
    tokens = tokens[:-1] if sampled_token == tokenizer.eos_token_id else tokens
    return tokenizer.decode(tokens)
    
  def __run_model(self,tgt,attention_mask,position=0):
    causal_mask = Llama3._build_masks(tgt.shape[1],attention_mask,tgt.device) if self.training else None
    tgt_embed = self.embedding(tgt)
    freqs_cis = self.freqs_cis[position:position + tgt_embed.shape[1]].to(tgt.device)

    for i in range(self.num_layers):
      tgt_embed = self.layers[i](tgt_embed,causal_mask,position,freqs_cis)

    return self.ffn(self.norm(tgt_embed))
  
  def forward(self,tgt,attention_mask,labels):

    logits = self.__run_model(tgt,attention_mask)
   
    return self.calc_loss(logits,labels)