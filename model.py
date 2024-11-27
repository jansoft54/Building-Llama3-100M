
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'
class RoPE(nn.Module):
  def __init__(self,head_dim=512,max_seq_len=256):
    super(RoPE, self).__init__()
    self.head_dim = head_dim
    freq_com = self.compute_freq(head_dim,max_seq_len)
    self.register_buffer('freq_com', freq_com)

  def compute_freq(self,head_dim: int,seq_len: int,base:int = 10**4):

    exp = -2 * torch.arange(0,head_dim,2).float() / head_dim
    thetas = torch.pow(base,exp)
    m = torch.arange(0,seq_len).float()
    freq = torch.outer(m,thetas).float()
    freq_comp = torch.polar(torch.ones_like(freq),freq)

    return freq_comp

  def apply_rotary_embedding(self,x: torch.Tensor):
    x_comp = torch.view_as_complex(x.float().reshape(*(x.shape[:-1]),-1,2))

    seq_len = x.shape[1]

    freq_com = self.freq_com[:seq_len,:]
    freq_com = freq_com.unsqueeze(0).unsqueeze(2)
    x_rotate = x_comp * freq_com
    x_out = torch.view_as_real(x_rotate).reshape(*x.shape)
    return x_out.float()

  def forward(self,x):
    return self.apply_rotary_embedding(x)



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
    x = nn.functional.silu(self.w1(x)) * self.v(x)
    return  self.w2(x)

import math
class MultiHeadGQAttention(nn.Module):
  def __init__(self, heads=4,
               d_model=256,
               seq_len=128,
               group_size=2,
               max_seq_len=256):
    super(MultiHeadGQAttention, self).__init__()
    self.heads = heads
    self.d_model = d_model
    self.group_size = group_size

    self.W_q = nn.Linear(d_model,d_model)
    self.W_k = nn.Linear(d_model,d_model//group_size)
    self.W_v = nn.Linear(d_model,d_model//group_size)
    self.W_o = nn.Linear(d_model,d_model)
    self.rope = RoPE(head_dim=d_model//heads,max_seq_len=max_seq_len)

  def forward(self, q,k,v,mask):
    d_k = self.d_model // self.heads
    q,k,v = self.W_q(q),self.W_k(k),self.W_v(v)



    q = q.view(q.shape[0],q.shape[1],self.heads,-1) 

    k = k.view(k.shape[0],k.shape[1],self.heads//self.group_size,-1) 
    v = v.view(v.shape[0],v.shape[1],self.heads//self.group_size,-1) 

    q = self.rope(q)
    k = self.rope(k)

    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)


    k = k.repeat(1,self.group_size,1,1)
    v = v.repeat(1,self.group_size,1,1)

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
    self.norm1 = nn.RMSNorm(d_model)
    self.norm2 = nn.RMSNorm(d_model)
    self.ffn = FFN(d_model=d_model)
    self.attention = MultiHeadGQAttention(heads=heads,
                                          d_model=d_model,
                                          group_size=group_size,
                                          max_seq_len=max_seq_len)

  def forward(self,x,tgt_causal_mask):
    x_norm = self.norm1(x)
    x = x + self.attention(x_norm,x_norm,x_norm,tgt_causal_mask)
    return x + self.ffn(self.norm2(x))

class Llama3(nn.Module):
  def __init__(self,vocab_size,
               d_model,
               heads,
               group_size=2,
               num_layers=8,
               max_seq_len=256,
               tokenizer=None,
               ignore_index=-100
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
    self.norm = nn.RMSNorm(d_model)
    self.ffn = nn.Linear(d_model,vocab_size)

  @staticmethod
  def _build_masks(seq_len,attention_mask):
    causal = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool)).to(attention_mask.device)
    attention_mask = attention_mask.unsqueeze(1).repeat(1,seq_len,1).int()
    return (causal & attention_mask).int()
  def __gen_labels(self,labels):
    eos_token = torch.full((labels.shape[0], 1), self.tokenizer.eos_token_id, dtype=labels.dtype).to(labels.device)
    labels = torch.cat((labels, eos_token), dim=-1)
    labels = labels[:, 1:].contiguous()
    return labels
  def calc_loss(self,logits,labels):
    loss = nn.functional.cross_entropy(logits.view(-1,logits.shape[-1]),labels.view(-1),ignore_index=self.ignore_index)
    return loss

  def generate(self,prompt,tokenizer,temp=1.0,top_k=None):
    device = 'cuda'
    tokenized = tokenizer(prompt, max_length=self.max_seq_len,truncation=True)
    tokens = torch.tensor(tokenized['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(tokenized['attention_mask']).unsqueeze(0).to(device)
    sampled_token = None
    i = 0
    print(temp)
    while sampled_token != tokenizer.eos_token_id and i < 128:
      i= i+1
      logits = self.__run_model(tokens,attention_mask)[:,-1,:] / temp

      probabilities = F.softmax(logits, dim=-1)
      new_token = torch.multinomial(probabilities.squeeze(), 1).view(1,1) #torch.argmax(probabilities,dim=-1).unsqueeze(0)
      tokens = torch.cat((tokens, new_token), dim=1)
      attention_mask = torch.full_like(tokens, fill_value=1, device=device)  # Efficient mask update
      sampled_token = new_token.squeeze().item()


    tokens = tokens.squeeze().tolist()
    tokens = tokens[:-1] if sampled_token == tokenizer.eos_token_id else tokens
    return tokenizer.decode(tokens)
 


  def __run_model(self,tgt,attention_mask):
    causal_mask = Llama3._build_masks(tgt.shape[1],attention_mask)
    tgt_embed = self.embedding(tgt)
    for i in range(self.num_layers):
      tgt_embed = self.layers[i](tgt_embed,causal_mask)

    logits = self.ffn(self.norm(tgt_embed))
    return logits


  def forward(self,tgt,attention_mask,labels):

    labels = self.__gen_labels(labels)
   
    logits = self.__run_model(tgt,attention_mask)
    return self.calc_loss(logits,labels)