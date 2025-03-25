import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, head_size, n_embd, context_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, context_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, context_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, context_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, context_size)
        self.ffwd = FeedFoward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
  def __init__(self, vocab_size, n_embd=32, context_size=8, n_head=4, n_layer=4): #1
    super().__init__()
    self.context_size = context_size

    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # lookup table, vocab_size x vocab_size
    self.position_embedding_table = nn.Embedding(context_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, context_size=context_size) for _ in range(n_layer)])

    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def generate(self, start_idx, number_of_tokens):
    idx = start_idx
    for _ in range(number_of_tokens):
      # crop to last block_size of tokens
      idx_cond = idx[:, -self.context_size:]
      logits, loss = self(idx_cond)
      # apply softmas to get probabilities
      logits = logits[:, -1, :] # (batch_size, context_size)
      probs = F.softmax(logits, dim=1) # (batch_size, context_size)
      idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
      idx = torch.cat((idx, idx_next), dim=1) # (batch_size, t + 1)
      # print(f"DEBUG: Shape of idx: {idx.shape}")
    return idx

  def forward(self, idx, targets=None):
    # idx (batch_size, block_size)
    # print(f"idx shape: {idx.shape}")  # Debugging
    if idx.dim() == 3:  # If extra dimension, fix it
        idx = idx.squeeze(1)  # Remove unnecessary dimension
    # print(f"updated shape:{idx.shape}")
    B, T = idx.shape

    emb = self.token_embedding_table(idx) # (batch_size, block_size, n_embd)
    pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (block_size, n_embd)
    x = emb + pos_emb # (batch_size, block_size, n_embd)

    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (batch_size, block_size, vocab_size)

    if targets is not None:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    else:
      loss = None

    return logits, loss
