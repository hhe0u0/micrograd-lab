"""
NanoGPT: A minimal GPT-style transformer for character-level language modeling

Architecture:
- n_layer = 4 transformer blocks
- n_head = 4 attention heads per block
- n_embd = 128 embedding dimension
- block_size = 128 context window
- dropout = 0.1

Based on Andrej Karpathy's nanoGPT, simplified for educational purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import os
import math

# Hyperparameters
BLOCK_SIZE = 128  # context length
BATCH_SIZE = 64
N_EMBD = 128  # embedding dimension
N_HEAD = 4  # number of attention heads
N_LAYER = 4  # number of transformer blocks
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# Load shakespeare dataset
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text)} characters")

# Create character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} characters")

# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Prepare data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

# Dataset class
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

train_dataset = CharDataset(train_data, BLOCK_SIZE)
val_dataset = CharDataset(val_data, BLOCK_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model components
class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Perform weighted aggregation
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Simple linear layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    """Minimal GPT-style transformer"""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # Get the predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Initialize model
model = NanoGPT(vocab_size).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Evaluation
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': 0, 'val': 0}

    for split, loader in [('train', train_loader), ('val', val_loader)]:
        total_loss = 0
        num_batches = 0
        for X, Y in loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            _, loss = model(X, Y)
            total_loss += loss.item()
            num_batches += 1
            if num_batches >= EVAL_ITERS:
                break
        losses[split] = total_loss / num_batches

    model.train()
    return losses

# Training loop
print("\nStarting training...")
start_time = time.time()
train_losses = []
val_losses = []

for iter_num in range(MAX_ITERS):
    # Sample a batch of data
    for X, Y in train_loader:
        X, Y = X.to(DEVICE), Y.to(DEVICE)

        # Forward pass
        _, loss = model(X, Y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        break  # Only one batch per iteration

    # Evaluate and print loss
    if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])

        elapsed = time.time() - start_time
        print(f"Step {iter_num:4d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f} | Time: {elapsed:.1f}s")

# Final evaluation
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
final_losses = estimate_loss()
print(f"Final train loss: {final_losses['train']:.4f}")
print(f"Final val loss: {final_losses['val']:.4f}")
print(f"Train perplexity: {torch.exp(torch.tensor(final_losses['train'])):.2f}")
print(f"Val perplexity: {torch.exp(torch.tensor(final_losses['val'])):.2f}")
print(f"Total training time: {time.time() - start_time:.1f}s")

# Generate some text
print("\n" + "="*60)
print("SAMPLE GENERATIONS")
print("="*60)
model.eval()
for temp in [0.8, 1.0, 1.2]:
    print(f"\nTemperature: {temp}")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=200, temperature=temp)
    print(decode(generated[0].tolist()))

# Save checkpoint
os.makedirs('checkpoints', exist_ok=True)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'n_embd': N_EMBD,
    'n_head': N_HEAD,
    'n_layer': N_LAYER,
    'block_size': BLOCK_SIZE,
    'stoi': stoi,
    'itos': itos,
    'train_loss': final_losses['train'],
    'val_loss': final_losses['val'],
}
torch.save(checkpoint, 'checkpoints/nanogpt_model.pt')
print(f"\nCheckpoint saved to checkpoints/nanogpt_model.pt")
