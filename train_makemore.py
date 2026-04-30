"""
Makemore: Character-level language model with bigram + simple MLP

Architecture:
- Character embeddings (vocab_size → 64)
- Linear layer 1 (64 → 128) with ReLU
- Linear layer 2 (128 → vocab_size)
- Context window: 8 characters

Based on Andrej Karpathy's makemore, simplified for educational purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import os

# Hyperparameters
BLOCK_SIZE = 8  # context length (look back 8 characters)
BATCH_SIZE = 64
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500
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
print(f"Vocabulary: {''.join(chars[:20])}...")

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

# Model definition
class MakemoreModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, block_size):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * block_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, block_size)
        B, T = x.shape
        emb = self.embedding(x)  # (B, T, embedding_dim)
        emb = emb.view(B, -1)  # (B, T * embedding_dim)
        hidden = F.relu(self.fc1(emb))  # (B, hidden_dim)
        logits = self.fc2(hidden)  # (B, vocab_size)
        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate new tokens given context idx"""
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # Get predictions
            logits = self(idx_cond)
            # Apply temperature
            logits = logits / temperature
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize model
model = MakemoreModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, BLOCK_SIZE).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Loss function
def compute_loss(logits, targets):
    B, T = targets.shape
    # For each position in the sequence, predict the next character
    loss = 0
    for i in range(T):
        # Predict character at position i+1 using context up to position i
        if i == 0:
            # First position: use just the embedding
            continue
        loss += F.cross_entropy(logits, targets[:, i])
    return loss / T

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
            logits = model(X)
            # Simple loss: predict last character from context
            loss = F.cross_entropy(logits, Y[:, -1])
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
        logits = model(X)
        # Predict last character from context
        loss = F.cross_entropy(logits, Y[:, -1])

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
    context = torch.zeros((1, BLOCK_SIZE), dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=200, temperature=temp)
    print(decode(generated[0].tolist()))

# Save checkpoint
os.makedirs('checkpoints', exist_ok=True)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': HIDDEN_DIM,
    'block_size': BLOCK_SIZE,
    'stoi': stoi,
    'itos': itos,
    'train_loss': final_losses['train'],
    'val_loss': final_losses['val'],
}
torch.save(checkpoint, 'checkpoints/makemore_model.pt')
print(f"\nCheckpoint saved to checkpoints/makemore_model.pt")
