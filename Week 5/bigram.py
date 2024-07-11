import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    indx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in indx])
    y = torch.stack([data[i+1:i+block_size+1] for i in indx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # during this we do not need .backward()
def estimate_loss(): # averages up losses over multiple batches
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, indx, targets=None):
        
        # indx and target are of shape (B, T)
        logits = self.token_embedding_table(indx) # plucks out a row from the embedding table corresponding to the indx, it will be a vocab_size tensor
        # (B, T, C) logits - scores for the next character in the sequence - this is just by seeing itself
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # targets.view(-1)
            loss = F.cross_entropy(logits, targets) # cross entropy expects the second argument to be the number of classes
        # functional.cross_entropy means we don't have to create a module for it
        return logits, loss
     
    def generate(self, indx, max_new_tokens):
        
        # indx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(indx)
            # take the last one because it is the actual next character
            logits = logits[:, -1, :]
            # softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            indx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to get the running sequence
            indx = torch.cat((indx, indx_next), dim=1)
        return indx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss() # because in previous method every batch can be more or less lucky
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))