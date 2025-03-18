#!/bin/bash

# Script to run training with test data to avoid HuggingFace permission issues

# Create output directory
OUTPUT_DIR="./output/extended_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Test GPU availability
echo "Testing GPU..."
python test_gpu.py

# Create some test data
echo "Creating test data..."
mkdir -p ./data
cat > ./data/train.txt << EOF
This is a test document. It contains text for training a language model.
The quick brown fox jumps over the lazy dog. This sentence contains all letters of the English alphabet.
Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.
Transformers are a type of model architecture that uses self-attention mechanisms to process sequential data.
Deep learning is a subset of machine learning that uses neural networks with many layers.
Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.
EOF

cat > ./data/eval.txt << EOF
This is a test document for evaluation.
The model should be able to generate text similar to this.
Transformers have revolutionized natural language processing tasks.
EOF

# Start training
echo "Starting training with PyTorch..."
cd /home/jim/repos/simple_gpt

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Create a very simple model to test GPU training
python -c "
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Create output directory
os.makedirs('$OUTPUT_DIR', exist_ok=True)

# Define a simple dataset
class TextDataset(Dataset):
    def __init__(self, file_path, seq_length=10):
        with open(file_path, 'r') as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.text = text
        self.seq_length = seq_length
        self.data = []
        
        # Create samples
        for i in range(0, len(text) - seq_length, seq_length // 2):
            input_seq = text[i:i+seq_length]
            target_seq = text[i+1:i+seq_length+1]
            if len(input_seq) == seq_length and len(target_seq) == seq_length:
                self.data.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        input_tensor = torch.tensor([self.char_to_idx[c] for c in input_seq], dtype=torch.long)
        target_tensor = torch.tensor([self.char_to_idx[c] for c in target_seq], dtype=torch.long)
        return input_tensor, target_tensor

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create dataset and loader
train_dataset = TextDataset('./data/train.txt', seq_length=20)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Create model
model = SimpleModel(train_dataset.vocab_size).to(device)
if device.type == 'cuda':
    print(f'Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')

# Training parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# Training loop
start_time = time.time()
metrics = []

print('Starting training...')
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    epoch_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        outputs = outputs.reshape(-1, train_dataset.vocab_size)
        targets = targets.reshape(-1)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if (batch_idx + 1) % 5 == 0:
            print(f'  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
            if device.type == 'cuda':
                print(f'  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
    
    # Record metrics
    epoch_metrics = {
        'epoch': epoch + 1,
        'loss': epoch_loss / len(train_loader),
        'time': time.time() - start_time
    }
    metrics.append(epoch_metrics)
    print(f'  Average loss: {epoch_metrics['loss']:.4f}')
    
    # Save metrics
    with open(os.path.join('$OUTPUT_DIR', 'metrics.json'), 'w') as f:
        import json
        json.dump(metrics, f, indent=2)

# Save model
torch.save(model.state_dict(), os.path.join('$OUTPUT_DIR', 'model.pt'))

# Generate some text
model.eval()
with torch.no_grad():
    # Sample starting text
    start_text = 'Machine'
    chars = [train_dataset.char_to_idx[c] for c in start_text]
    input_tensor = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)
    
    generated_text = start_text
    for i in range(100):
        output = model(input_tensor)
        # Get last character prediction
        last_output = output[0, -1, :]
        # Sample from the output distribution
        probs = torch.softmax(last_output / 0.8, dim=0)
        next_char_idx = torch.multinomial(probs, 1).item()
        
        # Convert to character and add to result
        next_char = train_dataset.idx_to_char[next_char_idx]
        generated_text += next_char
        
        # Update input for next iteration
        input_tensor = torch.cat([
            input_tensor, 
            torch.tensor([[next_char_idx]], dtype=torch.long, device=device)
        ], dim=1)
        
        # Keep only the last 20 characters
        if input_tensor.size(1) > 20:
            input_tensor = input_tensor[:, -20:]
    
    print('Generated text:')
    print(generated_text)

# Create summary
print('Training complete!')
print(f'Total time: {time.time() - start_time:.2f} seconds')
print(f'Model saved to {os.path.join('$OUTPUT_DIR', 'model.pt')}')
print(f'Metrics saved to {os.path.join('$OUTPUT_DIR', 'metrics.json')}')
if device.type == 'cuda':
    print(f'Final GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
"

echo "Training complete. Results saved to $OUTPUT_DIR"