#!/bin/bash

# Script to run simple training with test data directly (without screen)

# Create output directory
OUTPUT_DIR="./output/simple_training_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUTPUT_DIR/training.log"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p ./data
rm -f ./output/latest_simple_training 2>/dev/null
ln -sf "$OUTPUT_DIR" ./output/latest_simple_training

# Test GPU
echo "Testing GPU..."
python test_gpu.py

# Create test data if it doesn't exist
if [ ! -f "./data/train.txt" ]; then
  echo "Creating test data..."
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
fi

# Start simple training with direct output
echo "Starting training at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
export CUDA_VISIBLE_DEVICES=0

# Run the simple training script and output to log file
(
  python -c "
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import json
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
num_epochs = 100  # Extended for longer training

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
            # Calculate time elapsed and estimated remaining time
            elapsed = time.time() - start_time
            progress = (epoch * len(train_loader) + batch_idx) / (num_epochs * len(train_loader))
            if progress > 0:
                eta = elapsed / progress - elapsed
                print(f'  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, ETA: {int(eta/60)}m {int(eta%60)}s')
            else:
                print(f'  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            if device.type == 'cuda':
                print(f'  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
            
            # Log metrics
            step_metrics = {
                'step': epoch * len(train_loader) + batch_idx,
                'loss': loss.item(),
                'epoch': epoch + 1,
                'elapsed': elapsed,
                'timestamp': time.time()
            }
            metrics.append(step_metrics)
            
            # Save metrics periodically
            with open(os.path.join('$OUTPUT_DIR', 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save checkpoint periodically
            if (epoch * len(train_loader) + batch_idx) % 100 == 0 and epoch > 0:
                checkpoint_dir = os.path.join('$OUTPUT_DIR', f'checkpoint-{epoch * len(train_loader) + batch_idx}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pt'))
    
    # Record metrics for epoch
    epoch_metrics = {
        'step': (epoch + 1) * len(train_loader),
        'epoch': epoch + 1,
        'loss': epoch_loss / len(train_loader),
        'time': time.time() - start_time,
        'timestamp': time.time()
    }
    metrics.append(epoch_metrics)
    print(f'  Average loss: {epoch_metrics[\"loss\"]:.4f}')
    
    # Save metrics for epoch
    with open(os.path.join('$OUTPUT_DIR', 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate sample text every 10 epochs
    if (epoch + 1) % 10 == 0:
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
            
            # Save generated text
            with open(os.path.join('$OUTPUT_DIR', f'sample_epoch_{epoch+1}.txt'), 'w') as f:
                f.write(generated_text)
        model.train()

# Save final model
torch.save(model.state_dict(), os.path.join('$OUTPUT_DIR', 'model.pt'))

# Generate final text sample
model.eval()
with torch.no_grad():
    # Sample starting text
    start_text = 'Machine'
    chars = [train_dataset.char_to_idx[c] for c in start_text]
    input_tensor = torch.tensor(chars, dtype=torch.long).unsqueeze(0).to(device)
    
    generated_text = start_text
    for i in range(200):  # Generate longer text for final sample
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
    
    print('Final generated text:')
    print(generated_text)
    
    # Save generated text
    with open(os.path.join('$OUTPUT_DIR', 'final_sample.txt'), 'w') as f:
        f.write(generated_text)

# Create summary
print('Training complete!')
print(f'Total time: {time.time() - start_time:.2f} seconds')
print(f'Model saved to {os.path.join('$OUTPUT_DIR', 'model.pt')}')
print(f'Metrics saved to {os.path.join('$OUTPUT_DIR', 'metrics.json')}')
if device.type == 'cuda':
    print(f'Final GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')

# Save training summary
with open(os.path.join('$OUTPUT_DIR', 'training_summary.txt'), 'w') as f:
    f.write(f'Training Summary\n')
    f.write(f'===============\n\n')
    f.write(f'Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n')
    f.write(f'Total epochs: {num_epochs}\n')
    f.write(f'Total training time: {time.time() - start_time:.2f} seconds\n')
    f.write(f'Final loss: {metrics[-1][\"loss\"]:.4f}\n\n')
    f.write(f'Model architecture:\n')
    f.write(f'- Embedding dimension: 128\n')
    f.write(f'- Hidden dimension: 256\n')
    f.write(f'- Vocabulary size: {train_dataset.vocab_size}\n\n')
    f.write(f'Device used: {device}\n')
  " 
) > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!
echo "Training process started with PID $TRAINING_PID"
echo "To monitor progress, run: ./check_progress.sh $OUTPUT_DIR/metrics.json"
echo "To view the log: tail -f $LOG_FILE"