#!/usr/bin/env python

import os
import torch
from torch import nn
import time

# Set environment variables for HuggingFace cache
os.environ["HF_HOME"] = "./cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "./cache/datasets"

# Create a simple MLP model for testing
class MLP(nn.Module):
    def __init__(self, input_size=100, hidden_size=1000, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # Print PyTorch and CUDA information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Create a model
    print("Creating model...")
    model = MLP(hidden_size=10000)  # Large hidden size to stress GPU
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    if device.type == "cuda":
        print(f"Initial GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Create dummy data
    print("Creating data...")
    batch_size = 64
    input_data = torch.randn(batch_size, 100, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few steps
    print("Training...")
    num_steps = 100
    start_time = time.time()
    
    for step in range(num_steps):
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            # Print statistics
            print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")
            
            if device.type == "cuda":
                # Print GPU statistics
                print(f"  GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                # Only print utilization if pynvml is available
                try:
                    print(f"  GPU utilization: {torch.cuda.utilization(0)}%")
                except (ModuleNotFoundError, AttributeError):
                    pass
    
    # Calculate training speed
    end_time = time.time()
    elapsed_time = end_time - start_time
    steps_per_second = num_steps / elapsed_time
    
    print(f"Training complete. Total time: {elapsed_time:.2f} seconds")
    print(f"Training speed: {steps_per_second:.2f} steps/second")
    
    if device.type == "cuda":
        print(f"Final GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()