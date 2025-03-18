#!/usr/bin/env python

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--metrics_file", type=str, required=True,
                        help="JSON file containing training metrics")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save plots (defaults to same directory as metrics file)")
    return parser.parse_args()

def load_metrics(metrics_file):
    with open(metrics_file, 'r') as f:
        return json.load(f)

def extract_series(metrics, key):
    steps = []
    values = []
    
    for entry in metrics:
        if key in entry:
            steps.append(entry.get('step', 0))
            values.append(entry[key])
    
    return steps, values

def plot_metric(metrics, key, title, ylabel, output_file, log_scale=False):
    steps, values = extract_series(metrics, key)
    
    if not steps:
        print(f"No data points found for {key}")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, 'b-')
    plt.title(title)
    plt.xlabel('Training Steps')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    
    if log_scale and min(values) > 0:
        plt.yscale('log')
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_file}")

def plot_learning_rate(metrics, output_file):
    steps, learning_rates = extract_series(metrics, 'learning_rate')
    
    if not steps:
        print("No learning rate data points found")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, learning_rates, 'g-')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved learning rate plot to {output_file}")

def plot_combined_loss(metrics, output_file):
    train_steps, train_loss = extract_series(metrics, 'loss')
    eval_steps, eval_loss = extract_series(metrics, 'eval_loss')
    
    if not train_steps and not eval_steps:
        print("No loss data points found")
        return
    
    plt.figure(figsize=(10, 6))
    
    if train_steps:
        plt.plot(train_steps, train_loss, 'b-', label='Training Loss')
    
    if eval_steps:
        plt.plot(eval_steps, eval_loss, 'r-', label='Evaluation Loss')
    
    plt.title('Training and Evaluation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved combined loss plot to {output_file}")

def create_training_summary(metrics, output_file):
    # Extract relevant metrics
    latest_step = max([m.get('step', 0) for m in metrics])
    
    # Find the final metrics
    final_metrics = None
    for m in reversed(metrics):
        if m.get('step') == latest_step:
            final_metrics = m
            break
    
    if not final_metrics:
        print("Could not find final metrics")
        return
    
    # Calculate training time
    if 'elapsed' in final_metrics:
        elapsed_seconds = final_metrics['elapsed']
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    else:
        elapsed_str = "Unknown"
    
    # Extract key values
    perplexity = final_metrics.get('eval_perplexity', 'N/A')
    loss = final_metrics.get('loss', 'N/A')
    eval_loss = final_metrics.get('eval_loss', 'N/A')
    
    # Create the summary
    with open(output_file, 'w') as f:
        f.write("Training Summary\n")
        f.write("===============\n\n")
        f.write(f"Final Step: {latest_step}\n")
        f.write(f"Training Loss: {loss}\n")
        f.write(f"Evaluation Loss: {eval_loss}\n")
        f.write(f"Perplexity: {perplexity}\n")
        f.write(f"Total Training Time: {elapsed_str}\n")
        
        # Add timestamp
        f.write(f"\nSummary generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Saved training summary to {output_file}")

def main():
    args = parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.metrics_file)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    print(f"Loading metrics from {args.metrics_file}")
    metrics = load_metrics(args.metrics_file)
    print(f"Loaded {len(metrics)} data points")
    
    # Plot training loss
    plot_metric(
        metrics,
        'loss',
        'Training Loss',
        'Loss',
        os.path.join(args.output_dir, 'training_loss.png')
    )
    
    # Plot perplexity
    plot_metric(
        metrics,
        'eval_perplexity',
        'Evaluation Perplexity',
        'Perplexity',
        os.path.join(args.output_dir, 'perplexity.png'),
        log_scale=True
    )
    
    # Plot learning rate
    plot_learning_rate(
        metrics,
        os.path.join(args.output_dir, 'learning_rate.png')
    )
    
    # Plot combined loss
    plot_combined_loss(
        metrics,
        os.path.join(args.output_dir, 'combined_loss.png')
    )
    
    # Create training summary
    create_training_summary(
        metrics,
        os.path.join(args.output_dir, 'metrics_summary.txt')
    )
    
    print("All plots generated successfully")

if __name__ == "__main__":
    main()