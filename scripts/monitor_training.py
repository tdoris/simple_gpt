#!/usr/bin/env python

import os
import json
import argparse
import time
from datetime import datetime, timedelta
import curses
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--metrics_file", type=str, required=True,
                        help="JSON file containing training metrics")
    parser.add_argument("--refresh_rate", type=int, default=10,
                        help="Refresh rate in seconds")
    return parser.parse_args()

def load_metrics(metrics_file):
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def format_time(seconds):
    if seconds is None:
        return "Unknown"
    
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}h:{int(minutes):02d}m:{int(seconds):02d}s"

def monitor_training(stdscr, args):
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_RED, -1)
    
    # Set up colors
    GREEN = curses.color_pair(1)
    CYAN = curses.color_pair(2)
    YELLOW = curses.color_pair(3)
    RED = curses.color_pair(4)
    
    # Set up the screen
    max_y, max_x = stdscr.getmaxyx()
    
    start_time = datetime.now()
    last_known_timestamp = None
    
    while True:
        # Clear the screen
        stdscr.clear()
        
        # Draw title
        title = "SimpleGPT Training Monitor"
        stdscr.addstr(0, max(0, (max_x - len(title)) // 2), title, curses.A_BOLD)
        
        # Draw metrics file path
        stdscr.addstr(1, 0, f"Monitoring: {args.metrics_file}")
        
        # Load metrics
        metrics = load_metrics(args.metrics_file)
        
        # Draw monitoring time
        monitoring_time = datetime.now() - start_time
        stdscr.addstr(2, 0, f"Monitoring time: {format_time(monitoring_time.total_seconds())}")
        
        if not metrics:
            stdscr.addstr(4, 0, "No metrics data found. Waiting for training to start...", YELLOW)
        else:
            # Get latest metrics
            latest = metrics[-1]
            
            # Check if there are new updates
            is_active = True
            if 'timestamp' in latest:
                current_timestamp = latest['timestamp']
                if last_known_timestamp is not None and current_timestamp == last_known_timestamp:
                    # No new updates
                    time_since_update = datetime.now() - datetime.fromtimestamp(current_timestamp)
                    if time_since_update.total_seconds() > 300:  # 5 minutes
                        is_active = False
                last_known_timestamp = current_timestamp
            
            # Display training status
            row = 4
            if is_active:
                stdscr.addstr(row, 0, "Status: ACTIVE", GREEN)
            else:
                stdscr.addstr(row, 0, "Status: INACTIVE (no updates for >5min)", YELLOW)
            row += 1
            
            # Display step information
            step = latest.get('step', 0)
            stdscr.addstr(row, 0, f"Current step: {step}", CYAN)
            row += 2
            
            # Display loss information
            loss = latest.get('loss')
            if loss is not None:
                stdscr.addstr(row, 0, f"Training loss: {loss:.6f}")
                row += 1
            
            eval_loss = latest.get('eval_loss')
            if eval_loss is not None:
                stdscr.addstr(row, 0, f"Evaluation loss: {eval_loss:.6f}")
                row += 1
            
            perplexity = latest.get('eval_perplexity')
            if perplexity is not None:
                perplexity_str = f"{perplexity:.2f}"
                if perplexity > 1000:
                    stdscr.addstr(row, 0, f"Perplexity: {perplexity_str}", RED)
                elif perplexity > 100:
                    stdscr.addstr(row, 0, f"Perplexity: {perplexity_str}", YELLOW)
                else:
                    stdscr.addstr(row, 0, f"Perplexity: {perplexity_str}", GREEN)
                row += 1
            
            row += 1
            
            # Display training time information
            elapsed = latest.get('elapsed')
            if elapsed is not None:
                stdscr.addstr(row, 0, f"Training time: {format_time(elapsed)}")
                row += 1
            
            remaining = latest.get('estimated_remaining')
            if remaining is not None:
                stdscr.addstr(row, 0, f"Estimated remaining: {remaining}")
                row += 1
            
            # Display learning rate
            lr = latest.get('learning_rate')
            if lr is not None:
                stdscr.addstr(row, 0, f"Learning rate: {lr:.8f}")
                row += 1
            
            # Display training speed
            speed = latest.get('samples_per_second')
            if speed is not None:
                stdscr.addstr(row, 0, f"Training speed: {speed:.2f} samples/second")
                row += 1
            
            # Display epoch information
            epoch = latest.get('epoch')
            if epoch is not None:
                stdscr.addstr(row, 0, f"Current epoch: {epoch:.2f}")
                row += 1
        
        # Draw footer
        footer = "Press 'q' to exit"
        stdscr.addstr(max_y-1, 0, footer)
        stdscr.refresh()
        
        # Check for 'q' key press (with timeout)
        stdscr.timeout(1000)  # 1 second timeout
        key = stdscr.getch()
        if key == ord('q'):
            break
        
        # Sleep until next refresh
        time.sleep(args.refresh_rate)

def main():
    args = parse_args()
    
    # Check if the metrics file directory exists
    metrics_dir = os.path.dirname(args.metrics_file)
    if metrics_dir and not os.path.exists(metrics_dir):
        print(f"Directory does not exist: {metrics_dir}")
        return
    
    print(f"Starting monitor for {args.metrics_file}")
    print("Press Ctrl+C to exit")
    
    try:
        curses.wrapper(monitor_training, args)
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")

if __name__ == "__main__":
    main()