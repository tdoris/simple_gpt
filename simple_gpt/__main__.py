#!/usr/bin/env python

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Simple GPT CLI interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--args", nargs="*", help="Arguments to pass to train.py")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text with a trained model")
    generate_parser.add_argument("--args", nargs="*", help="Arguments to pass to generate.py")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "train":
        from simple_gpt.scripts.train import main as train_main
        sys.argv = [sys.argv[0]] + (args.args or [])
        train_main()
    elif args.command == "generate":
        from simple_gpt.scripts.generate import main as generate_main
        sys.argv = [sys.argv[0]] + (args.args or [])
        generate_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
