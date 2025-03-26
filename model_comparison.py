#!/usr/bin/env python
"""
Model Comparison Tool

This script compares the performance of two language models using Claude 3.5 Sonnet
as an evaluator. It tests the models on various categories of prompts and generates
a comprehensive performance scorecard.
"""

import os
import json
import time
import random
import argparse
import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from anthropic import AsyncAnthropic

# Define prompt categories with example prompts for each
PROMPT_CATEGORIES = {
    "Factual Knowledge": [
        "What is the capital of France and what's a notable landmark there?",
        "Explain how photosynthesis works in plants.",
        "Who was Marie Curie and what were her major contributions to science?",
        "What are the main causes of climate change?",
        "Describe the structure of a human cell and its key components.",
        "What happened during the Cuban Missile Crisis?",
        "Explain the difference between RAM and ROM in computing.",
        "Who wrote 'Pride and Prejudice' and when was it published?",
        "What is the process of plate tectonics?",
        "Explain the three branches of government in the United States."
    ],
    "Reasoning & Problem Solving": [
        "If a train travels at 60 mph, how long will it take to travel 150 miles?",
        "A ball is thrown upward with an initial velocity of 20 m/s. How high will it go?",
        "Five people need to cross a bridge at night with one flashlight. The bridge can only hold two people at a time. They take 1, 2, 5, 8, and 12 minutes respectively to cross. What's the minimum time needed for all to cross?",
        "In a room of 30 people, what's the probability that at least two share a birthday?",
        "If water is flowing into a tank at 10 gallons per minute and draining at 6 gallons per minute, how long will it take to fill a 100-gallon tank that's initially empty?",
        "Two trains are 200 miles apart and traveling toward each other. One train is moving at 70 mph and the other at 55 mph. How long until they meet?",
        "What logical fallacy is present in the statement: 'All politicians are corrupt because I've never met one who wasn't'?",
        "In a certain company, 40% of employees are women. If 25% of the women and 20% of the men are in management, what percentage of managers are women?",
        "A car depreciates in value by 15% each year. After how many years will its value be less than half its original value?",
        "If I flip a fair coin 10 times, what's the probability of getting exactly 6 heads?"
    ],
    "Creative Writing": [
        "Write a short story about a lost key that opens a mysterious door.",
        "Compose a poem about autumn leaves.",
        "Write a dialogue between the ocean and the moon.",
        "Create a description of an alien landscape unlike anything on Earth.",
        "Write the opening paragraph of a mystery novel set in Victorian London.",
        "Compose a letter from a time traveler who has just arrived from the year 3000.",
        "Write a fairy tale about a clever fox who outsmarts a dragon.",
        "Create a scene where two strangers connect during a power outage in a big city.",
        "Write a monologue from the perspective of an ancient redwood tree.",
        "Compose a short story where the main character discovers they can suddenly understand animal language."
    ],
    "Summarization & Analysis": [
        "Summarize the key arguments for and against nuclear energy.",
        "Explain the importance of The Great Gatsby in American literature.",
        "Summarize the major events of World War II in under 200 words.",
        "Analyze the causes and effects of the 2008 financial crisis.",
        "Provide a summary of Darwin's theory of evolution and its significance.",
        "Explain the key differences between capitalism and socialism as economic systems.",
        "Summarize the plot and themes of Shakespeare's 'Hamlet'.",
        "Provide an overview of the history and impact of the internet.",
        "Summarize the current understanding of quantum physics for a general audience.",
        "Analyze the global impact of the COVID-19 pandemic on healthcare systems."
    ],
    "Ethical Reasoning": [
        "Is it ethical to use AI to replace human workers? Consider multiple perspectives.",
        "Discuss the ethical implications of genetic engineering in humans.",
        "What ethical considerations should guide the development of autonomous vehicles?",
        "Is it ever justified to limit freedom of speech? Explain your reasoning.",
        "Discuss the ethics of animal testing for medical research.",
        "What ethical principles should guide the use of personal data by companies?",
        "Discuss the ethics of wealth inequality in modern society.",
        "Is there an ethical obligation to help those in poverty in other countries?",
        "What ethical considerations should guide end-of-life care decisions?",
        "Discuss the ethics of geoengineering as a solution to climate change."
    ]
}

@dataclass
class ModelResponse:
    """Class to store a model's response to a prompt."""
    prompt: str
    category: str
    response: str
    model_name: str
    response_time: float

@dataclass
class EvaluationResult:
    """Class to store Claude's evaluation of a pair of responses."""
    prompt: str
    category: str
    model_a_name: str
    model_b_name: str
    model_a_response: str
    model_b_response: str
    winner: str  # Either model_a_name, model_b_name, or "tie"
    reasoning: str
    score_a: int  # 1 if A wins, 0.5 for tie, 0 if B wins
    score_b: int  # 1 if B wins, 0.5 for tie, 0 if A wins

@dataclass
class ComparisonResults:
    """Class to store all evaluation results and summary statistics."""
    model_a_name: str
    model_b_name: str
    evaluations: List[EvaluationResult] = field(default_factory=list)
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_total_scores(self) -> Tuple[float, float]:
        """Get the total scores for both models."""
        score_a = sum(eval.score_a for eval in self.evaluations)
        score_b = sum(eval.score_b for eval in self.evaluations)
        return score_a, score_b
    
    def get_category_scores(self) -> Dict[str, Tuple[float, float]]:
        """Get scores broken down by category."""
        categories = {}
        for category in set(eval.category for eval in self.evaluations):
            cat_evals = [e for e in self.evaluations if e.category == category]
            score_a = sum(e.score_a for e in cat_evals)
            score_b = sum(e.score_b for e in cat_evals)
            categories[category] = (score_a, score_b)
        return categories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for JSON serialization."""
        return {
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "timestamp": self.timestamp,
            "total_prompts": len(self.evaluations),
            "total_score_a": self.get_total_scores()[0],
            "total_score_b": self.get_total_scores()[1],
            "category_scores": {
                category: {"score_a": scores[0], "score_b": scores[1]}
                for category, scores in self.get_category_scores().items()
            },
            "evaluations": [
                {
                    "prompt": e.prompt,
                    "category": e.category,
                    "winner": e.winner,
                    "score_a": e.score_a,
                    "score_b": e.score_b,
                    "model_a_response": e.model_a_response[:200] + "..." if len(e.model_a_response) > 200 else e.model_a_response,
                    "model_b_response": e.model_b_response[:200] + "..." if len(e.model_b_response) > 200 else e.model_b_response,
                    "reasoning": e.reasoning[:200] + "..." if len(e.reasoning) > 200 else e.reasoning
                }
                for e in self.evaluations
            ]
        }
    
    def save_to_file(self, filename: str = None) -> str:
        """Save results to a JSON file and return the filename."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filename

    def generate_report(self) -> str:
        """Generate a human-readable report of the comparison results."""
        total_a, total_b = self.get_total_scores()
        total_prompts = len(self.evaluations)
        
        report = []
        report.append("=" * 80)
        report.append(f"MODEL COMPARISON REPORT: {self.model_a_name} vs {self.model_b_name}")
        report.append(f"Date: {self.timestamp}")
        report.append("=" * 80)
        
        report.append("\nOVERALL RESULTS:")
        report.append("-" * 80)
        report.append(f"{self.model_a_name}: {total_a}/{total_prompts} ({total_a/total_prompts:.1%})")
        report.append(f"{self.model_b_name}: {total_b}/{total_prompts} ({total_b/total_prompts:.1%})")
        
        if total_a > total_b:
            report.append(f"\n🏆 WINNER: {self.model_a_name} (by {total_a - total_b} points)")
        elif total_b > total_a:
            report.append(f"\n🏆 WINNER: {self.model_b_name} (by {total_b - total_a} points)")
        else:
            report.append("\n🏆 RESULT: TIE")
        
        report.append("\nRESULTS BY CATEGORY:")
        report.append("-" * 80)
        category_scores = self.get_category_scores()
        for category, (score_a, score_b) in sorted(category_scores.items()):
            prompts_in_category = len([e for e in self.evaluations if e.category == category])
            report.append(f"\n{category} (based on {prompts_in_category} prompts):")
            report.append(f"  {self.model_a_name}: {score_a}/{prompts_in_category} ({score_a/prompts_in_category:.1%})")
            report.append(f"  {self.model_b_name}: {score_b}/{prompts_in_category} ({score_b/prompts_in_category:.1%})")
            
            if score_a > score_b:
                report.append(f"  Category Winner: {self.model_a_name}")
            elif score_b > score_a:
                report.append(f"  Category Winner: {self.model_b_name}")
            else:
                report.append(f"  Category Result: Tie")
        
        # Sample evaluations for each category
        report.append("\nSAMPLE EVALUATIONS:")
        report.append("-" * 80)
        for category in sorted(category_scores.keys()):
            cat_evals = [e for e in self.evaluations if e.category == category]
            if cat_evals:
                # Pick a sample evaluation (preferably where there was a clear winner)
                sample = next((e for e in cat_evals if e.winner != "tie"), random.choice(cat_evals))
                report.append(f"\n[Sample from {category}]")
                report.append(f"Prompt: {sample.prompt}")
                report.append(f"{self.model_a_name} response: {sample.model_a_response[:150]}...")
                report.append(f"{self.model_b_name} response: {sample.model_b_response[:150]}...")
                report.append(f"Winner: {sample.winner}")
                report.append(f"Reasoning: {sample.reasoning[:200]}...")
        
        return "\n".join(report)


class ModelComparer:
    """Class to handle the comparison of two models using Claude as an evaluator."""
    
    def __init__(
        self, 
        model_a_path: str, 
        model_b_path: str,
        anthropic_api_key: str,
        num_prompts_per_category: int = 1,  # Reduced to 1 for quick testing
        num_categories: int = 5  # Default to all categories
    ):
        self.model_a_path = model_a_path
        self.model_b_path = model_b_path
        # Extract model names from paths for display
        self.model_a_name = os.path.basename(model_a_path.rstrip('/'))
        self.model_b_name = os.path.basename(model_b_path.rstrip('/'))
        self.num_prompts_per_category = num_prompts_per_category
        self.num_categories = min(num_categories, 5)  # Cap at 5 categories
        self.results = ComparisonResults(model_a_name=self.model_a_name, model_b_name=self.model_b_name)
        
        # Initialize Anthropic client for Claude
        self.claude_client = AsyncAnthropic(api_key=anthropic_api_key)
    
    def select_prompts(self) -> List[Tuple[str, str]]:
        """Select prompts from each category for the comparison."""
        selected_prompts = []
        
        # Get a list of categories to use
        categories = list(PROMPT_CATEGORIES.keys())
        
        # Limit to the requested number of categories
        if self.num_categories < len(categories):
            print(f"Limiting evaluation to {self.num_categories} out of {len(categories)} categories")
            categories = random.sample(categories, self.num_categories)
        
        # Select prompts from each category
        for category in categories:
            prompts = PROMPT_CATEGORIES[category]
            
            # Choose the minimum of available prompts and requested number
            num_to_select = min(self.num_prompts_per_category, len(prompts))
            
            # Randomly select prompts from this category
            category_prompts = random.sample(prompts, num_to_select)
            
            # Add the category and prompt to our list
            for prompt in category_prompts:
                selected_prompts.append((category, prompt))
                
        print(f"Selected {len(selected_prompts)} prompts from {len(categories)} categories")
        
        return selected_prompts
    
    async def get_model_response(self, model_path: str, prompt: str) -> str:
        """
        Get a response from a local SimpleGPT model.
        """
        print(f"Getting response from model at {model_path} for prompt: {prompt[:50]}...")
        
        try:
            # We'll use subprocess to call our generate.py script
            import subprocess
            import tempfile
            
            # Create a temporary file to store the output
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
                tmp_name = tmp.name
            
            # Build the command to run our text generation script
            cmd = [
                "python", "-m", "simple_gpt.scripts.generate",
                "--model_path", model_path,
                "--prompt", prompt,
                "--max_length", "150",
                "--temperature", "0.7",
                "--top_k", "40",
                "--top_p", "0.95",
                "--repetition_penalty", "1.2",
                "--do_sample"
            ]
            
            # Run the command and capture the output
            print(f"Running command: {' '.join(cmd)}")
            with open(tmp_name, 'w') as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
            
            # Read the output from the temporary file
            with open(tmp_name, 'r') as f:
                output = f.read()
            
            # Clean up
            os.unlink(tmp_name)
            
            # Extract the generated text from the output
            # The output contains some debug information and the generated text
            # We need to extract just the generated text
            lines = output.strip().split('\n')
            generated_text = ""
            
            # Look for the line that starts with "Generated sequence 1:"
            for i, line in enumerate(lines):
                if "Generated sequence 1:" in line:
                    # Get the next line, which is the actual generated text
                    if i + 1 < len(lines):
                        generated_text = lines[i + 1]
                        break
            
            # If we couldn't find the generated text, try to extract any useful content
            if not generated_text:
                # Look for common debug output pattern
                for line in lines:
                    if "Model vocabulary size" not in line and line and not line.startswith("03/"):
                        generated_text = line
                        break
                
                # If still no useful text, use a subset of the full output
                if not generated_text:
                    generated_text = output[:500] if len(output) > 500 else output
            
            # Clean up the text and make sure we have something usable
            generated_text = generated_text.strip()
            if not generated_text:
                generated_text = f"[Model at {model_path} failed to generate a readable response]"
                
            return generated_text
            
        except Exception as e:
            print(f"Error generating response from {model_path}: {e}")
            return f"[Error generating response from {model_path}: {e}]"
    
    async def evaluate_responses(
        self, 
        prompt: str, 
        category: str,
        model_a_response: str, 
        model_b_response: str
    ) -> EvaluationResult:
        """Use Claude to evaluate the responses from both models."""
        # Randomize the order to avoid positional bias
        is_reversed = random.choice([True, False])
        
        if is_reversed:
            first_model, second_model = self.model_b_name, self.model_a_name
            first_response, second_response = model_b_response, model_a_response
        else:
            first_model, second_model = self.model_a_name, self.model_b_name
            first_response, second_response = model_a_response, model_b_response
        
        evaluation_prompt = f"""
        You are an expert evaluator of AI model outputs. You'll compare responses from two AI models to the same prompt.
        
        Prompt category: {category}
        User prompt: {prompt}
        
        Model {first_model} response: {first_response}
        
        Model {second_model} response: {second_response}
        
        Evaluate which response is better based on the following criteria:
        1. Accuracy and factual correctness
        2. Clarity and coherence
        3. Helpfulness and relevance to the prompt
        4. Depth and comprehensiveness
        5. Safety and ethical considerations (when applicable)
        
        First, analyze both responses point by point against each criterion.
        Then provide your final verdict on which response is better overall.
        If both responses are of equal quality, you may declare a tie.
        
        Your evaluation should be thorough, fair, and unbiased. Focus on the quality of the responses, not their style or tone.
        
        End your evaluation with exactly one of these statements:
        - "Winner: {first_model}"
        - "Winner: {second_model}"
        - "Winner: Tie"
        """
        
        print(f"Evaluating responses for prompt: {prompt[:50]}...")
        
        # Add retry logic for rate limits
        max_retries = 3
        retry_delay = 20  # start with 20 seconds
        
        for retry in range(max_retries):
            try:
                response = await self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=4000,
                    messages=[
                        {"role": "user", "content": evaluation_prompt}
                    ]
                )
                break  # Success, exit retry loop
            except Exception as e:
                if "rate_limit" in str(e).lower() and retry < max_retries - 1:
                    # If it's a rate limit error and we have retries left
                    wait_time = retry_delay * (retry + 1)  # Exponential backoff
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry {retry+1}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                else:
                    # If it's not a rate limit error or we're out of retries, raise the exception
                    raise
        
        evaluation_text = response.content[0].text
        
        # Extract the winner from the evaluation
        if f"Winner: {first_model}" in evaluation_text:
            winner = first_model
        elif f"Winner: {second_model}" in evaluation_text:
            winner = second_model
        else:
            winner = "tie"
            
        # Convert the winner to actual model names (accounting for potential order reversal)
        if is_reversed:
            if winner == self.model_b_name:
                actual_winner = self.model_b_name
                score_a, score_b = 0, 1
            elif winner == self.model_a_name:
                actual_winner = self.model_a_name
                score_a, score_b = 1, 0
            else:  # tie
                actual_winner = "tie"
                score_a, score_b = 0.5, 0.5
        else:
            if winner == self.model_a_name:
                actual_winner = self.model_a_name
                score_a, score_b = 1, 0
            elif winner == self.model_b_name:
                actual_winner = self.model_b_name
                score_a, score_b = 0, 1
            else:  # tie
                actual_winner = "tie"
                score_a, score_b = 0.5, 0.5
                
        return EvaluationResult(
            prompt=prompt,
            category=category,
            model_a_name=self.model_a_name,
            model_b_name=self.model_b_name,
            model_a_response=model_a_response,
            model_b_response=model_b_response,
            winner=actual_winner,
            reasoning=evaluation_text,
            score_a=score_a,
            score_b=score_b
        )
    
    async def process_single_prompt(self, category: str, prompt: str) -> EvaluationResult:
        """Process a single prompt through both models and evaluation."""
        # Get responses from both models
        start_time_a = time.time()
        model_a_response = await self.get_model_response(self.model_a_path, prompt)
        response_time_a = time.time() - start_time_a
        
        start_time_b = time.time()
        model_b_response = await self.get_model_response(self.model_b_path, prompt)
        response_time_b = time.time() - start_time_b
        
        # Have Claude evaluate the responses
        evaluation = await self.evaluate_responses(
            prompt=prompt,
            category=category,
            model_a_response=model_a_response,
            model_b_response=model_b_response
        )
        
        return evaluation
    
    async def run_comparison(self) -> ComparisonResults:
        """Run the full comparison and return the results."""
        # Select prompts to use
        selected_prompts = self.select_prompts()
        
        # Process prompts sequentially with rate limiting to avoid API rate limits
        evaluations = []
        for i, (category, prompt) in enumerate(selected_prompts):
            print(f"Processing prompt {i+1}/{len(selected_prompts)}: {category}")
            
            try:
                # Process one prompt at a time to avoid rate limiting
                evaluation = await self.process_single_prompt(category, prompt)
                evaluations.append(evaluation)
                
                # Add a delay between API calls to avoid rate limits
                # Only add delay if there are more prompts to process
                if i < len(selected_prompts) - 1:
                    delay = 5  # 5 seconds between evaluations
                    print(f"Rate limiting: waiting {delay} seconds before next evaluation...")
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                print(f"Error processing prompt {i+1}: {e}")
                print("Waiting 60 seconds before retrying...")
                await asyncio.sleep(60)  # Wait longer on error
                
                try:
                    # Retry once
                    print(f"Retrying prompt {i+1}...")
                    evaluation = await self.process_single_prompt(category, prompt)
                    evaluations.append(evaluation)
                except Exception as e2:
                    print(f"Failed to process prompt after retry: {e2}")
                    # Create a dummy evaluation showing the error
                    dummy_eval = EvaluationResult(
                        prompt=prompt,
                        category=category,
                        model_a_name=self.model_a_name,
                        model_b_name=self.model_b_name,
                        model_a_response="[Error generating response]",
                        model_b_response="[Error generating response]",
                        winner="tie",
                        reasoning=f"Evaluation failed due to error: {e2}",
                        score_a=0.5,
                        score_b=0.5
                    )
                    evaluations.append(dummy_eval)
        
        # Update results with evaluations
        self.results.evaluations = evaluations
        
        return self.results


async def main():
    parser = argparse.ArgumentParser(description="Compare two AI models using Claude as an evaluator")
    parser.add_argument("--model-a", type=str, required=True, help="Path to the first model")
    parser.add_argument("--model-b", type=str, required=True, help="Path to the second model")
    parser.add_argument("--api-key", type=str, help="Anthropic API key")
    parser.add_argument("--prompts-per-category", type=int, default=1, 
                        help="Number of prompts to use per category (max 10)")
    parser.add_argument("--categories", type=int, default=5,
                        help="Number of categories to test (max 5, default: all 5)")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Get API key from arguments or environment
    anthropic_api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        print("Error: Anthropic API key not provided. Use --api-key or set ANTHROPIC_API_KEY environment variable.")
        return
    
    # Validate prompts per category
    if args.prompts_per_category < 1 or args.prompts_per_category > 10:
        print("Error: prompts-per-category must be between 1 and 10")
        return
        
    # Validate categories
    if args.categories < 1 or args.categories > 5:
        print("Error: categories must be between 1 and 5")
        return
    
    print(f"Starting comparison of {args.model_a} vs {args.model_b}")
    print(f"Using {args.prompts_per_category} prompts per category across up to {args.categories} categories")
    
    # Create comparer and run comparison
    comparer = ModelComparer(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        anthropic_api_key=anthropic_api_key,
        num_prompts_per_category=args.prompts_per_category,
        num_categories=args.categories
    )
    
    results = await comparer.run_comparison()
    
    # Save detailed results to file
    output_file = args.output or None
    saved_file = results.save_to_file(output_file)
    print(f"Detailed results saved to {saved_file}")
    
    # Print summary report
    report = results.generate_report()
    print(report)


if __name__ == "__main__":
    asyncio.run(main())