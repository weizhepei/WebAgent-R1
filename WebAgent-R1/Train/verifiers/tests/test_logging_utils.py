#!/usr/bin/env python
"""
Test script for the print_prompt_completions_sample function.
This visualizes how the completions formatting works with different types of inputs.
"""

import sys
from typing import List, Dict, Any, Union

# Make sure rich is installed
try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    print("Rich is required for this test. Please install it with: uv add rich")
    sys.exit(1)

# Import the function to test
from verifiers.utils.logging_utils import print_prompt_completions_sample

def test_single_message_dict():
    """Test with a single message dictionary in completions."""
    print("\n=== TEST: Single Message Dictionary ===\n")
    
    prompts = ["Tell me a joke about programming."]
    completions = [
        {
            "role": "assistant",
            "content": "Why do programmers prefer dark mode? Because light attracts bugs!"
        }
    ]
    rewards = [0.95]
    step = 1
    
    print_prompt_completions_sample(prompts, completions, rewards, step)

def test_list_of_messages():
    """Test with a list of message dictionaries in completions."""
    print("\n=== TEST: List of Messages ===\n")
    
    prompts = ["Explain quantum computing in simple terms."]
    completions = [
        [
            {"role": "assistant", "content": "Quantum computing is like having a super-powered calculator that can try many answers at once instead of one at a time."},
            {"role": "user", "content": "Can you give me an analogy?"},
            {"role": "assistant", "content": "Imagine searching for a name in a phone book. A classical computer would check each name one-by-one, but a quantum computer is like looking at all pages simultaneously."}
        ]
    ]
    rewards = [0.87]
    step = 2
    
    print_prompt_completions_sample(prompts, completions, rewards, step) # type: ignore

def test_string_fallback():
    """Test the fallback case with string completions."""
    print("\n=== TEST: String Fallback ===\n")
    
    prompts = ["What is the capital of France?"]
    completions = ["Paris is the capital of France."]  # This will trigger the fallback
    rewards = [0.75]
    step = 3
    
    # Note: This should use the fallback case in the function
    print_prompt_completions_sample(prompts, completions, rewards, step)  # type: ignore

def test_multiple_examples():
    """Test with multiple examples in one call."""
    print("\n=== TEST: Multiple Examples ===\n")
    
    prompts = [
        "What is machine learning?",
        "Write a haiku about programming."
    ]
    
    completions = [
        {"role": "assistant", "content": "Machine learning is a branch of AI that enables computers to learn from data without being explicitly programmed."},
        {"role": "assistant", "content": "Fingers on keyboard\nCode flows like a gentle stream\nBugs appear upstream"}
    ]
    
    rewards = [0.92, 0.88]
    step = 4
    
    print_prompt_completions_sample(prompts, completions, rewards, step)

def main():
    """Run all test cases."""
    console = Console()
    console.print(Panel.fit("Testing print_prompt_completions_sample function", style="bold green"))
    
    test_single_message_dict()
    test_list_of_messages()
    test_string_fallback()
    test_multiple_examples()
    
    console.print(Panel.fit("All tests completed", style="bold green"))

if __name__ == "__main__":
    main() 