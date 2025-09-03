#!/usr/bin/env python
"""
Test script for validating ToolRubric reward functions.
This script creates sample conversation trajectories and applies the ToolRubric
to verify that the reward functions return expected scores.
"""

import sys
from typing import List, Dict, Any, Callable

# Add the root directory to the path to import verifiers
sys.path.append(".")

from verifiers.rubrics.tool_rubric import ToolRubric


def create_sample_trajectories() -> List[List[Dict[str, str]]]:
    """Create sample conversation trajectories for testing."""
    
    # Trajectory 1: Correct format with successful tool use and correct answer
    trajectory1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "<reasoning>\nTo answer this question, I need to search for information about France's capital.\n</reasoning>\n<tool>\n{\"name\": \"search\", \"args\": {\"query\": \"capital of France\"}}\n</tool>"},
        {"role": "user", "content": "<result>\nThe capital of France is Paris.\n</result>"},
        {"role": "assistant", "content": "<reasoning>\nBased on the search results, I can now answer the question.\n</reasoning>\n<answer>\nParis\n</answer>"}
    ]
    
    # Trajectory 2: Incorrect format (missing tags) with successful tool use
    trajectory2 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Spain?"},
        {"role": "assistant", "content": "I'll search for information about Spain's capital.\nsearch(\"capital of Spain\")"},
        {"role": "user", "content": "<result>\nThe capital of Spain is Madrid.\n</result>"},
        {"role": "assistant", "content": "<reasoning>\nBased on the search results, I can now answer the question.\n</reasoning>\n<answer>\nMadrid\n</answer>"}
    ]
    
    # Trajectory 3: Correct format but failed tool execution
    trajectory3 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Italy?"},
        {"role": "assistant", "content": "<reasoning>\nTo answer this question, I need to search for information about Italy's capital.\n</reasoning>\n<tool>\n{\"name\": \"search\", \"args\": {\"query\": \"capital of Italy\"}}\n</tool>"},
        {"role": "user", "content": "Error: Search failed due to network issues."},
        {"role": "assistant", "content": "<reasoning>\nSince the search failed, I'll provide the answer based on my knowledge.\n</reasoning>\n<answer>\nRome\n</answer>"}
    ]
    
    # Trajectory 4: Correct format with multiline content and newlines
    trajectory4 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Germany?"},
        {"role": "assistant", "content": "<reasoning>\nTo answer this question, I need to search for information about Germany's capital.\n\nI'll use the search tool for this purpose.\n</reasoning>\n\n<tool>\n{\"name\": \"search\", \"args\": {\"query\": \"capital of Germany\"}}\n</tool>"},
        {"role": "user", "content": "<result>\nThe capital of Germany is Berlin.\n</result>"},
        {"role": "assistant", "content": "<reasoning>\nBased on the search results, I can now answer the question.\n\nThe information indicates that Berlin is the capital.\n</reasoning>\n\n<answer>\nBerlin\n</answer>"}
    ]
    
    return [trajectory1, trajectory2, trajectory3, trajectory4]


def test_tool_rubric():
    """Test the ToolRubric with sample trajectories."""
    print("Testing ToolRubric reward functions...")
    
    # Create the rubric
    rubric = ToolRubric()
    
    # Get the direct access to the reward functions
    correctness_func = rubric.correctness_reward_func
    xml_func = rubric.xml_reward_func
    format_func = rubric.format_reward_func
    tool_execution_func = rubric.tool_execution_reward_func
    
    # Create sample trajectories
    trajectories = create_sample_trajectories()
    
    # Expected answers for each trajectory
    answers = ["Paris", "Madrid", "Rome", "Berlin"]
    
    # Define expected rewards for each function and trajectory based on actual behavior
    expected_correctness = [1.0, 1.0, 1.0, 1.0]  # All have correct answers
    expected_xml = [0.4, 0.2, 0.4, 0.4]  # XML scores based on tag counting implementation
    expected_format = [0.2, 0.1, 0.2, 0.2]  # Format scores for proper message structure
    expected_tool_execution = [0.2, 0.0, 0.0, 0.2]  # Tool execution as expected
    
    # Keep track of test results
    tests_passed = 0
    total_tests = 0
    
    # Test correctness_reward_func
    print("\n1. Testing correctness_reward_func...")
    correctness_rewards = correctness_func(completions=trajectories, answer=answers)
    for i, (expected, actual) in enumerate(zip(expected_correctness, correctness_rewards)):
        total_tests += 1
        result = abs(expected - actual) < 0.01  # Allow small floating-point differences
        tests_passed += 1 if result else 0
        if not result:
            print(f"  ❌ Trajectory {i+1}: expected={expected:.2f}, actual={actual:.2f}")
        else:
            print(f"  ✓ Trajectory {i+1}")
    
    # Test xml_reward_func
    print("\n2. Testing xml_reward_func...")
    xml_rewards = xml_func(completions=trajectories)
    for i, (expected, actual) in enumerate(zip(expected_xml, xml_rewards)):
        total_tests += 1
        result = abs(expected - actual) < 0.01
        tests_passed += 1 if result else 0
        if not result:
            print(f"  ❌ Trajectory {i+1}: expected={expected:.2f}, actual={actual:.2f}")
        else:
            print(f"  ✓ Trajectory {i+1}")
    
    # Test format_reward_func
    print("\n3. Testing format_reward_func...")
    format_rewards = format_func(completions=trajectories)
    for i, (expected, actual) in enumerate(zip(expected_format, format_rewards)):
        total_tests += 1
        result = abs(expected - actual) < 0.01
        tests_passed += 1 if result else 0
        if not result:
            print(f"  ❌ Trajectory {i+1}: expected={expected:.2f}, actual={actual:.2f}")
        else:
            print(f"  ✓ Trajectory {i+1}")
    
    # Test tool_execution_reward_func
    print("\n4. Testing tool_execution_reward_func...")
    tool_rewards = tool_execution_func(completions=trajectories)
    for i, (expected, actual) in enumerate(zip(expected_tool_execution, tool_rewards)):
        total_tests += 1
        result = abs(expected - actual) < 0.01
        tests_passed += 1 if result else 0
        if not result:
            print(f"  ❌ Trajectory {i+1}: expected={expected:.2f}, actual={actual:.2f}")
        else:
            print(f"  ✓ Trajectory {i+1}")
    
    # Print summary
    print(f"\nSummary: {tests_passed}/{total_tests} tests passed")
    if tests_passed == total_tests:
        print("✅ All reward functions are working correctly!")
    else:
        print("❌ Some reward functions are not returning expected values.")


if __name__ == "__main__":
    test_tool_rubric() 