"""
Simple tests for verifier tools that don't require LLM dependencies.
Run with: python test_tools.py
"""

# Import tools directly from their files to avoid package-level imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from verifiers.tools.calculator import calculator  # type: ignore
from verifiers.tools.search import search  # type: ignore

def main():
    # Test calculator
    print("\nTesting calculator...")
    
    # Basic operations
    assert calculator("2 + 2") == "4", "Basic addition failed"
    assert calculator("3 * (17 + 4)") == "63", "Order of operations failed"
    assert calculator("100 / 5") == "20.0", "Division failed"
    print("✓ Basic operations")
    
    # Edge cases
    assert calculator("0 + 0") == "0", "Zero handling failed"
    assert calculator("1.5 * 2") == "3.0", "Decimal handling failed"
    print("✓ Edge cases")
    
    # Error cases
    assert calculator("").startswith("Error"), "Empty input handling failed"
    assert calculator("import os").startswith("Error"), "Import blocking failed"
    assert calculator("x = 1").startswith("Error"), "Variable assignment blocking failed"
    assert calculator("print('hello')").startswith("Error"), "Function call blocking failed"
    assert calculator("2 + 2; rm -rf /").startswith("Error"), "Command injection blocking failed"
    print("✓ Error cases")
    
    # Test search
    print("\nTesting search...")
    
    # Basic search with JSON format
    query = "python programming language creator"
    print(f"\nTesting query: {query}")
    print("JSON that would be passed to LLM:")
    print('{"name": "search", "args": {"query": "python programming language creator", "num_results": 5}}')
    print("\nResult:")
    result = search(query)
    print(result)
    assert isinstance(result, str), "Search result should be string"
    assert "Guido" in result or "Van Rossum" in result, "Creator search failed"
    print("✓ Basic search")
    
    # Number of results
    print("\nTesting different result counts:")
    print('{"name": "search", "args": {"query": "climate change", "num_results": 1}}')
    results1 = search("climate change", num_results=1)
    print("\nOne result:")
    print(results1)
    
    print('\n{"name": "search", "args": {"query": "climate change", "num_results": 3}}')
    results3 = search("climate change", num_results=3)
    print("\nThree results:")
    print(results3)
    
    assert len(results1.split("\n\n")) < len(results3.split("\n\n")), "num_results parameter failed"
    print("✓ Result count")
    
    # Error cases
    empty_result = search("")
    assert empty_result.startswith("Error") or empty_result == "No results found", "Empty query handling failed"
    print("✓ Error cases")
    
    # Very long query
    long_query = "what is " * 100
    result = search(long_query)
    assert isinstance(result, str), "Long query handling failed"
    print("✓ Long query")
    
    print("\nAll tests passed! 🎉")

if __name__ == "__main__":
    main() 