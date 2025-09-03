from verifiers.parsers import XMLParser

if __name__ == "__main__":
    # Example 1: Using plain field names.
    parser1 = XMLParser(['reasoning', 'answer'])
    formatted1 = parser1.format(reasoning="Step-by-step reasoning", answer="42")
    print("Formatted XML (Example 1):")
    print(formatted1)
    parsed1 = parser1.parse(formatted1)
    print("Parsed fields (Example 1):")
    print("reasoning:", parsed1.reasoning)  # Should have a value.
    print("answer:", parsed1.answer)        # Should have a value.
    
    print("\n" + "-"*50 + "\n")
    
    # Example 2: Using alternative names for the second field.
    # The schema is ['reasoning', ('code', 'answer')].
    # Formatting will use the canonical tag <code>.
    parser2 = XMLParser(['reasoning', ('code', 'answer')])
    
    # Formatting using the canonical name.
    formatted2 = parser2.format(reasoning="Detailed reasoning", code="print('Hello, world!')")
    print("Formatted XML (Example 2 - canonical tag):")
    print(formatted2)
    
    # Parsing back from the formatted string.
    parsed2 = parser2.parse(formatted2)
    print("Parsed fields (Example 2 - canonical tag):")
    print("reasoning:", parsed2.reasoning)  # from <reasoning>
    print("code:", parsed2.code)            # from <code>
    print("answer:", parsed2.answer)        # should be None because only <code> was used.
    
    print("\n" + "-"*50 + "\n")
    
    # Example 3: Parsing an XML string that uses the alternative tag for the second field.
    xml_alternative = """\
<reasoning>
This is the reasoning.
</reasoning>
<answer>
print("Alternative tag used!")
</answer>
"""
    parsed3 = parser2.parse(xml_alternative)
    print("Parsed fields (Example 3 - alternative tag):")
    print("reasoning:", parsed3.reasoning)  # from <reasoning>
    print("code:", parsed3.code)            # should be None since <code> is missing.
    print("answer:", parsed3.answer)        # from <answer>
    
    print("\n" + "-"*50 + "\n")
    
    # Example 4: Parsing an XML string where one of the fields is missing.
    xml_missing = """\
<reasoning>
Only reasoning provided.
</reasoning>
"""
    parsed4 = parser1.parse(xml_missing)
    print("Parsed fields (Example 4 - missing 'answer'):")
    print("reasoning:", parsed4.reasoning)  # has a value.
    print("answer:", parsed4.answer)        # Should be None.
