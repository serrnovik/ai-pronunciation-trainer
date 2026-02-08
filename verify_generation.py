from generator import SentenceGenerator

def verify():
    gen = SentenceGenerator()
    
    # Random level usually picks quotes, but let's see if we can trigger number conversion
    # We can't easily force specific text into the generator model, but we can check if the method exists and works
    
    print("\n--- Testing Number Conversion Method ---")
    test_cases = [
        ("In 1984, George Orwell wrote a book.", "en"),
        ("Il y a 3 pommes.", "fr"),
        ("Das ist 1 Test.", "de"),
        ("Hay 100 cosas.", "es")
    ]
    
    for text, lang in test_cases:
        converted = gen._convert_numbers_to_words(text, lang)
        print(f"Original: {text}")
        print(f"Converted: {converted}")
        print("-" * 20)
        
    print("\n--- Testing Generation (Smoke Test) ---")
    try:
        sentence = gen.generate_sample("en", 0)
        print(f"Generated (Random): {sentence}")
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    verify()
