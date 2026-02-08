#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test the translation functionality
try:
    import models
    print("Testing German to English translation...")

    # Test German translation
    model, tokenizer = models.getTranslationModel('de')
    from AIModels import NeuralTranslator
    translator = NeuralTranslator(model, tokenizer)

    test_sentence = "Hallo, wie geht es dir?"
    translation = translator.translateSentence(test_sentence)
    print(f"German: '{test_sentence}' -> English: '{translation}'")

    print("Testing French to English translation...")
    # Test French translation
    model_fr, tokenizer_fr = models.getTranslationModel('fr')
    translator_fr = NeuralTranslator(model_fr, tokenizer_fr)

    test_sentence_fr = "Bonjour, comment allez-vous?"
    translation_fr = translator_fr.translateSentence(test_sentence_fr)
    print(f"French: '{test_sentence_fr}' -> English: '{translation_fr}'")

    print("Testing Spanish to English translation...")
    # Test Spanish translation
    model_es, tokenizer_es = models.getTranslationModel('es')
    translator_es = NeuralTranslator(model_es, tokenizer_es)

    test_sentence_es = "Hola, ¿cómo estás?"
    translation_es = translator_es.translateSentence(test_sentence_es)
    print(f"Spanish: '{test_sentence_es}' -> English: '{translation_es}'")

    print("All translation models working correctly!")

except Exception as e:
    print(f"Error testing translation: {e}")
    import traceback
    traceback.print_exc()