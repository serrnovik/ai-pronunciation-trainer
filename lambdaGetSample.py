
import pandas as pd
import json
import RuleBasedModels
import epitran
import random
import pickle
import models
import AIModels


class TextDataset():
    def __init__(self, table):
        self.table_dataframe = table
        self.number_of_samples = len(table)

    def __getitem__(self, idx):

        line = [self.table_dataframe['sentence'].iloc[idx]]
        return line

    def __len__(self):
        return self.number_of_samples


sample_folder = "./databases/"
lambda_database = {}
lambda_ipa_converter = {}
available_languages = ['de', 'en', 'fr', 'es']

for language in available_languages:
    df = pd.read_csv(sample_folder+'data_'+language+'.csv',delimiter=';')
    lambda_database[language] = TextDataset(df)
    lambda_ipa_converter[language] = RuleBasedModels.get_phonem_converter(language)

import generator
generator_instance = generator.SentenceGenerator()

lambda_translate_new_sample = False

# Cache for translation models
translation_cache = {}


def lambda_handler(event, context):

    body = json.loads(event['body'])

    category = int(body['category'])

    language = body['language']

    # Try dynamic generation first
    generated_sentence = None
    
    if 'custom_text' in body:
        current_transcript = [body['custom_text']]
    else:
        if generator_instance.enabled:
            print("Attempting generation...")
            generated_sentence = generator_instance.generate_sample(language, category)
        
        if generated_sentence:
            current_transcript = [generated_sentence]
        else:
            # Fallback to CSV
            print("Fallback to CSV...")
            sample_in_category = False

            while(not sample_in_category):
                valid_sequence = False
                while not valid_sequence:
                    try:
                        sample_idx = random.randint(0, len(lambda_database[language]))
                        current_transcript = lambda_database[language][
                            sample_idx]
                        valid_sequence = True
                    except:
                        pass

                sentence_category = getSentenceCategory(
                    current_transcript[0])

                sample_in_category = (sentence_category ==
                                      category) or category == 0

    translated_trascript = ""

    current_ipa = lambda_ipa_converter[language].convertToPhonem(
        current_transcript[0])

    # Translate to English if not already in English
    translated_transcript = ""
    if language != 'en':
        try:
            if language not in translation_cache:
                print(f"Loading translation model for {language}")
                model, tokenizer = models.getTranslationModel(language)
                translator = AIModels.NeuralTranslator(model, tokenizer)
                translation_cache[language] = translator

            translator = translation_cache[language]
            translated_transcript = translator.translateSentence(current_transcript[0])
            print(f"Translated '{current_transcript[0]}' to '{translated_transcript}'")
        except Exception as e:
            print(f"Translation failed for {language}: {e}")
            translated_transcript = f"[Translation unavailable for {language}]"

    result = {'real_transcript': current_transcript,
              'ipa_transcript': current_ipa,
              'transcript_translation': translated_transcript}

    return json.dumps(result)


def getSentenceCategory(sentence) -> int:
    number_of_words = len(sentence.split())
    categories_word_limits = [0, 8, 20, 100000]
    for category in range(len(categories_word_limits)-1):
        if number_of_words > categories_word_limits[category] and number_of_words <= categories_word_limits[category+1]:
            return category+1
