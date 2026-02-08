from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import re
from num2words import num2words

class SentenceGenerator:
    def __init__(self):
        try:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
            self.enabled = True
        except Exception as e:
            print(f"Failed to load generation model: {e}")
            self.enabled = False

    def generate_sample(self, language, level_int):
        if not self.enabled:
            return None

        level_map = {
            1: ("very short", 5, 8),  # (description, target_words, max_words)
            2: ("medium length", 10, 15),
            3: ("long and complex", 18, 25)
        }
        
        # Default to standard if level is weird, though 0 is handled as 'random' usually
        if level_int == 0: 
             level_desc, target_words, max_words = random.choice(list(level_map.values()))
        else:
             level_desc, target_words, max_words = level_map.get(level_int, ("medium length", 10, 15))

        lang_map = {
            'en': 'English',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish'
        }
        
        target_lang = lang_map.get(language, 'English')

        # More explicit prompt with exact word count and natural language
        input_text = f"Generate a quotation from a classic book in {target_lang} that is {level_desc}. Use EXACTLY {target_words} words. Use common vocabulary suitable for language learners. Output ONLY the complete sentence with proper spelling and accents."
        
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
                outputs = self.model.generate(
                    input_ids, 
                    max_length=60, 
                    num_return_sequences=1, 
                    do_sample=True, 
                    temperature=0.8 - (attempt * 0.1)  # Reduce temperature on retries
                )
                sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                # Remove any extra punctuation or quotes
                sentence = sentence.strip('"').strip("'").strip()
                
                # Basic validation - reject if clearly bad
                # if not sentence or len(sentence) < 3:
                #     print(f"[Backend] Attempt {attempt + 1}/{max_attempts} - Rejected: too short")
                #     continue
                
                # Check for common generation artifacts
                bad_patterns = ['...', '****', 'example:', 'sentence:', '\n', '\t']
                if any(pattern in sentence.lower() for pattern in bad_patterns):
                    print(f"[Backend] Attempt {attempt + 1}/{max_attempts} - Rejected: contains artifacts")
                    continue
                
                # Must start with capital letter and end with punctuation
                if not sentence[0].isupper():
                    sentence = sentence.capitalize()
                if not sentence[-1] in '.!?':
                    sentence += '.'
                
                word_count = len(sentence.split())
                print(f"[Backend] Attempt {attempt + 1}/{max_attempts} - Generated ({level_int}): {sentence} ({word_count} words)")

                # Convert numbers to words
                sentence = self._convert_numbers_to_words(sentence, language)

                # Check if within acceptable limits
                if level_int > 0 and word_count <= max_words and word_count > 0:
                    print(f"[Backend] ✓ Sentence accepted ({word_count} words ≤ {max_words})")
                    return sentence
                elif level_int == 0 and word_count > 0:
                    # For random difficulty, be more lenient
                    return sentence
                    
                # If too long, try cropping intelligently
                # if word_count > max_words:
                #     cropped = self._crop_sentence(sentence, max_words)
                #     if cropped:
                #         cropped_word_count = len(cropped.split())
                #         print(f"[Backend] ✓ Cropped sentence ({word_count} → {cropped_word_count} words)")
                #         return cropped
                    
            except Exception as e:
                print(f"[Backend] Attempt {attempt + 1} failed: {e}")
                continue
        
        print(f"[Backend] ✗ All generation attempts failed. Falling back to CSV.")
        return None
    
    def _crop_sentence(self, sentence, max_words):
        """Intelligently crop a sentence to max_words by finding a natural break point."""
        words = sentence.split()
        if len(words) <= max_words:
            return sentence
            
        # Try to find a natural break point (comma, conjunction, etc.)
        truncated = words[:max_words]
        
        # Check if we can end at a punctuation mark
        for i in range(len(truncated) - 1, max(0, len(truncated) - 3), -1):
            if truncated[i].endswith((',', '.', ';', ':')):
                result = ' '.join(truncated[:i+1])
                # Ensure it ends with proper punctuation
                if not result.endswith('.'):
                    result = result.rstrip(',;:') + '.'
                return result
        
        # Otherwise just truncate and add period
        result = ' '.join(truncated)
        if not result.endswith(('.', '!', '?')):
            result += '.'
        return result

    def _convert_numbers_to_words(self, sentence, language):
        """Finds numbers in the sentence and converts them to words."""
        def replace_num(match):
            number = match.group(0)
            try:
                # Convert language code to num2words locale (e.g. 'en' -> 'en', 'fr' -> 'fr')
                # Map simple codes to full locales if needed, but basic 2-letter codes often work
                lang_code = language
                if language == 'en': lang_code = 'en'
                elif language == 'fr': lang_code = 'fr'
                elif language == 'de': lang_code = 'de'
                elif language == 'es': lang_code = 'es'
                
                return num2words(number, lang=lang_code)
            except Exception as e:
                print(f"[Backend] Failed to convert number {number}: {e}")
                return number

        # Replace all sequences of digits
        return re.sub(r'\d+', replace_num, sentence)
