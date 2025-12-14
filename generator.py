from transformers import T5Tokenizer, T5ForConditionalGeneration
import random

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
            1: "very short (maximum 5 words)",
            2: "medium length (maximum 10 words)",
            3: "long and complex (maximum 20 words)"
        }
        
        # Default to standard if level is weird, though 0 is handled as 'random' usually
        if level_int == 0: 
             level_desc = random.choice(list(level_map.values()))
        else:
             level_desc = level_map.get(level_int, "medium length (maximum 10 words)")

        lang_map = {
            'en': 'English',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish'
        }
        
        target_lang = lang_map.get(language, 'English')

        input_text = f"Write a {level_desc} sentence in {target_lang} for a language learner. Output only the sentence."
        
        try:
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
            outputs = self.model.generate(
                input_ids, 
                max_length=60, 
                num_return_sequences=1, 
                do_sample=True, 
                temperature=0.9
            )
            sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return sentence
        except Exception as e:
            print(f"Generation failed: {e}")
            return None
