import torch
import torch.nn as nn
import pickle
from ModelInterfaces import IASRModel
from AIModels import NeuralASR 

def getASRModel(language: str,use_whisper:bool=False) -> IASRModel:
    """
    Get ASR model for the specified language.
    
    Uses Silero STT for: de, en, es (faster, lighter)
    Uses Whisper for: fr (Silero doesn't support French STT)
    
    Set use_whisper=True to force Whisper for all languages.
    """
    
    # Silero STT supported languages
    silero_supported = ['de', 'en', 'es']
    
    # Force Whisper if requested, or if language not supported by Silero
    if use_whisper or language not in silero_supported:
        from whisper_wrapper import WhisperASRModel
        print(f"Using Whisper ASR for {language}")
        return WhisperASRModel(language=language)
    
    # Use Silero STT for supported languages
    print(f"Using Silero ASR for {language}")
    model, decoder, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_stt',
        language=language,
        device=torch.device('cpu')
    )
    model.eval()
    return NeuralASR(model, decoder)


def getTTSModel(language: str) -> nn.Module:
    """
    Load TTS model for the specified language.
    
    Note: Different Silero TTS models return different tuple sizes:
    - English: Returns 5 values (model, example_text, sample_rate, speaker, device)
    - German/French/Spanish: Returns 2 values (model, example_text)
    
    We handle this by always taking the first element which is the model.
    """
    
    speaker_map = {
        'de': 'thorsten_v2',
        'en': 'lj_v2',  # v2 models have apply_tts method
        'fr': 'gilles_v2',
        'es': 'tux_v2'
    }
    
    if language not in speaker_map:
        raise ValueError(f'Language not implemented: {language}')
    
    speaker = speaker_map[language]
    
    result = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=language,
        speaker=speaker
    )
    
    # Extract model from result (handle both tuple and single return)
    if isinstance(result, tuple):
        model = result[0]  # First element is always the model
        print(f"Loaded {language} TTS (returned {len(result)} values)")
    else:
        model = result
        print(f"Loaded {language} TTS (single return)")
    
    return model


def getTranslationModel(language: str) -> nn.Module:
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM

    model_map = {
        'de': "Helsinki-NLP/opus-mt-de-en",
        'fr': "Helsinki-NLP/opus-mt-fr-en",
        'es': "Helsinki-NLP/opus-mt-es-en"
    }

    if language not in model_map:
        raise ValueError(f'Language not implemented for translation: {language}')

    model_name = model_map[language]

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Cache models to avoid Hugging face processing
        with open(f'translation_model_{language}.pickle', 'wb') as handle:
            pickle.dump(model, handle)
        with open(f'translation_tokenizer_{language}.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle)
    except Exception as e:
        print(f"Failed to load translation model for {language}: {e}")
        # Try to load from cache
        try:
            with open(f'translation_model_{language}.pickle', 'rb') as handle:
                model = pickle.load(handle)
            with open(f'translation_tokenizer_{language}.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
        except Exception as cache_e:
            raise ValueError(f'Could not load translation model for {language}: {e}, cache error: {cache_e}')

    return model, tokenizer
