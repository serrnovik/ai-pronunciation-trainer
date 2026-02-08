
import models
import soundfile as sf
import json
import AIModels
#from flask import Response
import utilsFileIO
import os
import base64

sampling_rate = 16000
loaded_models = {}

def get_or_load_model(lang):
    if lang not in loaded_models:
        print(f"Loading TTS model for {lang}...")
        loaded_models[lang] = AIModels.NeuralTTS(models.getTTSModel(lang), sampling_rate)
    return loaded_models[lang]

# Preload default if needed, or just lazy load
# model_TTS_lambda = AIModels.NeuralTTS(models.getTTSModel('de'), sampling_rate)


def lambda_handler(event, context):

    try:
        body = json.loads(event['body'])

        text_string = body['value']
        language = body.get('language', 'de') # Default to de for backward compatibility

        linear_factor = 0.2
        
        model = get_or_load_model(language)
        
        audio = model.getAudioFromSentence(
            text_string).detach().numpy()*linear_factor
        random_file_name = utilsFileIO.generateRandomString(20)+'.wav'

        sf.write('./'+random_file_name, audio, 16000)

        with open(random_file_name, "rb") as f:
            audio_byte_array = f.read()

        # Check if file exists before removing to be safe
        if os.path.exists(random_file_name):
            os.remove(random_file_name)


        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps(
                {
                    "wavBase64": str(base64.b64encode(audio_byte_array))[2:-1],
                },
            )
        }
    except Exception as e:
        import traceback
        return {
            'statusCode': 200, # Return 200 to see the error in frontend instead of 500
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        }
