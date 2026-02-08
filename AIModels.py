import ModelInterfaces
import torch
import numpy as np


class NeuralASR(ModelInterfaces.IASRModel):
    word_locations_in_samples = None
    audio_transcript = None
    sampling_rate = 16000  # audio is resampled to 16kHz before ASR

    def __init__(self, model: torch.nn.Module, decoder) -> None:
        super().__init__()
        self.model = model
        self.decoder = decoder  # Decoder from CTC-outputs to transcripts

    def getTranscript(self) -> str:
        """Get the transcripts of the process audio"""
        assert(self.audio_transcript != None,
               'Can get audio transcripts without having processed the audio')
        return self.audio_transcript

    def getWordLocations(self) -> list:
        """Get the pair of words location from audio"""
        assert(self.word_locations_in_samples != None,
               'Can get word locations without having processed the audio')

        return self.word_locations_in_samples

    def processAudio(self, audio: torch.Tensor):
        """Process the audio"""
        audio_length_in_samples = audio.shape[1]
        with torch.inference_mode():
            nn_output = self.model(audio)

            logits = nn_output[0, :, :].detach()

            # Silero STT decoder API differs between versions.
            # Some return (text, word_timestamps), some return only text, and some return dicts.
            decode_result = None
            last_err = None
            candidates = [
                # Old API used in this repo historically
                lambda: self.decoder(logits, audio_length_in_samples, word_align=True),
                lambda: self.decoder(logits.cpu(), audio_length_in_samples, word_align=True),
                # Common newer APIs
                lambda: self.decoder(logits.cpu(), word_align=True),
                lambda: self.decoder(logits.cpu()),
                lambda: self.decoder(logits),
            ]

            for fn in candidates:
                try:
                    decode_result = fn()
                    # Some versions return empty tuple/list on failure
                    if decode_result is None:
                        continue
                    if isinstance(decode_result, (tuple, list)) and len(decode_result) == 0:
                        continue
                    break
                except Exception as e:
                    last_err = e
                    continue

            if decode_result is None:
                raise ValueError(
                    f"ASR decoder returned no result. Last error: {last_err}"
                )

            transcript = ""
            word_locs = []

            # Parse decoder outputs
            if isinstance(decode_result, tuple) and len(decode_result) == 2:
                transcript, word_locs = decode_result
            elif isinstance(decode_result, dict):
                transcript = decode_result.get("text") or decode_result.get("transcript") or ""
                word_locs = (
                    decode_result.get("word_timestamps")
                    or decode_result.get("words")
                    or decode_result.get("timestamps")
                    or []
                )
            elif isinstance(decode_result, str):
                transcript = decode_result
                word_locs = []
            else:
                # Fallback: try to coerce simple list output
                transcript = str(decode_result)
                word_locs = []

            # Normalize word locations into list[dict(start_ts,end_ts,word?)] in SAMPLES
            normalized = []
            if isinstance(word_locs, (list, tuple)):
                for w in word_locs:
                    if isinstance(w, dict):
                        # Try multiple common key variants
                        start = w.get("start_ts", w.get("start", w.get("start_time")))
                        end = w.get("end_ts", w.get("end", w.get("end_time")))
                        word = w.get("word", w.get("text", ""))
                    elif isinstance(w, (list, tuple)) and len(w) >= 2:
                        start, end = w[0], w[1]
                        word = w[2] if len(w) >= 3 else ""
                    else:
                        continue

                    if start is None or end is None:
                        continue

                    # Convert seconds -> samples when values look like seconds
                    # Heuristic: seconds are typically small floats (< audio duration in seconds),
                    # while sample indices are ints up to audio_length_in_samples.
                    try:
                        start_f = float(start)
                        end_f = float(end)
                    except Exception:
                        continue

                    if start_f <= audio_length_in_samples and end_f <= audio_length_in_samples:
                        # could already be samples (or could be seconds for very short clips);
                        # if values are fractional, treat as seconds.
                        if (start_f % 1 != 0) or (end_f % 1 != 0) or (end_f <= 60.0):
                            start_samp = int(max(0, start_f * self.sampling_rate))
                            end_samp = int(min(audio_length_in_samples - 1, end_f * self.sampling_rate))
                        else:
                            start_samp = int(max(0, start_f))
                            end_samp = int(min(audio_length_in_samples - 1, end_f))
                    else:
                        # Out of range: assume seconds
                        start_samp = int(max(0, start_f * self.sampling_rate))
                        end_samp = int(min(audio_length_in_samples - 1, end_f * self.sampling_rate))

                    normalized.append({"word": word, "start_ts": start_samp, "end_ts": end_samp})

            self.audio_transcript = transcript
            self.word_locations_in_samples = normalized


class NeuralTTS(ModelInterfaces.ITextToSpeechModel):
    def __init__(self, model: torch.nn.Module, sampling_rate: int) -> None:
        super().__init__()
        self.model = model
        self.sampling_rate = sampling_rate

    def getAudioFromSentence(self, sentence: str) -> torch.Tensor:
        # Avoid looping in some Silero models by ensuring punctuation
        if sentence and not sentence.strip()[-1] in ".!?":
            sentence = sentence.strip() + "."

        with torch.inference_mode():
            audio = self.model.apply_tts(texts=[sentence],
                                         sample_rate=self.sampling_rate)[0]

        # Trim loop if it occurs (common in French model)
        # Convert to numpy for processing
        audio_np = audio.detach().cpu().numpy()
        energy = np.abs(audio_np)
        max_energy = np.max(energy)
        if max_energy > 0:
            threshold = 0.05 * max_energy
            
            # Find first sound
            indices = np.where(energy > threshold)[0]
            if len(indices) > 0:
                first_sound = indices[0]
                gap_len = int(0.3 * self.sampling_rate)
                silent_streak = 0
                for i in range(first_sound, len(audio_np)):
                    if energy[i] < threshold:
                        silent_streak += 1
                    else:
                        silent_streak = 0
                    if silent_streak > gap_len:
                        # Trim and convert back to tensor
                        # Keep a small buffer (100ms)
                        trimmed_end = i - gap_len + int(0.1 * self.sampling_rate)
                        return torch.from_numpy(audio_np[:trimmed_end])
        
        return audio


class NeuralTranslator(ModelInterfaces.ITranslationModel):
    def __init__(self, model: torch.nn.Module, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def translateSentence(self, sentence: str) -> str:
        """Get the transcripts of the process audio"""
        tokenized_text = self.tokenizer(sentence, return_tensors='pt')
        translation = self.model.generate(**tokenized_text)
        translated_text = self.tokenizer.batch_decode(
            translation, skip_special_tokens=True)[0]

        return translated_text
