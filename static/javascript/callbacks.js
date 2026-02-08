// Audio context initialization
let mediaRecorder, audioChunks, audioBlob, stream, audioRecorded;
// WaveSurfer instances
var wsReference = null;
var wsRecording = null;
var wsRegionsReference = null;
var wsRegionsRecording = null;

// Track trim offsets for accurate playback
var referenceTrimOffset = 0;
var recordingTrimOffset = 0;

// Track active playback timeouts
var activePlaybackTimeout = null;

// Feature flag: Enable/disable region selection on waveforms
const ENABLE_WAVEFORM_REGIONS = true; // Set to true to enable interval selection

// Audio context
var AudioContext = window.AudioContext || window.webkitAudioContext;
var ctx = new AudioContext();
let currentAudioForPlaying;
let lettersOfWordAreCorrect = [];

// UI-related variables
const page_title = "AI Pronunciation Trainer";
const accuracy_colors = ["green", "orange", "red"];
let badScoreThreshold = 30;

function fetchReferenceAudio(text) {
  // console.log("=== fetchReferenceAudio START ===");
  // console.log("Language:", AILanguage);
  // console.log("Text:", text);
  // console.log("wsReference exists:", !!wsReference);

  try {
    if (!wsReference) {
      console.error("❌ wsReference is null - cannot load waveform!");
      return;
    }

    // Use the backend TTS instead of browser TTS for visualization
    // console.log(
    //   "Sending TTS request to:",
    //   apiMainPathTTS + "/getAudioFromText"
    // );

    fetch(apiMainPathTTS + "/getAudioFromText", {
      method: "post",
      body: JSON.stringify({ value: text, language: AILanguage }),
      headers: { "X-Api-Key": STScoreAPIKey },
    })
      .then((res) => {
        // console.log("TTS Response status:", res.status);
        return res.json();
      })
      .then((data) => {
        // console.log("TTS Response data:", data);

        // Parse the body if it's a stringified JSON
        let parsedData = data;
        if (data.body && typeof data.body === "string") {
          // console.log("Parsing stringified body...");
          try {
            parsedData = JSON.parse(data.body);
            // console.log("Parsed data:", parsedData);
          } catch (e) {
            console.error("Failed to parse body:", e);
            parsedData = data;
          }
        }

        if (parsedData.wavBase64) {
          // console.log(
          //   "✓ Received TTS audio, length:",
          //   parsedData.wavBase64.length
          // );

          // Convert base64 to blob/url
          const byteCharacters = atob(parsedData.wavBase64);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: "audio/wav" });
          const audioUrl = URL.createObjectURL(blob);

          // console.log("Created blob URL:", audioUrl);
          // console.log("Loading into wsReference...");

          wsReference.load(audioUrl);
          // console.log("✓ wsReference.load() called");

          // Show the waveform container
          document.getElementById("waveform-reference").classList.add("loaded");
          // console.log("✓ Added 'loaded' class to waveform-reference");

          // Reference TTS audio typically has minimal leading silence,
          // but we can trim a small amount if needed (usually ~0.1s)
          if (wsReference) {
            trimWaveformStart(wsReference, 0.05, false); // Trim 50ms if present
          }

          // console.log("=== fetchReferenceAudio COMPLETE ===");
        } else if (parsedData.error) {
          console.error("❌ TTS Error:", parsedData.error);
          if (parsedData.traceback) {
            console.error("Traceback:", parsedData.traceback);
          }
        } else {
          console.error("❌ No wavBase64 in response");
          console.error("Full data:", data);
          console.error("Parsed data:", parsedData);
        }
      })
      .catch((err) => {
        console.error("❌ Error in fetch:", err);
        console.error("Stack:", err.stack);
      });
  } catch (e) {
    console.error("❌ Reference audio fetch failed", e);
    console.error("Stack:", e.stack);
  }
}
let mediumScoreThreshold = 70;
let currentSample = 0;
let currentScore = 0;
let sample_difficult = 0;
let scoreMultiplier = 1;
let playAnswerSounds = true;
let isNativeSelectedForPlayback = true;
let isRecording = false;
let serverIsInitialized = false;
let serverWorking = true;
let languageFound = true;
let currentSoundRecorded = false;
let currentText, currentIpa, real_transcripts_ipa, matched_transcripts_ipa;
let wordCategories;
// Custom Text State
let customTextChunks = [];
let currentCustomIndex = 0;
let isCustomMode = false;

const toggleCustomMode = () => {
  const container = document.getElementById("custom-text-container");
  const isVisible = container.style.display === "block";
  container.style.display = isVisible ? "none" : "block";

  if (!isVisible) {
    document.getElementById("customTextInput").focus();
  }
};

const closeCustomMode = () => {
  document.getElementById("custom-text-container").style.display = "none";
  // If we haven't started practice, revert radio button?
  // For now, let's just hide the modal.
};

// Stop words for filtering (articles and prepositions)
const STOP_WORDS = {
  de: [
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "einer",
    "einen",
    "einem",
    "eines",
    "in",
    "auf",
    "an",
    "zu",
    "mit",
    "von",
    "aus",
    "nach",
    "bei",
    "für",
    "um",
    "durch",
    "über",
    "unter",
    "vor",
    "zwischen",
  ],
  en: [
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "from",
    "by",
    "of",
    "into",
    "onto",
    "about",
    "through",
    "over",
    "under",
    "before",
    "after",
    "between",
    "among",
  ],
  fr: [
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "de",
    "du",
    "d'",
    "à",
    "dans",
    "en",
    "sur",
    "pour",
    "avec",
    "par",
    "chez",
    "entre",
    "sous",
    "vers",
  ],
  es: [
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "de",
    "del",
    "a",
    "al",
    "en",
    "por",
    "para",
    "con",
    "sin",
    "sobre",
    "entre",
    "hacia",
    "hasta",
  ],
};

const fetchWikiArticle = async () => {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.classList.add("visible");

  try {
    const url = `https://${AILanguage}.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1&generator=random&grnnamespace=0&origin=*`;
    const response = await fetch(url);
    const data = await response.json();
    const pages = data.query.pages;
    const pageId = Object.keys(pages)[0];
    const extract = pages[pageId].extract;

    // Refined cleaning
    const cleanedText = extract
      .split("\n")
      // Remove section titles (== Title ==) and portal/footer lines
      .filter((line) => {
        const trimmed = line.trim();
        return (
          trimmed.length > 0 &&
          !trimmed.startsWith("==") &&
          !trimmed.startsWith("Portail") &&
          !trimmed.startsWith("Notes et références") &&
          !trimmed.includes(": ")
        ); // Often metadata like "Jeuxvideo.com : 10/20"
      })
      .join(" ")
      // Remove parenthesis content (often foreign script or dates)
      .replace(/\([^)]*\)/g, "")
      // Remove non-Latin/European characters (CJK, Cyrillic, etc.)
      // We keep: Basic Latin, Latin-1 Supplement, Latin Extended-A
      .replace(/[^\u0000-\u007F\u00C0-\u00FF\u0100-\u017F]/g, "")
      // Final cleanup of spaces and punctuation glitches
      .replace(/\s+/g, " ")
      .replace(/\s+([.,!?;:])/g, "$1") // remove space before punctuation
      .trim();

    document.getElementById("customTextInput").value = cleanedText;
  } catch (e) {
    console.error("Wikipedia fetch failed", e);
    alert("Failed to fetch Wikipedia article.");
  } finally {
    if (overlay) overlay.classList.remove("visible");
  }
};

const startCustomPractice = () => {
  const text = document.getElementById("customTextInput").value;
  if (!text.trim()) {
    alert("Please enter some text.");
    return;
  }

  const mode = document.querySelector('input[name="splitMode"]:checked').value;

  if (mode === "sentence") {
    // Split by sentences (approximated by . ? !)
    customTextChunks = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [text];
    customTextChunks = customTextChunks
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  } else if (mode === "word") {
    // Split by spaces and filter stop words
    const words = text
      .split(/\s+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    const langStopWords = STOP_WORDS[AILanguage] || [];

    customTextChunks = words.filter((word) => {
      // Remove punctuation from word for checking
      const cleanWord = word.toLowerCase().replace(/[.,!?;:()"]/g, "");
      return !langStopWords.includes(cleanWord);
    });
  } else {
    // Split by comma or pause-like punctuation
    customTextChunks = text
      .split(/[,:;]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  }

  if (customTextChunks.length === 0) {
    alert(
      "Could not split text into chunks (maybe all words were filtered out?).",
    );
    return;
  }

  isCustomMode = true;
  currentCustomIndex = 0;
  closeCustomMode();
  getNextSample();
};

let startTime, endTime;

// API related variables
let AILanguage = "de"; // Standard is German
if (
  typeof Storage !== "undefined" &&
  localStorage.getItem("selectedLanguage")
) {
  AILanguage = localStorage.getItem("selectedLanguage");
}

let STScoreAPIKey = "rll5QsTiv83nti99BW6uCmvs9BDVxSB39SVFceYb"; // Public Key. If, for some reason, you would like a private one, send-me a message and we can discuss some possibilities
const apiMainPathSample = "http://127.0.0.1:3000";
const apiMainPathSTS = "http://127.0.0.1:3000";
const apiMainPathTTS = "http://127.0.0.1:3000"; // 'https://a3hj0l2j2m.execute-api.eu-central-1.amazonaws.com/Prod';

// Variables to playback accuracy sounds
let soundsPath = "../static"; //'https://stscore-sounds-bucket.s3.eu-central-1.amazonaws.com';
let soundFileGood = null;
let soundFileOkay = null;
let soundFileBad = null;

// Speech generation
var synth = window.speechSynthesis;
let voice_idx = 0;
let voice_synth = null;

//############################ UI general control functions ###################
let isPlaying = false;

const setPlayButtonState = (isPlayingState) => {
  isPlaying = isPlayingState;
  const playBtn = document.getElementById("playSampleAudio");
  const playRecBtn = document.getElementById("playRecordedAudio");
  const icon = isPlaying ? "stop" : "play_arrow";
  const recIcon = isPlaying ? "stop" : "record_voice_over";

  // We update both icons to be safe, though usually only one plays at a time
  if (playBtn)
    playBtn.innerHTML = `<i class="material-icons icon-text">${icon}</i>`;
  if (playRecBtn)
    playRecBtn.innerHTML = `<i class="material-icons icon-text">${recIcon}</i>`;
};

const unblockUI = () => {
  setPlayButtonState(false);
  document.getElementById("recordAudio").classList.remove("disabled");
  document.getElementById("playSampleAudio").classList.remove("disabled");
  document.getElementById("buttonNext").onclick = () => getNextSample();
  document.getElementById("original_script").classList.remove("disabled");
  document.getElementById("buttonNext").style["background-color"] = "#58636d";

  if (currentSoundRecorded)
    document.getElementById("playRecordedAudio").classList.remove("disabled");
};

const blockUI = () => {
  // Don't disable play buttons if we are playing, so we can stop
  if (!isPlaying) {
    document.getElementById("playSampleAudio").classList.add("disabled");
    document.getElementById("playRecordedAudio").classList.add("disabled");
  }

  document.getElementById("recordAudio").classList.add("disabled");
  document.getElementById("buttonNext").onclick = null;
  document.getElementById("original_script").classList.add("disabled");

  document.getElementById("buttonNext").style["background-color"] = "#adadad";
};

const UIError = (e) => {
  blockUI();
  document.getElementById("buttonNext").onclick = () => getNextSample();
  document.getElementById("buttonNext").style["background-color"] = "#58636d";

  document.getElementById("recorded_ipa_script").innerHTML = "";
  document.getElementById("single_word_ipa_pair").innerHTML = "Error";
  document.getElementById("ipa_script").innerHTML = "Error";

  document.getElementById("main_title").innerHTML = "Server Error";
  let msg =
    "Server error. Either the daily quota of the server is over or there was some internal error.";
  if (e) {
    msg += " <br/><b>Details:</b> " + e.toString();
    if (e.stack)
      msg +=
        '<br/><pre style="text-align:left; font-size:0.8em; overflow:auto; max-height:200px;">' +
        e.stack +
        "</pre>";
  }
  document.getElementById("original_script").innerHTML = msg;
};

const UINotSupported = () => {
  unblockUI();

  document.getElementById("main_title").innerHTML = "Browser unsupported";
};

const UIRecordingError = () => {
  unblockUI();
  document.getElementById("main_title").innerHTML =
    "Recording error, please try again or restart page.";
  startMediaDevice();
};

//################### Application state functions #######################
function updateScore(currentPronunciationScore) {
  if (isNaN(currentPronunciationScore)) return;
  currentScore += currentPronunciationScore * scoreMultiplier;
  currentScore = Math.round(currentScore);
}

const cacheSoundFiles = async () => {
  await fetch(soundsPath + "/ASR_good.wav")
    .then((data) => data.arrayBuffer())
    .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer))
    .then((decodeAudioData) => {
      soundFileGood = decodeAudioData;
    });

  await fetch(soundsPath + "/ASR_okay.wav")
    .then((data) => data.arrayBuffer())
    .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer))
    .then((decodeAudioData) => {
      soundFileOkay = decodeAudioData;
    });

  await fetch(soundsPath + "/ASR_bad.wav")
    .then((data) => data.arrayBuffer())
    .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer))
    .then((decodeAudioData) => {
      soundFileBad = decodeAudioData;
    });
};

const getNextSample = async () => {
  blockUI();

  if (!serverIsInitialized) await initializeServer();

  if (!serverWorking) {
    UIError();
    return;
  }

  // Show overlay
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.classList.add("visible");

  if (soundFileBad == null) cacheSoundFiles();

  updateScore(parseFloat(currentScore));

  document.getElementById("main_title").innerHTML = "Processing new sample...";

  if (document.getElementById("lengthCat1").checked) {
    sample_difficult = 0;
    scoreMultiplier = 1.3;
    isCustomMode = false; // Reset if user clicks other radios
  } else if (document.getElementById("lengthCat2").checked) {
    sample_difficult = 1;
    scoreMultiplier = 1;
    isCustomMode = false;
  } else if (document.getElementById("lengthCat3").checked) {
    sample_difficult = 2;
    scoreMultiplier = 1.3;
    isCustomMode = false;
  } else if (document.getElementById("lengthCat4").checked) {
    sample_difficult = 3;
    scoreMultiplier = 1.6;
    isCustomMode = false;
  } else if (document.getElementById("modeCustom").checked) {
    // Keep isCustomMode state or set it if we are just switching back to the tab
    // But actual custom mode starts via "Start Practice"
  }

  try {
    // console.log(`[Frontend] Requesting sample. Language: ${AILanguage}, Difficulty ID: ${sample_difficult}`);

    let bodyPayload = {
      category: sample_difficult.toString(),
      language: AILanguage,
    };

    if (isCustomMode) {
      if (currentCustomIndex >= customTextChunks.length) {
        // Finished all chunks
        alert("Custom practice finished!");
        isCustomMode = false;
        currentCustomIndex = 0;
        // Optionally revert to random
        document.getElementById("lengthCat1").checked = true;
        return getNextSample();
      }
      bodyPayload.custom_text = customTextChunks[currentCustomIndex];
      // Increment for next time (or do it after success?)
      // Let's increment after success to allow retrying?
      // Actually standard behavior is "Next" button gets new sample.
      // So we increment here is fine, but we need to match the "Next" button click vs "Retry" logic if any.
      // The getNextSample is called on "Next" button.
      currentCustomIndex++;
    }

    await fetch(apiMainPathSample + "/getSample", {
      method: "post",
      body: JSON.stringify(bodyPayload),
      headers: { "X-Api-Key": STScoreAPIKey },
    })
      .then((res) => res.json())
      .then((data) => {
        // console.log("Received sample data:", data);

        let doc = document.getElementById("original_script");
        currentText = data.real_transcript;
        doc.innerHTML = currentText;

        // console.log("Current text:", currentText);

        currentIpa = data.ipa_transcript;

        let doc_ipa = document.getElementById("ipa_script");
        doc_ipa.innerHTML = "/ " + currentIpa + " /";

        document.getElementById("recorded_ipa_script").innerHTML = "";
        document.getElementById("pronunciation_accuracy").innerHTML = "";
        document.getElementById("single_word_ipa_pair").innerHTML =
          "Reference | Spoken";

        // Update Badge
        document.getElementById("score-value").innerHTML =
          currentScore.toString();
        document.getElementById("score-count").innerHTML =
          "(" + currentSample.toString() + ")";

        currentSample += 1;

        document.getElementById("main_title").innerHTML = page_title;

        // Show English translation for non-English languages
        if (AILanguage !== "en" && data.transcript_translation) {
          document.getElementById("translated_script").innerHTML =
            "<strong>English:</strong> " + data.transcript_translation;
          document.getElementById("translated_script").style.display = "block";
        } else {
          document.getElementById("translated_script").style.display = "none";
        }

        // Fetch audio for reference waveform
        let textToFetch = Array.isArray(currentText)
          ? currentText[0]
          : currentText;
        fetchReferenceAudio(textToFetch);

        // Clear previous regions (if feature enabled)
        if (ENABLE_WAVEFORM_REGIONS) {
          if (wsRegionsReference) wsRegionsReference.clearRegions();
          if (wsRegionsRecording) wsRegionsRecording.clearRegions();
        }

        // Hide waveform containers until new audio loads
        document
          .getElementById("waveform-reference")
          .classList.remove("loaded");
        document.getElementById("waveform-recorder").classList.remove("loaded");

        // Reset trim offsets for new sample
        referenceTrimOffset = 0;
        recordingTrimOffset = 0;
        // console.log("Reset trim offsets for new sample");

        currentSoundRecorded = false;
        unblockUI();
        const overlay = document.getElementById("loading-overlay");
        if (overlay) overlay.classList.remove("visible");
        document.getElementById("playRecordedAudio").classList.add("disabled");
      });
  } catch (e) {
    console.error(e);
    const overlay = document.getElementById("loading-overlay");
    if (overlay) overlay.classList.remove("visible");
    UIError(e);
  }
};

const generateWordModal = (word_idx) => {
  document.getElementById("single_word_ipa_pair").innerHTML =
    wrapWordForPlayingLink(
      real_transcripts_ipa[word_idx],
      word_idx,
      false,
      "black",
    ) +
    " | " +
    wrapWordForPlayingLink(
      matched_transcripts_ipa[word_idx],
      word_idx,
      true,
      accuracy_colors[parseInt(wordCategories[word_idx])],
    );
};

const recordSample = async () => {
  document.getElementById("main_title").innerHTML =
    "Recording... click again when done speaking";
  document.getElementById("recordIcon").innerHTML = "pause_presentation";
  blockUI();
  document.getElementById("recordAudio").classList.remove("disabled");
  audioChunks = [];
  isRecording = true;
  mediaRecorder.start();
};

const updateRecordingState = () => {
  if (isRecording) {
    stopRecording();
  } else {
    recordSample();
    // Don't start mic visualization - we'll show waveform after recording completes
  }
};

const changeLanguage = (language, generateNewSample = false) => {
  voices = synth.getVoices();
  AILanguage = language;
  languageFound = false;
  let languageIdentifier, languageName;
  switch (language) {
    case "de":
      document.getElementById("languageBox").innerHTML =
        '<span class="flag-icon flag-icon-de shadow-sm" style="margin-right: 8px;"></span>German';
      languageIdentifier = "de";
      languageName = "Anna";
      break;

    case "en":
      document.getElementById("languageBox").innerHTML =
        '<span class="flag-icon flag-icon-gb shadow-sm" style="margin-right: 8px;"></span>English';
      languageIdentifier = "en";
      languageName = "Daniel";
      break;

    case "fr":
      document.getElementById("languageBox").innerHTML =
        '<span class="flag-icon flag-icon-fr shadow-sm" style="margin-right: 8px;"></span>French';
      languageIdentifier = "fr";
      languageName = "Thomas";
      break;

    case "es":
      document.getElementById("languageBox").innerHTML =
        '<span class="flag-icon flag-icon-es shadow-sm" style="margin-right: 8px;"></span>Spanish';
      languageIdentifier = "es";
      languageName = "Monica";
      break;
  }

  // Save to localStorage
  if (typeof Storage !== "undefined") {
    localStorage.setItem("selectedLanguage", language);
  }

  for (idx = 0; idx < voices.length; idx++) {
    if (
      voices[idx].lang.slice(0, 2) == languageIdentifier &&
      voices[idx].name == languageName
    ) {
      voice_synth = voices[idx];
      languageFound = true;
      break;
    }
  }
  // If specific voice not found, search anything with the same language
  if (!languageFound) {
    for (idx = 0; idx < voices.length; idx++) {
      if (voices[idx].lang.slice(0, 2) == languageIdentifier) {
        voice_synth = voices[idx];
        languageFound = true;
        break;
      }
    }
  }
  if (generateNewSample) getNextSample();
};

//################### Speech-To-Score function ########################
const mediaStreamConstraints = {
  audio: {
    channelCount: 1,
    sampleRate: 48000,
  },
};

const startMediaDevice = () => {
  navigator.mediaDevices
    .getUserMedia(mediaStreamConstraints)
    .then((_stream) => {
      stream = _stream;
      mediaRecorder = new MediaRecorder(stream);

      // Visualization moved to updateRecordingState to only show when recording

      let currentSamples = 0;
      mediaRecorder.ondataavailable = (event) => {
        currentSamples += event.data.length;
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        document.getElementById("recordIcon").innerHTML = "mic";
        blockUI();

        audioBlob = new Blob(audioChunks, { type: "audio/ogg;" });

        let audioUrl = URL.createObjectURL(audioBlob);
        audioRecorded = new Audio(audioUrl);

        // Visualise Recording - will trim after we get start_time from backend
        if (wsRecording) {
          wsRecording.load(audioUrl);
          // Show the waveform container
          document.getElementById("waveform-recorder").classList.add("loaded");
        }

        let audioBase64 = await convertBlobToBase64(audioBlob);

        let minimumAllowedLength = 6;
        if (audioBase64.length < minimumAllowedLength) {
          setTimeout(UIRecordingError, 50); // Make sure this function finished after get called again
          return;
        }

        try {
          // Get currentText from "original_script" div, in case user has change it
          let text = document.getElementById("original_script").innerHTML;
          // Remove html tags
          text = text.replace(/<[^>]*>?/gm, "");
          //Remove spaces on the beginning and end
          text = text.trim();
          // Remove double spaces
          text = text.replace(/\s\s+/g, " ");
          currentText = [text];

          await fetch(apiMainPathSTS + "/GetAccuracyFromRecordedAudio", {
            method: "post",
            body: JSON.stringify({
              title: currentText[0],
              base64Audio: audioBase64,
              language: AILanguage,
            }),
            headers: { "X-Api-Key": STScoreAPIKey },
          })
            .then((res) => res.json())
            .then((data) => {
              if (!data || data.error) {
                console.error(
                  "❌ Scoring Error:",
                  data?.error,
                  data?.traceback,
                );
                UIError();
                return;
              }

              // Some warm-up responses may return { ok: true }
              if (
                data.ok === true &&
                data.pronunciation_accuracy === undefined
              ) {
                return;
              }

              if (playAnswerSounds)
                playSoundForAnswerAccuracy(
                  parseFloat(data.pronunciation_accuracy),
                );

              document.getElementById("recorded_ipa_script").innerHTML =
                "/ " + data.ipa_transcript + " /";
              document.getElementById("recordAudio").classList.add("disabled");
              document.getElementById("main_title").innerHTML = page_title;
              document.getElementById("pronunciation_accuracy").innerHTML =
                data.pronunciation_accuracy + "%";
              document.getElementById("ipa_script").innerHTML =
                data.real_transcripts_ipa;

              lettersOfWordAreCorrect =
                data.is_letter_correct_all_words.split(" ");

              startTime = data.start_time;
              endTime = data.end_time;

              // Trim silence from beginning of recording waveform
              if (wsRecording && startTime) {
                const firstWordStart = parseFloat(startTime.split(" ")[0]);
                // Only trim if there's significant silence (> 0.3 seconds)
                if (firstWordStart > 0.3) {
                  trimWaveformStart(wsRecording, firstWordStart - 0.1, true); // Keep 0.1s buffer
                }
              }

              real_transcripts_ipa = data.real_transcripts_ipa.split(" ");
              matched_transcripts_ipa = data.matched_transcripts_ipa.split(" ");
              wordCategories = data.pair_accuracy_category.split(" ");
              let currentTextWords = currentText[0].split(" ");

              coloredWords = "";
              for (
                let word_idx = 0;
                word_idx < currentTextWords.length;
                word_idx++
              ) {
                wordTemp = "";
                for (
                  let letter_idx = 0;
                  letter_idx < currentTextWords[word_idx].length;
                  letter_idx++
                ) {
                  letter_is_correct =
                    lettersOfWordAreCorrect[word_idx][letter_idx] == "1";
                  if (letter_is_correct) color_letter = "green";
                  else color_letter = "red";

                  wordTemp +=
                    "<font color=" +
                    color_letter +
                    ">" +
                    currentTextWords[word_idx][letter_idx] +
                    "</font>";
                }
                currentTextWords[word_idx];
                coloredWords +=
                  " " + wrapWordForIndividualPlayback(wordTemp, word_idx);
              }

              document.getElementById("original_script").innerHTML =
                coloredWords;

              currentSoundRecorded = true;
              unblockUI();
              document
                .getElementById("playRecordedAudio")
                .classList.remove("disabled");
            });
        } catch {
          UIError();
        }
      };
    });
};
startMediaDevice();

// ################### Audio playback ##################
const playSoundForAnswerAccuracy = async (accuracy) => {
  currentAudioForPlaying = soundFileGood;
  if (accuracy < mediumScoreThreshold) {
    if (accuracy < badScoreThreshold) {
      currentAudioForPlaying = soundFileBad;
    } else {
      currentAudioForPlaying = soundFileOkay;
    }
  }
  playback();
};

const playAudio = async () => {
  // console.log("playAudio button clicked");

  // Clear any pending playback timeout
  if (activePlaybackTimeout) {
    clearTimeout(activePlaybackTimeout);
    activePlaybackTimeout = null;
  }

  // Stop Recording Playback if active
  if (wsRecording && wsRecording.isPlaying()) {
    wsRecording.pause();
  }
  if (isPlaying) {
    // This flag is often used for the audioRecorded element playback as well
    if (audioRecorded) {
      audioRecorded.pause();
      audioRecorded.currentTime = 0;
    }
    unblockUI(); // This resets icons including the recording one
  }

  // Check for WaveSurfer regions first (only if feature enabled)
  if (ENABLE_WAVEFORM_REGIONS && wsReference && wsRegionsReference) {
    const regions = wsRegionsReference.getRegions();
    // console.log("Reference regions found:", regions.length);

    if (regions.length > 0) {
      const region = regions[0];
      // console.log(
      //   "Playing reference region from button:",
      //   region.start,
      //   "to",
      //   region.end
      // );

      // Toggle play/pause for region
      if (wsReference.isPlaying()) {
        // console.log("Pausing reference playback");
        wsReference.pause();
      } else {
        // Play the selected region
        try {
          wsReference.setTime(region.start);
          wsReference.play();

          // Stop at region end
          const duration = (region.end - region.start) * 1000; // Convert to ms
          // console.log("Will stop after", duration, "ms");

          activePlaybackTimeout = setTimeout(() => {
            if (wsReference && wsReference.isPlaying()) {
              // console.log("Auto-stopping reference region playback");
              wsReference.pause();
            }
            activePlaybackTimeout = null;
          }, duration);
        } catch (err) {
          console.error("Error playing reference region:", err);
        }
      }
      return;
    }
  }

  if (wsReference) {
    // Toggle play/pause
    // console.log("Playing/pausing full reference waveform");
    if (wsReference.isPlaying()) {
      wsReference.pause();
    } else {
      wsReference.play();
    }
    return;
  }

  if (isPlaying) {
    synth.cancel();
    unblockUI();
    return;
  }

  setPlayButtonState(true);
  document.getElementById("main_title").innerHTML = "Generating sound...";

  const selection = getSelectionIndices();
  let textToPlay = currentText[0];

  if (selection) {
    // Reconstruct the text from selected words
    const words = currentText[0].split(" ");
    // selection.end is inclusive index
    textToPlay = words.slice(selection.start, selection.end + 1).join(" ");
  }

  playWithMozillaApi(textToPlay);
  document.getElementById("main_title").innerHTML = "Current Sound was played";
};

function playback() {
  const playSound = ctx.createBufferSource();
  playSound.buffer = currentAudioForPlaying;
  playSound.connect(ctx.destination);
  playSound.start(ctx.currentTime);
}

const playRecording = async (start = null, end = null) => {
  // console.log("playRecording button clicked, start:", start, "end:", end);

  // Clear any pending playback timeout
  if (activePlaybackTimeout) {
    clearTimeout(activePlaybackTimeout);
    activePlaybackTimeout = null;
  }

  // Stop Reference Playback if active
  if (wsReference && wsReference.isPlaying()) {
    wsReference.pause();
  }
  synth.cancel(); // Stop Mozilla API speech
  // Reset Play button state if it was playing reference
  const playBtn = document.getElementById("playSampleAudio");
  if (playBtn)
    playBtn.innerHTML = `<i class="material-icons icon-text">play_arrow</i>`;

  // Check for WaveSurfer regions first (only if feature enabled)
  if (
    ENABLE_WAVEFORM_REGIONS &&
    wsRecording &&
    wsRegionsRecording &&
    start === null &&
    end === null
  ) {
    const regions = wsRegionsRecording.getRegions();
    // console.log("Recording regions found:", regions.length);
    // console.log("wsRecording state:", {
    //   exists: !!wsRecording,
    //   isPlaying: wsRecording ? wsRecording.isPlaying() : false,
    //   duration: wsRecording ? wsRecording.getDuration() : 0,
    //   currentTime: wsRecording ? wsRecording.getCurrentTime() : 0,
    // });

    if (regions.length > 0) {
      const region = regions[0];
      // console.log(
      //   "Playing recording region from button:",
      //   region.start,
      //   "to",
      //   region.end
      // );

      // Toggle play/pause for region
      if (wsRecording.isPlaying()) {
        // console.log("Pausing recording playback");
        wsRecording.pause();
      } else {
        // Play the selected region
        try {
          // console.log("Setting time to:", region.start);
          wsRecording.setTime(region.start);

          // console.log("Starting playback...");
          wsRecording.play();

          // Stop at region end
          const duration = (region.end - region.start) * 1000; // Convert to ms
          // console.log("Will stop after", duration, "ms");

          activePlaybackTimeout = setTimeout(() => {
            if (wsRecording && wsRecording.isPlaying()) {
              // console.log("Auto-stopping recording region playback");
              wsRecording.pause();
            }
            activePlaybackTimeout = null;
          }, duration);
        } catch (err) {
          console.error("Error playing recording region:", err);
        }
      }
      return;
    }
  }

  // If using WaveSurfer without regions, toggle play/pause
  if (wsRecording && start === null && end === null) {
    // console.log("Playing/pausing full recording waveform");
    if (wsRecording.isPlaying()) {
      wsRecording.pause();
    } else {
      wsRecording.play();
    }
    return;
  }

  if (isPlaying) {
    audioRecorded.pause();
    audioRecorded.currentTime = 0;
    unblockUI();
    return;
  }

  setPlayButtonState(true);
  blockUI();

  try {
    if (start == null || end == null) {
      const selection = getSelectionIndices();

      if (selection && startTime && endTime) {
        // Play selected range
        const sTime = parseFloat(startTime.split(" ")[selection.start]);
        const eTime = parseFloat(endTime.split(" ")[selection.end]);

        audioRecorded.currentTime = sTime;

        const durationMs = Math.round((eTime - sTime) * 1000);

        setTimeout(function () {
          if (isPlaying) {
            unblockUI();
            audioRecorded.pause();
            audioRecorded.currentTime = 0;
            document.getElementById("main_title").innerHTML =
              "Recorded Sound was played";
          }
        }, durationMs);

        await audioRecorded.play();
      } else {
        // Play full recording (or from start if specified)
        if (start != null) {
          audioRecorded.currentTime = start;
        } else {
          endTimeInMs = Math.round(audioRecorded.duration * 1000);
        }

        audioRecorded.addEventListener(
          "ended",
          function () {
            audioRecorded.currentTime = 0;
            unblockUI();
            document.getElementById("main_title").innerHTML =
              "Recorded Sound was played";
          },
          { once: true },
        );
        await audioRecorded.play();
      }
    } else {
      // Logic for specific segment playback (not used by new "play from word" flow but kept for compatibility)
      audioRecorded.currentTime = start;
      audioRecorded.play();
      durationInSeconds = end - start;
      endTimeInMs = Math.round(durationInSeconds * 1000);
      setTimeout(function () {
        if (isPlaying) {
          // Only stop if still playing this specific segment
          unblockUI();
          audioRecorded.pause();
          audioRecorded.currentTime = 0;
          document.getElementById("main_title").innerHTML =
            "Recorded Sound was played";
        }
      }, endTimeInMs);
    }
  } catch {
    UINotSupported();
    unblockUI();
  }
};

const getSelectionIndices = () => {
  const selection = window.getSelection();
  if (selection.rangeCount === 0 || selection.toString().trim() === "")
    return null;

  const range = selection.getRangeAt(0);
  const div = document.getElementById("original_script");

  // Find all 'a' tags with data-word-index within the range
  const allLinks = Array.from(div.getElementsByTagName("a"));
  const selectedLinks = allLinks.filter((link) =>
    selection.containsNode(link, true),
  );

  if (selectedLinks.length === 0) return null;

  const indices = selectedLinks
    .map((link) => parseInt(link.getAttribute("data-word-index")))
    .sort((a, b) => a - b);
  return {
    start: indices[0],
    end: indices[indices.length - 1],
  };
};

const playNativeAndRecordedWord = async (word_idx) => {
  // If there is a selection, we ignore the clicked word logic and play the selection instead
  // Unless the click was part of making the selection

  // Actually, usually click clears selection unless it's a drag.
  // The requirement is "clicking on the word -> play from word", "selecting text -> play text"
  // So this function (onClick) should remain strictly for "play from word"

  if (isNativeSelectedForPlayback) playCurrentWord(word_idx);
  else playRecordedWord(word_idx);

  isNativeSelectedForPlayback = !isNativeSelectedForPlayback;
};

const stopRecording = () => {
  isRecording = false;
  mediaRecorder.stop();

  document.getElementById("main_title").innerHTML = "Processing audio...";
};

const playCurrentWord = async (word_idx) => {
  document.getElementById("main_title").innerHTML = "Generating word...";
  playWithMozillaApi(currentText[0].split(" ")[word_idx]);
  document.getElementById("main_title").innerHTML = "Word was played";
};

// TODO: Check if fallback is correct
const playWithMozillaApi = (text) => {
  if (languageFound) {
    blockUI();
    if (voice_synth == null) changeLanguage(AILanguage);

    var utterThis = new SpeechSynthesisUtterance(text);
    utterThis.voice = voice_synth;
    utterThis.rate = 0.7;
    utterThis.onend = function (event) {
      unblockUI();
    };
    synth.speak(utterThis);
  } else {
    UINotSupported();
  }
};

const playRecordedWord = (word_idx) => {
  wordStartTime = parseFloat(startTime.split(" ")[word_idx]);
  wordEndTime = parseFloat(endTime.split(" ")[word_idx]);

  // Play from start to end (pass null as end time)
  playRecording(wordStartTime, null);
};

// ############# Utils #####################
const convertBlobToBase64 = async (blob) => {
  return await blobToBase64(blob);
};

const blobToBase64 = (blob) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });

const wrapWordForPlayingLink = (
  word,
  word_idx,
  isFromRecording,
  word_accuracy_color,
) => {
  if (isFromRecording)
    return (
      '<a data-word-index="' +
      word_idx +
      '" style = " white-space:nowrap; color:' +
      word_accuracy_color +
      '; " href="javascript:playRecordedWord(' +
      word_idx.toString() +
      ')"  >' +
      word +
      "</a> "
    );
  else
    return (
      '<a data-word-index="' +
      word_idx +
      '" style = " white-space:nowrap; color:' +
      word_accuracy_color +
      '; " href="javascript:playCurrentWord(' +
      word_idx.toString() +
      ')" >' +
      word +
      "</a> "
    );
};

const wrapWordForIndividualPlayback = (word, word_idx) => {
  return (
    '<a data-word-index="' +
    word_idx +
    '" onmouseover="generateWordModal(' +
    word_idx.toString() +
    ')" style = " white-space:nowrap; " href="javascript:playNativeAndRecordedWord(' +
    word_idx.toString() +
    ')"  >' +
    word +
    "</a> "
  );
};

const trimWaveformStart = (wavesurfer, trimSeconds, isRecording = false) => {
  try {
    if (!wavesurfer) {
      console.warn("Cannot trim waveform: wavesurfer not initialized");
      return;
    }

    const applyTrim = () => {
      const duration = wavesurfer.getDuration();
      if (trimSeconds >= duration || trimSeconds < 0) {
        console.warn("Invalid trim time");
        return;
      }

      // Store the trim offset for accurate region playback
      if (isRecording) {
        recordingTrimOffset = trimSeconds;
        // console.log(
        //   `✓ Recording trim offset stored: ${trimSeconds.toFixed(2)}s`
        // );
      } else {
        referenceTrimOffset = trimSeconds;
        // console.log(
        //   `✓ Reference trim offset stored: ${trimSeconds.toFixed(2)}s`
        // );
      }

      // Shift the view to start at the speech beginning
      const container = wavesurfer.getWrapper();
      if (container) {
        // Find the actual waveform canvases container
        const canvasesDiv =
          container.querySelector('div[part="canvases"]') ||
          container.querySelector("div").querySelector("div");

        if (canvasesDiv) {
          const containerWidth = container.clientWidth;
          // Calculate how many pixels to shift based on trim time ratio
          const trimRatio = trimSeconds / duration;
          const shiftPixels = trimRatio * containerWidth;

          // Use CSS transform for smooth, performant shifting
          canvasesDiv.style.transform = `translateX(-${shiftPixels}px)`;
          canvasesDiv.style.transition = "transform 0.3s ease";
        }

        // Remove any existing overlays
        const existingOverlay = container.querySelector(".silence-overlay");
        if (existingOverlay) {
          existingOverlay.remove();
        }

        // Add a thin marker at the left edge to show where silence was trimmed
        if (trimSeconds > 0.05) {
          const overlay = document.createElement("div");
          overlay.className = "silence-overlay";
          overlay.style.position = "absolute";
          overlay.style.left = "0";
          overlay.style.width = "3px";
          overlay.style.height = "100%";
          overlay.style.backgroundColor = "rgba(150, 150, 150, 0.6)";
          overlay.style.pointerEvents = "none";
          overlay.style.zIndex = "2";
          overlay.style.top = "0";
          overlay.title = `${trimSeconds.toFixed(2)}s silence trimmed`;
          container.style.position = "relative";
          container.appendChild(overlay);
        }
      }

      // console.log(
      //   `✓ Shifted waveform by ${trimSeconds.toFixed(2)}s to align speech start`
      // );
    };

    // Check if already ready, otherwise wait for ready event
    if (wavesurfer.getDuration() > 0) {
      applyTrim();
    } else {
      wavesurfer.once("ready", applyTrim);
    }
  } catch (e) {
    console.error("Error trimming waveform:", e);
  }
};

// ########## Function to initialize server ###############
// This is to try to avoid aws lambda cold start
try {
  fetch(apiMainPathSTS + "/GetAccuracyFromRecordedAudio", {
    method: "post",
    body: JSON.stringify({ title: "", base64Audio: "", language: AILanguage }),
    headers: { "X-Api-Key": STScoreAPIKey },
  });
} catch {}

const initializeServer = async () => {
  valid_response = false;
  document.getElementById("main_title").innerHTML =
    "Initializing server, this may take up to 2 minutes...";
  let number_of_tries = 0;
  let maximum_number_of_tries = 4;

  while (!valid_response) {
    if (number_of_tries > maximum_number_of_tries) {
      serverWorking = false;
      break;
    }

    try {
      await fetch(apiMainPathSTS + "/GetAccuracyFromRecordedAudio", {
        method: "post",
        body: JSON.stringify({
          title: "",
          base64Audio: "",
          language: AILanguage,
        }),
        headers: { "X-Api-Key": STScoreAPIKey },
      }).then((valid_response = true));
      serverIsInitialized = true;
    } catch {
      number_of_tries += 1;
    }
  }
};

// Initialize Application when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  // Ensure voices are loaded before setting initial language
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.addEventListener("voiceschanged", () =>
      changeLanguage(AILanguage, false),
    );
  }
  // Fallback/Initial load
  setTimeout(() => changeLanguage(AILanguage, false), 500);

  // Initialize WaveSurfers
  // Regions Plugin for Reference (only if enabled)
  if (ENABLE_WAVEFORM_REGIONS) {
    wsRegionsReference = WaveSurfer.Regions.create();
  }

  const wsReferencePlugins = ENABLE_WAVEFORM_REGIONS
    ? [wsRegionsReference]
    : [];

  wsReference = WaveSurfer.create({
    container: "#waveform-reference",
    waveColor: "#A8DBA8",
    progressColor: "#3B8686",
    cursorColor: "#3B8686",
    barWidth: 2,
    height: 60,
    responsive: true,
    normalize: true,
    plugins: wsReferencePlugins,
  });

  // console.log("✓ wsReference initialized successfully");

  // Debug event listeners
  wsReference.on("play", () => {
    /* console.log("wsReference: play event fired") */
  });
  wsReference.on("pause", () => {
    /* console.log("wsReference: pause event fired") */
  });
  wsReference.on("ready", () => {
    /* console.log(
      "wsReference: ready event fired, duration:",
      wsReference.getDuration()
    );
    // console.log(
    //   "wsReference container:",
    //   document.getElementById("waveform-reference")
    // ); */
  });
  wsReference.on("load", () => {
    /* console.log("wsReference: load event fired") */
  });
  wsReference.on("decode", () => {
    /* console.log("wsReference: decode event fired") */
  });
  wsReference.on("error", (err) => console.error("❌ wsReference error:", err));

  // Region-based features (DISABLED for now - set ENABLE_WAVEFORM_REGIONS to true to enable)
  if (ENABLE_WAVEFORM_REGIONS && wsRegionsReference) {
    // Enable drag selection for Reference
    wsRegionsReference.enableDragSelection({
      color: "rgba(0, 255, 0, 0.2)",
    });

    // Ensure only one region can exist at a time for Reference
    wsRegionsReference.on("region-created", (newRegion) => {
      // console.log(
      //   "Reference region created:",
      //   newRegion.start,
      //   "to",
      //   newRegion.end
      // );
      const regions = wsRegionsReference.getRegions();
      regions.forEach((region) => {
        if (region !== newRegion) {
          // console.log("Removing old reference region");
          region.remove();
        }
      });
    });

    wsRegionsReference.on("region-clicked", (region, e) => {
      e.stopPropagation(); // Prevent clicking on the waveform from seeking to that point
      // console.log("Reference region clicked:", region.start, "to", region.end);
      // console.log("wsReference state:", {
      //   exists: !!wsReference,
      //   isPlaying: wsReference ? wsReference.isPlaying() : false,
      //   duration: wsReference ? wsReference.getDuration() : 0,
      // });

      // Clear any pending playback timeout
      if (activePlaybackTimeout) {
        clearTimeout(activePlaybackTimeout);
        activePlaybackTimeout = null;
        // console.log("Cleared previous playback timeout");
      }

      // Stop if already playing
      if (wsReference && wsReference.isPlaying()) {
        wsReference.pause();
        // console.log("Stopped current playback");
      }

      // Play the region manually
      if (wsReference) {
        try {
          // console.log("Setting time to:", region.start);
          wsReference.setTime(region.start);

          // console.log("Starting playback...");
          wsReference.play();

          const duration = (region.end - region.start) * 1000;
          // console.log("Will play for", duration, "ms");

          activePlaybackTimeout = setTimeout(() => {
            if (wsReference && wsReference.isPlaying()) {
              wsReference.pause();
              // console.log("Reference region playback auto-stopped");
            }
            activePlaybackTimeout = null;
          }, duration);
        } catch (err) {
          console.error("Error playing region:", err);
        }
      } else {
        console.error("wsReference is not available!");
      }
    });

    // Double-click on region to remove it
    wsRegionsReference.on("region-double-clicked", (region, e) => {
      e.stopPropagation();
      // console.log("Double-click: removing reference region");
      region.remove();
    });

    // Stop playback when region ends
    wsRegionsReference.on("region-out", (region) => {
      if (wsReference && wsReference.isPlaying()) {
        wsReference.pause();
      }
    });

    // Click on waveform background: Play from cursor or region
    // Using mouseup to capture "release" of mouse button
    document
      .getElementById("waveform-reference")
      .addEventListener("mouseup", (e) => {
        // Timeout to ensure WaveSurfer has updated cursor/regions
        setTimeout(() => {
          // Stop conflicting audio
          if (activePlaybackTimeout) {
            clearTimeout(activePlaybackTimeout);
            activePlaybackTimeout = null;
          }
          if (wsRecording && wsRecording.isPlaying()) {
            wsRecording.pause();
          }
          if (audioRecorded) {
            audioRecorded.pause();
            audioRecorded.currentTime = 0;
          }
          synth.cancel();
          unblockUI(); // Reset icons

          // Play Logic
          if (wsRegionsReference) {
            const regions = wsRegionsReference.getRegions();
            // If a region was just created/selected, play it
            // We assume if a region exists, we want to play it.
            // Note: Standard click might clear region first if we kept that listener.
            // We removed the 'click' clears regions listener to allow this.
            if (regions.length > 0) {
              // Play region (re-using existing logic logic via manual play)
              const region = regions[0];
              wsReference.setTime(region.start);
              wsReference.play();
              const duration = (region.end - region.start) * 1000;
              activePlaybackTimeout = setTimeout(() => {
                if (wsReference && wsReference.isPlaying()) {
                  wsReference.pause();
                }
                activePlaybackTimeout = null;
              }, duration);
              return;
            }
          }
          // Fallback: Play from cursor
          if (wsReference) {
            wsReference.play();
          }
        }, 50);
      });
  }

  // Don't use Record Plugin - we only want to visualize after recording is complete
  // The jumping waveform during recording is distracting

  // Regions Plugin for Recording (only if enabled)
  if (ENABLE_WAVEFORM_REGIONS) {
    wsRegionsRecording = WaveSurfer.Regions.create();
  }

  const wsRecordingPlugins = ENABLE_WAVEFORM_REGIONS
    ? [wsRegionsRecording]
    : [];

  wsRecording = WaveSurfer.create({
    container: "#waveform-recorder",
    waveColor: "#ffb3b3",
    progressColor: "#ff4d4d",
    cursorColor: "#ff4d4d",
    barWidth: 2,
    height: 60,
    responsive: true,
    normalize: true,
    plugins: wsRecordingPlugins,
  });

  // console.log("✓ wsRecording initialized successfully");

  // Debug event listeners
  wsRecording.on("play", () => {
    /* console.log("wsRecording: play event fired") */
  });
  wsRecording.on("pause", () => {
    /* console.log("wsRecording: pause event fired") */
  });
  wsRecording.on("ready", () => {
    /* console.log(
      "wsRecording: ready event fired, duration:",
      wsRecording.getDuration()
    );
    // console.log(
    //   "wsRecording container:",
    //   document.getElementById("waveform-recorder")
    // ); */
  });
  wsRecording.on("load", () => {
    /* console.log("wsRecording: load event fired") */
  });
  wsRecording.on("decode", () => {
    /* console.log("wsRecording: decode event fired") */
  });
  wsRecording.on("error", (err) => console.error("❌ wsRecording error:", err));

  // Region-based features (DISABLED for now - set ENABLE_WAVEFORM_REGIONS to true to enable)
  if (ENABLE_WAVEFORM_REGIONS && wsRegionsRecording) {
    // Enable drag selection for Recording
    wsRegionsRecording.enableDragSelection({
      color: "rgba(255, 0, 0, 0.2)",
    });

    // Ensure only one region can exist at a time for Recording
    wsRegionsRecording.on("region-created", (newRegion) => {
      // console.log(
      //   "Recording region created:",
      //   newRegion.start,
      //   "to",
      //   newRegion.end
      // );
      const regions = wsRegionsRecording.getRegions();
      regions.forEach((region) => {
        if (region !== newRegion) {
          // console.log("Removing old recording region");
          region.remove();
        }
      });
    });

    wsRegionsRecording.on("region-clicked", (region, e) => {
      e.stopPropagation();
      // console.log("Recording region clicked:", region.start, "to", region.end);
      // console.log("wsRecording state:", {
      //   exists: !!wsRecording,
      //   isPlaying: wsRecording ? wsRecording.isPlaying() : false,
      //   duration: wsRecording ? wsRecording.getDuration() : 0,
      // });

      // Clear any pending playback timeout
      if (activePlaybackTimeout) {
        clearTimeout(activePlaybackTimeout);
        activePlaybackTimeout = null;
        // console.log("Cleared previous playback timeout");
      }

      // Stop if already playing
      if (wsRecording && wsRecording.isPlaying()) {
        wsRecording.pause();
        // console.log("Stopped current playback");
      }

      // Play the region manually
      if (wsRecording) {
        try {
          // console.log("Setting time to:", region.start);
          wsRecording.setTime(region.start);

          // console.log("Starting playback...");
          wsRecording.play();

          const duration = (region.end - region.start) * 1000;
          // console.log("Will play for", duration, "ms");

          activePlaybackTimeout = setTimeout(() => {
            if (wsRecording && wsRecording.isPlaying()) {
              wsRecording.pause();
              // console.log("Recording region playback auto-stopped");
            }
            activePlaybackTimeout = null;
          }, duration);
        } catch (err) {
          console.error("Error playing region:", err);
        }
      } else {
        console.error("wsRecording is not available!");
      }
    });

    // Double-click on region to remove it
    wsRegionsRecording.on("region-double-clicked", (region, e) => {
      e.stopPropagation();
      // console.log("Double-click: removing recording region");
      region.remove();
    });

    // Stop playback when region ends
    wsRegionsRecording.on("region-out", (region) => {
      if (wsRecording && wsRecording.isPlaying()) {
        wsRecording.pause();
      }
    });

    // Click on waveform background to clear all regions (recording)
    // Using mouseup for click-to-play
    document
      .getElementById("waveform-recorder")
      .addEventListener("mouseup", (e) => {
        setTimeout(() => {
          // Stop conflicting audio
          if (activePlaybackTimeout) {
            clearTimeout(activePlaybackTimeout);
            activePlaybackTimeout = null;
          }
          if (wsReference && wsReference.isPlaying()) {
            wsReference.pause();
          }
          synth.cancel();
          // Reset Ref Play button
          const playBtn = document.getElementById("playSampleAudio");
          if (playBtn)
            playBtn.innerHTML = `<i class="material-icons icon-text">play_arrow</i>`;

          // Play Logic
          if (wsRegionsRecording) {
            const regions = wsRegionsRecording.getRegions();
            if (regions.length > 0) {
              const region = regions[0];
              wsRecording.setTime(region.start);
              wsRecording.play();
              const duration = (region.end - region.start) * 1000;
              activePlaybackTimeout = setTimeout(() => {
                if (wsRecording && wsRecording.isPlaying()) {
                  wsRecording.pause();
                }
                activePlaybackTimeout = null;
              }, duration);
              return;
            }
          }
          // Fallback
          if (wsRecording) {
            wsRecording.play();
          }
        }, 50);
      });
  }
});

// Keyboard Shortcuts
document.addEventListener("keydown", function (event) {
  // Ignore if typing in an input field
  const activeElement = document.activeElement;
  const isInput =
    activeElement.tagName === "INPUT" ||
    activeElement.tagName === "TEXTAREA" ||
    activeElement.isContentEditable;
  if (isInput) return;

  if (event.code === "Space") {
    event.preventDefault();
    if (event.shiftKey) {
      // Shift + Space -> Play Reference
      playAudio();
    } else {
      // Space -> Play Recording
      playRecording();
    }
  } else if (event.code === "ArrowRight") {
    event.preventDefault();
    getNextSample();
  } else if (event.code === "ArrowDown") {
    // ArrowDown -> Record
    event.preventDefault();
    updateRecordingState();
  } else if (event.code === "Escape" && ENABLE_WAVEFORM_REGIONS) {
    // Escape -> Clear all selected regions/intervals (only if regions enabled)
    event.preventDefault();
    if (wsRegionsReference) {
      wsRegionsReference.clearRegions();
    }
    if (wsRegionsRecording) {
      wsRegionsRecording.clearRegions();
    }
  }
});

// Force Difficulty Check on Init
document.addEventListener("DOMContentLoaded", () => {
  // Check which radio is checked by default
  if (document.getElementById("lengthCat1").checked) sample_difficult = 0;
  else if (document.getElementById("lengthCat2").checked) sample_difficult = 1;
  else if (document.getElementById("lengthCat3").checked) sample_difficult = 2;
  else if (document.getElementById("lengthCat4").checked) sample_difficult = 3;
});
