###########################################################################
#                                IMPORTS
###########################################################################
import json
import os
import base64
import time
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
import re
import subprocess
import numpy as np

###########################################################################
#                         CONFIG AND PROMPTS
###########################################################################
BOSON_API_KEY = os.getenv("BOSON_API_KEY")
if not BOSON_API_KEY:
    raise ValueError("Set BOSON_API_KEY in your environment variables.")
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1                

###########################################################################
#                               HELPERS
###########################################################################

def record_audio(
    max_duration=10,           # safety timeout (seconds)
    silence_threshold=600,     # loudness threshold to detect voice
    silence_duration=2.0,      # silence time to stop recording
    pre_speech_padding=0.5,    # time to keep in buffer
):
    print("🎙️ Waiting for speech...")
    chunk_size = int(0.1 * SAMPLE_RATE)    # capture 0.1s per chunk
    buffer = []                            # temp storage for capture audio before recording
    recording = []                         # storage for the recording
    silence_chunks = 0                     # counter for silent chunks
    started = False                        # flag for recording start
    start_time = time.time()               # timer for timeout

    # prepare stream for recording
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    try:
        # starts stream
        with stream:
            while True:
                # read a chunk
                chunk, _ = stream.read(chunk_size)
                # save the chunk to buffer
                buffer.extend(chunk.flatten())

                # convert audio to a float for more precision (from int16)
                chunk_float = chunk.astype(np.float32)
                # calculate rms (measure of loudness) to detect start of speech
                rms = np.sqrt(np.mean(np.square(chunk_float)))

                # detect start of speech
                if not started and rms > silence_threshold:
                    started = True
                    print("🎤 Speech detected, recording...")
                    # include pre-speech padding to the recording, so that information isnt cut-off
                    recording.extend(buffer[-int(pre_speech_padding * SAMPLE_RATE):]) 
                    buffer = []                 # clear old buffer
                    silence_chunks = 0          # reset silence counter

                # while recording
                elif started:
                    # add a chunk to the recording
                    recording.extend(chunk.flatten())

                    # count silent chunks
                    if rms < silence_threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0

                    # stop after a specific amount of silence
                    if silence_chunks * 0.1 >= silence_duration:
                        print("🛑 Speech ended.")
                        break

                # safety timeout to prevent infinite listening
                if time.time() - start_time > max_duration:
                    print("⏰ Timeout reached, stopping.")
                    break

        # bulletproofing
        if not started:
            print("⚠️ No speech detected.")
            return np.array([], dtype='int16')

        return np.array(recording, dtype='int16')
    
    # ensure stream is properly closed, even if errors occur
    finally:
        stream.close()

def save_wav(data, filename="temp.wav"):
    # clean overwrite by deleting existing
    if os.path.exists(filename):
        os.remove(filename)
    write(filename, SAMPLE_RATE, data)
    return filename

def audio_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

###########################################################################
#                             TRANSCRIBER
###########################################################################

TRANSCRIBER_PROMPT="""
Listen carefully to the audio and transcribe it accurately in the same language.
Ignore background noise, non-verbal sounds, or pauses. 
Return only the spoken text.
"""

def transcribe_audio(audio_path):
    audio_b64 = audio_to_base64(audio_path)
    # api call to the TRANSCRIBER
    resp = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": TRANSCRIBER_PROMPT},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
            ]}
        ],
        max_completion_tokens=128,
        temperature=0.2
    )
    transcript = resp.choices[0].message.content.strip()
    return transcript

###########################################################################
#                           VOICE ANALYZER
###########################################################################

VOICE_ANALYZER_PROMPT="""
You are an expert voice analysis and visitor profiling system for a museum guide AI.
Analyze the visitor’s **spoken audio** carefully and return a structured JSON profile to help the AI generate responses appropriately.

Analyze and estimate the following aspects:

- **estimated_age_group**: one of ("child", "teen", "adult", "elderly")
- **interest_level**: one of ("low", "medium", "high") — based on speech energy, tempo, and clarity. High = lively, energetic tone; Low = slow, flat, or disengaged.
- **emotional_tone**: one of ("neutral", "curious", "happy", "bored", "angry", "sad", "excited", "confused")
- **speaking_style**: short description of manner (e.g., "confident and expressive", "quiet and hesitant", "formal and clear")
- **engagement_type**: one of ("asking a question", "making a comment", "expressing emotion", "unclear")
- **language_level**: one of ("simple", "average", "complex") — estimate from vocabulary and phrasing complexity
- **spoken_language**: detect the language being spoken (e.g., "English", "French")
- **speaking_speed**: one of ("slow", "moderate", "fast") — based on words per minute and rhythm of speech

Return **only valid JSON** with keys exactly as shown below:

{
  "estimated_age_group": "",
  "interest_level": "",
  "emotional_tone": "",
  "speaking_style": "",
  "engagement_type": "",
  "language_level": "",
  "spoken_language": "",
  "speaking_speed": ""
}
"""

def analyze_voice(audio_path):
    audio_b64 = audio_to_base64(audio_path)
    # api call to the VOICE ANALYZER
    resp = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": VOICE_ANALYZER_PROMPT},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
            ]}
        ],
        max_completion_tokens=512,
        temperature=0.4             # randomness a bit higher
    )

    import json
    raw_output = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw_output)
    # bulletproofing
    except json.JSONDecodeError:
        # Return full set of keys with "unknown" defaults
        return {
            "estimated_age_group": "unknown",
            "interest_level": "unknown",
            "emotional_tone": "neutral",
            "speaking_style": "unknown",
            "engagement_type": "unknown",
            "language_level": "unknown",
            "spoken_language": "unknown",
            "speaking_speed": "unknown"
        }

###########################################################################
#                             VOICE CREATOR
###########################################################################

VOICE_CREATOR_INSTRUCTION_PROMPT = """
You are an advanced assistant that generates instructions for a text-to-speech AI. 
Your goal is to produce a concise, detailed prompt that will guide the TTS to speak in a voice **tailored to a specific visitor**, based on their profile.

Input (Visitor Profile):
- estimated_age_group: {estimated_age_group}  # child, teen, adult, elderly
- interest_level: {interest_level}          # low, medium, high
- emotional_tone: {emotional_tone}          # neutral, curious, happy, bored, angry, sad, excited, confused
- speaking_style: {speaking_style}          # e.g., "confident and expressive", "quiet and hesitant"
- engagement_type: {engagement_type}        # asking a question, making a comment, expressing emotion, unclear
- language_level: {language_level}          # simple, average, complex
- spoken_language: {spoken_language}        # language of visitor
- speaking_speed: {speaking_speed}          # slow, moderate, fast

Instructions:
1. Analyze each attribute of the visitor profile and reason about **how it should influence the TTS voice**:
   - Pitch, tone, and energy
   - Speech speed and rhythm
   - Emotional inflection
   - Vocabulary and phrasing style

2. Generate a **clear instruction prompt** for the TTS system that specifies:
   - How the speech should sound (pitch, tone, energy)
   - How fast or slow it should be delivered
   - How expressive or formal it should sound
   - Any language or style adjustments for clarity and engagement

3. Output the instructions in natural language **ready to be fed into a TTS system**, without extra commentary or formatting.

Example Output:
"Speak in a friendly, high-energy, slightly playful tone, moderate pace, clear pronunciation, using simple vocabulary suitable for a curious child."

Your task: Reason carefully and produce the most precise, TTS-ready instructions for this visitor.
"""

def clean_answer(raw_text):
    """
    Remove everything between <think> and </think> tags, including the tags themselves.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    cleaned = re.sub(r"\[.*?\]", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

def voice_style_from_analysis(voice_data):
    if isinstance(voice_data, dict):
        voice_data_str = json.dumps(voice_data, indent=2)
    else:
        voice_data_str = str(voice_data)
    messages = [
        {"role": "system", "content": VOICE_CREATOR_INSTRUCTION_PROMPT},
        {"role": "user", "content": voice_data_str}
    ]
    # api call to the VOICE CREATOR
    resp = client.chat.completions.create(
        model="Qwen3-32B-thinking-Hackathon",
        messages=messages,
        max_completion_tokens=2048,
        temperature=0.2,
        top_p=0.80,         # lower for more focused responses
    )
    prompt = resp.choices[0].message.content.strip()
    prompt = clean_answer(prompt)
    return prompt


###########################################################################
#                        GENERATE SHORT ANSWER
###########################################################################

GENERATOR_PROMPT = """
You are a professional museum art gallery guide generating text that will be spoken by a TTS engine. 
Your goal is to produce concise, polite, and engaging responses tailored to the visitor's profile. The output will be read aloud by a TTS system, so phrasing, vocabulary, and style must match the visitor's characteristics.

Input:
- Visitor JSON profile:
  - estimated_age_group: {estimated_age_group}   # child, teen, adult, elderly
  - interest_level: {interest_level}           # low, medium, high
  - emotional_tone: {emotional_tone}           # neutral, curious, happy, bored, angry, sad, excited, confused
  - speaking_style: {speaking_style}           # e.g., "confident and expressive", "quiet and hesitant"
  - engagement_type: {engagement_type}         # asking a question, making a comment, expressing emotion, unclear
  - language_level: {language_level}           # simple, average, complex
  - intelligence_score: {intelligence_score}   # optional, guides complexity of content
  - spoken_language: {spoken_language}         # language of the visitor
  - speaking_speed: {speaking_speed}           # slow, moderate, fast

Guidelines:
1. Always produce text that will sound natural when read aloud by a TTS system.
2. Adjust **vocabulary, sentence length, and complexity** based on the visitor's intelligence, age, and language_level.
3. Match tone and expressiveness to emotional_tone, interest_level, and speaking_style.
4. Keep answers concise, polite, and kind. Limit to one paragraph, under 50 words.
5. Base content on the background knowledge provided.
6. If the visitor's question is unclear, nonsensical, or just breathing/noise, return an empty string.

Example:
Visitor: {"estimated_age_group": "adult", "interest_level": "high", "emotional_tone": "curious", "language_level": "complex"}
Output: "Leonardo DaVinci's Mona Lisa, painted in the early 16th century, exemplifies Renaissance mastery with subtle expressions and exquisite technique."

Your task: Generate the **text content** for TTS that fulfills all these requirements.
"""

BACKGROUND_KNOWLEDGE = "The Mona Lisa was painted by Leonardo DaVinci."

def generate_short_answer(transcript_text, tone_hint):
    messages = [
        {"role": "system", "content": f"{GENERATOR_PROMPT}\n\n{tone_hint}"},
        {"role": "user", "content": transcript_text}
    ]
    # api call to the GENERATOR
    resp = client.chat.completions.create(
        model="Qwen3-32B-thinking-Hackathon",
        messages=messages,
        max_completion_tokens=2048,
        temperature=0.5,
        top_p=0.80,         # lower for more focused responses
    )
    answer = resp.choices[0].message.content.strip()
    answer = clean_answer(answer)       # because it doesnt behave
    return answer

###########################################################################
#                            TEXT TO SPEECH
###########################################################################

def speak_text(text, tone_hint):
    # Launch ffplay to play audio chunks as they come in
    proc = subprocess.Popen(
        ["ffplay", "-f", "s16le", "-ar", "24000", "-i", "-", "-nodisp", "-autoexit", "-loglevel", "error"],
        stdin=subprocess.PIPE,
    )

    # Create the stream
    stream = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[{"role": "user", "content": text},
                  {"role": "system", "content": tone_hint}],
        audio={"format": "pcm16"},
        max_completion_tokens=512,
        temperature=0.8,
        top_p=0.95,
        stream=True,
    )

    # do stuff chunk by chunk
    for chunk in stream:
        if proc.poll() is not None:
            break
        delta = getattr(chunk.choices[0], "delta", None)
        audio = getattr(delta, "audio", None)
        if not audio:
            continue
        data = base64.b64decode(audio["data"])
        proc.stdin.write(data)
        proc.stdin.flush()

    if proc.stdin:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
    proc.wait()

###########################################################################
#                            MAIN LOOP
###########################################################################

def main():
    print("=== MuseAI Museum Guide ===")
    while True:
        try:
            audio_data = record_audio()
            audio_file = save_wav(audio_data)
            transcript = transcribe_audio(audio_file)
            if not transcript.strip():
                print("⚠️ No speech detected.")
                continue
            print(f"🧠 Transcript: {transcript}")

            voice_data = analyze_voice(audio_file)
            print(f"🧠 vd: {voice_data}")
            tone_hint = voice_style_from_analysis(voice_data) 
            print(f"tone_hint = {tone_hint}")
            answer = generate_short_answer(transcript, tone_hint)
            print(f"🎤 Guide says: {answer}")
            speak_text(answer, tone_hint)
            print("\n--- Listening for next question ---\n")
        except KeyboardInterrupt:
            print("Exiting MuseAI...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
