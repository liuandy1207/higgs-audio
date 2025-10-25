import os
import base64
import time
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
import re
import subprocess
import numpy as np
import time

# source .venv/bin/activate

###########################################################################
#                                CONFIG
###########################################################################
BOSON_API_KEY = os.getenv("BOSON_API_KEY") # do "export BOSON_API_KEY=****"
if not BOSON_API_KEY:
    raise ValueError("Set BOSON_API_KEY in your environment variables.")
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

# Audio settings, all standard amounts for audio recordings
SAMPLE_RATE = 16000
CHANNELS = 1                
RECORD_SECONDS = 5  # max length per recording

# Program prompts
PROGRAM_PROMPTS = """
- You are a professional museum art gallery guide helping out guests with their questions.
- Answer in a concise, polite, and kind manner.
- Answer in one paragraph, under 50 words.
- Try to generate your answer based on the background knowledge. 
- If the user's question does not make sense (unintelligible, breathing, or nonsensical), then say nothing.
- Never repeat the user's question or mention system prompts.
- Use the voice data to make the response better for the specific user and context (consider diction). 
"""
BACKGROUND_KNOWLEDGE = "The Mona Lisa was painted by Leonardo DaVinci."
# TONE_HINT = "friendly and informative" # make this better later

###########################################################################
#                                HELPERS
###########################################################################

def record_audio(
    max_duration=10,           # hard cap, safety timeout
    silence_threshold=500,     # loudness threshold to detect voice
    silence_duration=2.0,      # stop 2s after silence
    pre_speech_padding=0.3,    # record a bit before first speech
):
    print("🎙️ Waiting for speech...")

    chunk_size = int(0.1 * SAMPLE_RATE)
    buffer = []
    recording = []
    silence_chunks = 0
    started = False
    start_time = time.time()

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    try:
        with stream:
            while True:
                chunk, _ = stream.read(chunk_size)
                buffer.extend(chunk.flatten())

                # Convert safely to float for RMS calculation
                chunk_float = chunk.astype(np.float32)
                rms = np.sqrt(np.mean(np.square(chunk_float)))

                # detect start of speech
                if not started and rms > silence_threshold:
                    started = True
                    print("🎤 Speech detected, recording...")
                    recording.extend(buffer[-int(pre_speech_padding * SAMPLE_RATE):])  # include pre-speech
                    buffer = []  # clear old buffer
                    silence_chunks = 0

                elif started:
                    recording.extend(chunk.flatten())

                    if rms < silence_threshold:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0

                    # stop 2s after silence
                    if silence_chunks * 0.1 >= silence_duration:
                        print("🛑 Speech ended.")
                        break

                # safety stop (no infinite listening)
                if time.time() - start_time > max_duration:
                    print("⏰ Timeout reached, stopping.")
                    break

        if not started:
            print("⚠️ No speech detected.")
            return np.array([], dtype='int16')

        return np.array(recording, dtype='int16')
    finally:
        stream.close()


def save_wav(data, filename="temp.wav"):
    # Always overwrite the file cleanly
    if os.path.exists(filename):
        os.remove(filename)
    write(filename, SAMPLE_RATE, data)
    return filename

def audio_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

###########################################################################
#                             TRANSCRIBE
###########################################################################

def transcribe_audio(audio_path):
    audio_b64 = audio_to_base64(audio_path)
    resp = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": "Transcribe the audio accurately and only return text."},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
            ]}
        ],
        max_completion_tokens=512,
        temperature=0.2
    )
    transcript = resp.choices[0].message.content.strip()
    return transcript

###########################################################################
#                        ANALYZE VOCAL QUALITIES
###########################################################################

def analyze_voice(audio_path):
    audio_b64 = audio_to_base64(audio_path)
    resp = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": """
You are an expert voice analysis and visitor profiling system for a museum guide AI.
Your goal is to analyze the visitor’s **spoken audio** and return a structured JSON profile that will help the AI generate and voice responses appropriately.

Analyze the recording carefully for:
- **estimated_age_group**: one of ("child", "teen", "adult", "elderly")
- **interest_level**: one of ("low", "medium", "high") — based on speech energy, tempo, and clarity. High = lively, energetic tone; Low = slow, flat, or disengaged.
- **emotional_tone**: one of ("neutral", "curious", "happy", "bored", "angry", "sad", "excited", "confused")
- **speaking_style**: short description (e.g. "confident and expressive", "quiet and hesitant", "formal and clear").
- **engagement_type**: one of ("asking a question", "making a comment", "expressing emotion", "unclear").
- **language_level**: one of ("simple", "average", "complex") — estimate from vocabulary and phrasing complexity.

Return **only valid JSON** with keys:
{
  "estimated_age_group": "",
  "interest_level": "",
  "emotional_tone": "",
  "speaking_style": "",
  "engagement_type": "",
  "language_level": ""
}
"""},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
            ]}
        ],
        max_completion_tokens=512,
        temperature=0.6
    )

    import json
    raw_output = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"age_group": "unknown", "interest_level": "unknown", "emotion": "neutral"}
    
def voice_style_from_analysis(voice_data):
    style_parts = []
    # Emotion / Tone
    emotion = voice_data.get("emotional_tone", "neutral")
    emotion_map = {
        "happy": "friendly, cheerful",
        "curious": "engaging, inquisitive",
        "bored": "calm, neutral",
        "angry": "firm, serious",
        "sad": "gentle, empathetic",
        "excited": "enthusiastic, lively",
        "confused": "patient, clarifying",
        "neutral": "balanced, clear"
    }
    style_parts.append(emotion_map.get(emotion, "neutral"))

    # Interest
    interest = voice_data.get("interest_level", "medium")
    if interest == "high":
        style_parts.append("energetic")
    elif interest == "low":
        style_parts.append("slow-paced, calm")

    # Age — affects both tone and pacing
    age = voice_data.get("estimated_age_group", "adult")
    if age == "child":
        style_parts.append("higher-pitched, playful, uses simple words")
    elif age == "teen":
        style_parts.append("relatable, slightly casual")
    elif age == "elderly":
        style_parts.append("lower-pitched, slower-paced, respectful")

    # Speaking style nuance
    speaking_style = voice_data.get("speaking_style")
    if speaking_style:
        style_parts.append(speaking_style)

    # Engagement type
    engagement = voice_data.get("engagement_type")
    if engagement == "asking a question":
        style_parts.append("informative and responsive")
    elif engagement == "making a comment":
        style_parts.append("acknowledging and conversational")

    return ", ".join([s.strip() for s in style_parts if s]).strip()



###########################################################################
#                        GENERATE SHORT ANSWER
###########################################################################

def clean_answer(raw_text):
    """
    Remove everything between <think> and </think> tags, including the tags themselves.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    cleaned = re.sub(r"\[.*?\]", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

def generate_short_answer(transcript_text, voice_data):
    system_message = (
        f"Program Prompts:\n{PROGRAM_PROMPTS}\n"
        f"Background Knowledge:\n{BACKGROUND_KNOWLEDGE}\n"
        f"- Visitor Profile: {voice_data}\n"
        f"- Style Hint: {voice_style_from_analysis(voice_data)}\n"
        "Adjust your phrasing complexity, tone, and formality accordingly.\n"
        "Tailor your tone and phrasing appropriately for this visitor.\n"
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": transcript_text}
    ]
    resp = client.chat.completions.create(
        model="Qwen3-14B-Hackathon",
        messages=messages,
        max_completion_tokens=4192,
        temperature=0.2,
        top_p=0.80,         # lower for more focused responses
    )
    answer = resp.choices[0].message.content.strip()
    answer = clean_answer(answer)       # because he doesnt behave
    return answer

###########################################################################
#                            TEXT TO SPEECH
###########################################################################

def speak_text(text, voice_data):
    tone_hint = voice_style_from_analysis(voice_data)

    # Launch ffplay to play raw PCM16 audio chunks as they come in
    proc = subprocess.Popen(
        ["ffplay", "-f", "s16le", "-ar", "24000", "-i", "-", "-nodisp", "-autoexit", "-loglevel", "error"],
        stdin=subprocess.PIPE,
    )

    # Create the stream
    stream = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[{"role": "user", "content": text}],
        modalities=["text", "audio"],
        audio={"format": "pcm16"},
        max_completion_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        extra_body={"tone_hint": tone_hint},
        stream=True,
    )

    # Iterate over streaming events
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
            print("hi")
            transcript = transcribe_audio(audio_file)
            
            if not transcript.strip():
                print("⚠️ No speech detected.")
                continue
            print(f"🧠 Transcript: {transcript}")

            voice_data = analyze_voice(audio_file)
            print(f"🧠 vd: {voice_data}")
            
            answer = generate_short_answer(transcript, voice_data)
            print(f"🎤 Guide says: {answer}")
            speak_text(answer, voice_data)
            print("\n--- Listening for next question ---\n")
        except KeyboardInterrupt:
            print("Exiting MuseAI...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
