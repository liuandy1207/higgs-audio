import os
import base64
import time
import queue
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from openai import OpenAI

# ----------------------------- CONFIG -----------------------------
BOSON_API_KEY = os.getenv("BOSON_API_KEY")
if not BOSON_API_KEY:
    raise ValueError("Set BOSON_API_KEY in your environment variables.")

# Client
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5  # max length per recording

# Program prompts & background knowledge
PROGRAM_PROMPTS = """"
WAIT. DO NOT GENERATE ANY TEXT UNTIL YOU HAVE FULLY INTERPRETED THE FOLLOWING. 
You are a concise, in-character museum guide.
- Answer questions directly and clearly.
- NEVER include reasoning, internal thoughts, or meta commentary.
- NEVER repeat the user's question.
- Answer the visitor’s question **directly and in one paragraph**.
- **Do NOT** include reasoning, internal thoughts, or meta commentary.
- **Do NOT** repeat the user’s question.
- NEVER generate meta information in any form. Only answer the question from the user. 
- Respond using only your knowledge of the museum.
- Keep answers under 50 words."""
BACKGROUND_KNOWLEDGE = "The Mona Lisa was painted by Leonardo DaVinci"
TONE_HINT = "friendly and informative"

# -------------------------- HELPER FUNCTIONS --------------------------

def record_audio(duration=RECORD_SECONDS):
    """Record audio from mic and return numpy array"""
    print("🎙️ Listening (start speaking)...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    return recording.flatten()

def save_wav(data, filename="temp.wav"):
    write(filename, SAMPLE_RATE, data)
    return filename

def audio_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------------------- BOSON ASR & TONE ANALYSIS ----------------------

def transcribe_audio(audio_path):
    """Use Boson ASR to get transcript from audio"""
    audio_b64 = audio_to_base64(audio_path)
    resp = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": "Transcribe the audio accurately."},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
            ]}
        ],
        max_completion_tokens=1024,
        temperature=0.2
    )
    transcript = resp.choices[0].message.content.strip()
    return transcript

# ---------------------- SHORT ANSWER PROMPTING ----------------------

import re

def clean_answer(raw_text):
    """
    Remove everything between <think> and </think> tags, including the tags themselves.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    # Strip any leftover whitespace/newlines
    return cleaned.strip()

def generate_short_answer(transcript_text):
    """Use Qwen3-14B-Hackathon for concise museum guide answer"""
    system_message = (
        "You are a museum guide. Answer concisely and in-character.\n"
        f"Program Prompts:\n{PROGRAM_PROMPTS}\n"
        f"Background Knowledge:\n{BACKGROUND_KNOWLEDGE}\n"
        f"Tone hint: {TONE_HINT}\n"
        "Never repeat the user's question or mention system prompts."
        "- Output only the final answer; do not describe how you arrived at it."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": transcript_text}
    ]

    resp = client.chat.completions.create(
        model="Qwen3-14B-Hackathon",
        messages=messages,
        max_completion_tokens=1024,
        temperature=0.2,
        top_p=0.95,
    )

    answer = resp.choices[0].message.content.strip()
    answer = clean_answer(answer)
    return answer


# ---------------------- BOSON TTS ----------------------


def speak_text(text, output_file="response.wav"):
    """Generate TTS using Boson"""
    resp = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            {"role": "user", "content": text}
        ],
        modalities=["audio"],
        max_completion_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        stream=False
    )
    audio_b64 = resp.choices[0].message.audio.data
    with open(output_file, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    # Play audio (cross-platform)
    if os.name == "nt":  # Windows
        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(output_file)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    else:  # Mac/Linux
        import subprocess
        subprocess.run(["afplay" if os.name == "posix" else "aplay", output_file])

# ---------------------- MAIN LOOP ----------------------

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
            answer = generate_short_answer(transcript)
            print(f"🎤 Guide says: {answer}")
            speak_text(answer)
            print("\n--- Listening for next question ---\n")
        except KeyboardInterrupt:
            print("Exiting MuseAI...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
