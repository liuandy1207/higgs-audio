import sounddevice as sd
import soundfile as sf
import base64
import os
import subprocess
from openai import OpenAI

# ---------------- CONFIG ----------------
BOSON_API_KEY = os.getenv("BOSON_API_KEY")
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

SAMPLE_RATE = 16000
DURATION = 4  # seconds to record
INPUT_WAV = "input.wav"
OUTPUT_WAV = "output.wav"

# ---------------- AUDIO RECORDING ----------------
def record_audio(filename, duration, samplerate):
    print(f"\n🎤 Recording {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Saved recording to {filename}")

# ---------------- ENCODE AUDIO ----------------
def encode_audio_base64(filename):
    with open(filename, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------------- ASR: AUDIO → TEXT ----------------
def transcribe_audio(filename):
    audio_b64 = encode_audio_base64(filename)
    resp = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": "Transcribe the audio."},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
                ]
            }
        ],
        max_completion_tokens=256,
        temperature=0.0
    )
    transcription = resp.choices[0].message.content
    print("📝 Transcribed:", transcription)
    return transcription

# ---------------- QWEN: TEXT → ANSWER ----------------
def generate_answer(prompt_text):
    resp = client.chat.completions.create(
        model="Qwen3-32B-thinking-Hackathon",
        messages=[
            {"role": "system", "content": "You are a helpful museum guide."},
            {"role": "user", "content": prompt_text}
        ],
        max_completion_tokens=512,
        temperature=0.7
    )
    answer = resp.choices[0].message.content
    print("💬 Answer:", answer)
    return answer

# ---------------- TTS: TEXT → AUDIO ----------------
def text_to_speech(answer_text):
    resp = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            {"role": "user", "content": answer_text}
        ],
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=0.95
    )
    audio_b64 = resp.choices[0].message.audio.data
    with open(OUTPUT_WAV, "wb") as f:
        f.write(base64.b64decode(audio_b64))
    print(f"🎵 Generated audio saved to {OUTPUT_WAV}")
    return OUTPUT_WAV

# ---------------- PLAY AUDIO ----------------
import subprocess
import sys
import platform

def play_audio(filename):
    system = platform.system()
    print(f"▶ Playing {filename} on {system}")

    if system == "Darwin":  # macOS
        subprocess.run(["afplay", filename])
    elif system == "Linux":  # Linux
        subprocess.run(["aplay", filename])
    elif system == "Windows":  # Windows
        # Use powershell Media.SoundPlayer
        cmd = f'(New-Object Media.SoundPlayer "{filename}").PlaySync();'
        subprocess.run(["powershell", "-Command", cmd], shell=True)
    else:
        print("❌ Unsupported OS, cannot play audio.")


# ---------------- MAIN LOOP ----------------
def main():
    while True:
        record_audio(INPUT_WAV, DURATION, SAMPLE_RATE)
        question = transcribe_audio(INPUT_WAV)
        if not question.strip():
            print("❌ No speech detected, try again.")
            continue
        answer = generate_answer(question)
        audio_file = text_to_speech(answer)
        play_audio(audio_file)

if __name__ == "__main__":
    main()
