import openai
import os
import wave


BOSON_API_KEY = os.getenv("BOSON_API_KEY")

client = openai.Client(
    api_key=BOSON_API_KEY,
    base_url="https://hackathon.boson.ai/v1"
)

# for this api, we onlu support PCM format output
response = client.audio.speech.create(
    model="higgs-audio-generation-Hackathon",
    voice="zh_man_sichuan",
    content="Hey guys, welcome to MAT 2 9 2 O. D. E.",
    response_format="pcm"
)

# You can use these parameters to write PCM data to a WAV file
num_channels = 1        
sample_width = 2        
sample_rate = 24000   

pcm_data = response.content

with wave.open('belinda_test.wav', 'wb') as wav:
    wav.setnchannels(num_channels)
    wav.setsampwidth(sample_width)
    wav.setframerate(sample_rate)
    wav.writeframes(pcm_data)
