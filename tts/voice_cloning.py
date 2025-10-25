from openai import OpenAI
import base64
import os

BOSON_API_KEY = os.getenv("BOSON_API_KEY")
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode("utf-8")

def read_transcript(file_path: str, remove_timestamps: bool = True, join_lines: bool = True) -> str:
    """
    Reads a transcript text file and returns it as a single string.
    
    Args:
        file_path: Path to the transcript .txt file
        remove_timestamps: If True, removes [mm:ss] or [hh:mm:ss] timestamps
        join_lines: If True, joins all lines into one paragraph (with spaces)
    
    Returns:
        A string of the processed transcript
    """
    import re
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed = []
    for line in lines:
        line = line.strip()
        if remove_timestamps:
            line = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]\s*', '', line)
        if line:
            processed.append(line)
    
    if join_lines:
        return ' '.join(processed)
    else:
        return '\n'.join(processed)


reference_path = "./input/guerzhoyb.wav"
reference_transcript = read_transcript("transcript_guerzhoy_a.txt", True, True) # change input file names

resp = client.chat.completions.create(
    model="higgs-audio-generation-Hackathon",
    messages=[
        {"role": "user", "content": reference_transcript},
        {
            "role": "assistant",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": b64(reference_path), "format": "wav"}
            }],
        },
        {"role": "user", "content": "C’est la lutte finale :Groupons-nous, et demain, L’Internationale， Sera le genre humain"},
    ],
    modalities=["text", "audio"],
    max_completion_tokens=4096,
    temperature=1.0,
    top_p=0.95,
    stream=False,
    stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
    extra_body={"top_k": 50},
)

audio_b64 = resp.choices[0].message.audio.data
open("./output/output.wav", "wb").write(base64.b64decode(audio_b64)) # change output file name