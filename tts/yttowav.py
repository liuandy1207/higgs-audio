import os
import yt_dlp
from pydub import AudioSegment


def download_youtube_as_wav(url: str, output_path: str = ".", filename: str | None = None) -> str:
    """
    Downloads a YouTube video as audio and converts it to a WAV file.
    
    Args:
        url: The YouTube video URL.
        output_path: Directory to save the WAV file.
        filename: Optional name for the WAV file (without extension).
    
    Returns:
        The path to the saved WAV file.
    """
    # Ensure output path exists
    os.makedirs(output_path, exist_ok=True)
    
    # Temporary download as best audio
    temp_file = os.path.join(output_path, "temp_audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": temp_file,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = ydl.prepare_filename(info)
    
    # Convert to WAV
    base_name = filename or info.get("title", "output").replace("/", "_")
    wav_file = os.path.join(output_path, f"{base_name}.wav")

    # Load and export using pydub
    AudioSegment.from_file(downloaded_file).export(wav_file, format="wav")

    # Clean up temporary file if different
    if os.path.exists(downloaded_file):
        os.remove(downloaded_file)
    
    return wav_file


if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=iN4BQ8sO7yc"
    output = download_youtube_as_wav(url, "input")
    print(f"✅ Saved as: {output}")
