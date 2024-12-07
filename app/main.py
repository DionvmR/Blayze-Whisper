from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import yt_dlp
import asyncio
import os
from dotenv import load_dotenv
from transcribe import compress_audio, transcribe_audio

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionRequest(BaseModel):
    url: str

@app.post("/api/transcribe")
async def transcribe(request: TranscriptionRequest):
    try:
        # Download audio if it's a YouTube URL
        if "youtube.com" in request.url or "youtu.be" in request.url:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }]
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(request.url, download=True)
                audio_path = info['id'] + ".mp3"
        else:
            # Handle direct video/audio file uploads
            audio_path = request.url

        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe audio
        result = model.transcribe(audio_path)
        
        # Return segments with timestamps
        return result["segments"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv('OPENAI_API_KEY') 

# When you receive a file:
compressed_file = compress_audio("path/to/uploaded/file.mp4")
transcription = transcribe_audio(compressed_file) 