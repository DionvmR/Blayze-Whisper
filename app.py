from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
import json
import asyncio
import whisper
import yt_dlp
import os
import warnings
import torch
import numpy as np

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

app = FastAPI()

# Whisper uses 16kHz audio
SAMPLE_RATE = 16000

def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube video"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'uploads/%(title)s.%(ext)s',
        'nocheckcertificate': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = f"uploads/{info['title']}.mp3"
        return audio_file

async def process_audio(audio_file: str):
    """Process audio file and yield segments in real-time"""
    print("Starting transcription process...")
    model = whisper.load_model("base")
    print("Model loaded, starting transcription...")
    
    # Load audio
    audio = whisper.load_audio(audio_file)
    
    # Get duration and calculate chunk size (e.g., 30 seconds)
    duration = len(audio) / SAMPLE_RATE
    chunk_duration = 30.0  # seconds
    chunk_size = int(SAMPLE_RATE * chunk_duration)
    
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Processing in {chunk_duration}-second chunks")
    
    # Process audio in chunks
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        
        # Pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        # Transcribe chunk
        result = model.transcribe(chunk)
        
        # Adjust timestamp to account for chunk position
        chunk_start_time = i / SAMPLE_RATE
        for segment in result["segments"]:
            segment["start"] += chunk_start_time
            yield {
                "type": "segment",
                "data": {
                    "start": segment["start"],
                    "text": segment["text"].strip()
                }
            }
            
        print(f"Processed chunk starting at {chunk_start_time:.2f} seconds")

@app.get("/stream-transcription")
async def stream_transcription(request: Request, url: str):
    async def event_generator():
        try:
            print(f"Starting download for URL: {url}")
            audio_file = download_youtube_audio(url)
            print(f"Download complete: {audio_file}")
            
            async for segment in process_audio(audio_file):
                if await request.is_disconnected():
                    print("Client disconnected")
                    break
                
                # Send pure JSON without any SSE formatting
                data = json.dumps(segment)
                print(f"Sending data: {data}")
                yield f"data: {data}\n\n"
            
            print("Cleaning up audio file...")
            os.remove(audio_file)
            
            # Send completion message
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Blayze AI</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body>
            <div class="min-h-screen bg-gray-50 py-8 px-4">
                <div class="max-w-3xl mx-auto">
                    <h1 class="text-5xl font-bold text-center text-gray-900 mb-2">Blayze AI</h1>
                    <p class="text-center text-gray-600 mb-8">ML-powered speech recognition directly in your browser</p>

                    <div class="bg-white rounded-lg shadow p-6">
                        <div class="flex gap-4 justify-center mb-8">
                            <button class="flex items-center gap-2 px-4 py-2 rounded-md hover:bg-gray-100">
                                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M12.232 4.232a2.5 2.5 0 013.536 3.536l-1.225 1.224a.75.75 0 001.061 1.06l1.224-1.224a4 4 0 00-5.656-5.656l-3 3a4 4 0 00.225 5.865.75.75 0 00.977-1.138 2.5 2.5 0 01-.142-3.667l3-3z" />
                                    <path d="M11.603 7.963a.75.75 0 00-.977 1.138 2.5 2.5 0 01.142 3.667l-3 3a2.5 2.5 0 01-3.536-3.536l1.225-1.224a.75.75 0 00-1.061-1.06l-1.224 1.224a4 4 0 105.656 5.656l3-3a4 4 0 00-.225-5.865z" />
                                </svg>
                                From URL
                            </button>
                            <button class="flex items-center gap-2 px-4 py-2 rounded-md hover:bg-gray-100">
                                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M9.25 13.25a.75.75 0 001.5 0V4.636l2.955 3.129a.75.75 0 001.09-1.03l-4.25-4.5a.75.75 0 00-1.09 0l-4.25 4.5a.75.75 0 101.09 1.03L9.25 4.636v8.614z" />
                                    <path d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z" />
                                </svg>
                                From File
                            </button>
                        </div>

                        <form id="transcribe-form" class="space-y-4">
                            <input 
                                type="text" 
                                name="url" 
                                placeholder="Enter YouTube URL" 
                                required
                                class="w-full p-2 border rounded-md"
                            >
                            <button 
                                type="submit" 
                                id="submit-btn"
                                class="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 disabled:bg-blue-300 flex items-center justify-center gap-2"
                            >
                                <div class="spinner hidden w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                <span>Transcribe</span>
                            </button>
                        </form>

                        <div id="transcription" class="mt-8 space-y-2"></div>
                    </div>
                </div>
            </div>

            <script>
            document.getElementById('transcribe-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const form = e.target;
                const btn = document.getElementById('submit-btn');
                const spinner = btn.querySelector('.spinner');
                const btnText = btn.querySelector('span');
                const transcriptionDiv = document.getElementById('transcription');
                
                // Clear previous transcription
                transcriptionDiv.innerHTML = '';
                
                btn.disabled = true;
                spinner.classList.remove('hidden');
                btnText.textContent = 'Transcribing...';
                
                const url = new FormData(form).get('url');
                const eventSource = new EventSource(`/stream-transcription?url=${encodeURIComponent(url)}`);
                
                eventSource.onmessage = function(event) {
                    try {
                        console.log('Raw event data:', event.data);
                        const jsonStr = event.data.replace(/^data: /, '');
                        console.log('Cleaned JSON string:', jsonStr);
                        const data = JSON.parse(jsonStr);
                        console.log('Parsed data:', data);
                        
                        if (data.type === 'segment' && data.data) {
                            const segmentDiv = document.createElement('div');
                            segmentDiv.className = 'p-4 bg-gray-50 rounded-lg mb-2';
                            segmentDiv.innerHTML = `
                                <span class="text-gray-500 mr-2">[${formatTimestamp(data.data.start)}]</span>
                                <span>${data.data.text}</span>
                            `;
                            transcriptionDiv.appendChild(segmentDiv);
                            
                            window.scrollTo({
                                top: document.body.scrollHeight,
                                behavior: 'smooth'
                            });
                        } else if (data.type === 'complete') {
                            eventSource.close();
                            btn.disabled = false;
                            spinner.classList.add('hidden');
                            btnText.textContent = 'Transcribe';
                        }
                    } catch (error) {
                        console.error('Error processing message:', error);
                        console.error('Raw event data that caused error:', event.data);
                    }
                };
                
                eventSource.onerror = function(error) {
                    console.error('EventSource error:', error);
                    eventSource.close();
                    btn.disabled = false;
                    spinner.classList.add('hidden');
                    btnText.textContent = 'Transcribe';
                };
            });

            function formatTimestamp(seconds) {
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                const remainingSeconds = Math.floor(seconds % 60);
                
                // Always return in HH:MM:SS format
                return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
            }
            </script>
        </body>
    </html>
    """

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 