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
                
                yield {"event": "message", "data": json.dumps(segment)}
            
            print("Cleaning up audio file...")
            os.remove(audio_file)
            
            # Send completion message
            yield {"event": "message", "data": json.dumps({'type': 'complete'})}
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            yield {"event": "message", "data": json.dumps({'type': 'error', 'message': str(e)})}

    return EventSourceResponse(event_generator())

@app.get("/", response_class=HTMLResponse)
async def home():
    return '''
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
                        
                        <div id="download-buttons" class="mt-4 flex gap-4 justify-center hidden">
                            <button 
                                id="download-txt"
                                class="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                            >
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                                </svg>
                                Download as TXT
                            </button>
                            <button 
                                id="download-json"
                                class="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                            >
                                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                                </svg>
                                Download as JSON
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                let transcriptSegments = [];
                
                document.getElementById("download-txt").addEventListener("click", () => downloadTranscript("txt"));
                document.getElementById("download-json").addEventListener("click", () => downloadTranscript("json"));
                
                document.getElementById("transcribe-form").addEventListener("submit", function(e) {
                    e.preventDefault();
                    
                    const form = e.target;
                    const btn = document.getElementById("submit-btn");
                    const spinner = btn.querySelector(".spinner");
                    const btnText = btn.querySelector("span");
                    const transcriptionDiv = document.getElementById("transcription");
                    const downloadButtons = document.getElementById("download-buttons");
                    
                    transcriptionDiv.innerHTML = "";
                    downloadButtons.classList.add("hidden");
                    transcriptSegments = [];
                    
                    btn.disabled = true;
                    spinner.classList.remove("hidden");
                    btnText.textContent = "Transcribing...";
                    
                    const url = new FormData(form).get("url");
                    const eventSource = new EventSource(`/stream-transcription?url=${encodeURIComponent(url)}`);
                    
                    eventSource.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            
                            if (data.type === "segment" && data.data) {
                                transcriptSegments.push(data.data);
                                
                                const segmentDiv = document.createElement("div");
                                segmentDiv.className = "p-4 bg-gray-50 rounded-lg mb-2";
                                segmentDiv.innerHTML = `
                                    <span class="text-gray-500 mr-2">[${formatTimestamp(data.data.start)}]</span>
                                    <span>${data.data.text}</span>
                                `;
                                transcriptionDiv.appendChild(segmentDiv);
                                
                                window.scrollTo({
                                    top: document.body.scrollHeight,
                                    behavior: "smooth"
                                });
                            } else if (data.type === "complete") {
                                console.log("Transcription complete, showing download buttons");
                                eventSource.close();
                                btn.disabled = false;
                                spinner.classList.add("hidden");
                                btnText.textContent = "Transcribe";
                                
                                if (transcriptSegments.length > 0) {
                                    downloadButtons.classList.remove("hidden");
                                }
                            }
                        } catch (error) {
                            console.error("Error processing message:", error);
                            console.error("Raw event data that caused error:", event.data);
                        }
                    };
                    
                    eventSource.onerror = function(error) {
                        console.error("EventSource error:", error);
                        eventSource.close();
                        btn.disabled = false;
                        spinner.classList.add("hidden");
                        btnText.textContent = "Transcribe";
                    };
                });

                function formatTimestamp(seconds) {
                    const hours = Math.floor(seconds / 3600);
                    const minutes = Math.floor((seconds % 3600) / 60);
                    const remainingSeconds = Math.floor(seconds % 60);
                    return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`;
                }
                
                function downloadTranscript(format) {
                    console.log("Downloading transcript in format:", format);
                    console.log("Number of segments:", transcriptSegments.length);
                    
                    if (transcriptSegments.length === 0) {
                        console.error("No transcript segments to download");
                        return;
                    }
                    
                    let content;
                    let filename;
                    let mimeType;
                    
                    if (format === "txt") {
                        content = transcriptSegments
                            .map(segment => `[${formatTimestamp(segment.start)}] ${segment.text}`)
                            .join("\\n");
                        filename = "transcript.txt";
                        mimeType = "text/plain";
                    } else {
                        content = JSON.stringify({
                            segments: transcriptSegments,
                            metadata: {
                                totalSegments: transcriptSegments.length,
                                generatedAt: new Date().toISOString()
                            }
                        }, null, 2);
                        filename = "transcript.json";
                        mimeType = "application/json";
                    }
                    
                    console.log("Creating download with content length:", content.length);
                    
                    const blob = new Blob([content], { type: mimeType });
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.style.display = "none";
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                }
            </script>
        </body>
    </html>
    '''

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)