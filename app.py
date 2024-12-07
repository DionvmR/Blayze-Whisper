from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import uvicorn
import yt_dlp
import whisper
import os
import ssl

# Create unverified HTTPS context
ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI()

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

@app.get("/", response_class=HTMLResponse)
async def home(transcription=None, error: str = None):
    transcription_html = ""
    if transcription and isinstance(transcription, dict):
        if "segments" in transcription:
            transcription_html = "".join([
                f'<div class="p-4 bg-gray-50 rounded-lg mb-2">'
                f'<span class="text-gray-500 mr-2">[{segment["start"]:.1f}s]</span>'
                f'<span>{segment["text"]}</span>'
                f'</div>'
                for segment in transcription["segments"]
            ])
        elif "text" in transcription:
            transcription_html = f'<div class="p-4 bg-gray-50 rounded-lg">{transcription["text"]}</div>'

    return f"""
    <html>
        <head>
            <title>Blayze AI</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body>
            <div class="min-h-screen bg-gray-50 py-8 px-4">
                <div class="max-w-3xl mx-auto">
                    <h1 class="text-5xl font-bold text-center text-gray-900 mb-2">Blayze AI</h1>
                    <p class="text-center text-gray-600 mb-8">
                        ML-powered speech recognition directly in your browser
                    </p>

                    <div class="bg-white rounded-lg shadow p-6">
                        <div class="flex gap-4 justify-center mb-8">
                            <button class="flex items-center gap-2 px-4 py-2 rounded-md hover:bg-gray-100">
                                <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M12.232 4.232a2.5 2.5 0 013.536 3.536l-1.225 1.224a.75.75 0 001.061 1.06l1.224-1.224a4 4 0 00-5.656-5.656l-3 3a4 4 0 00.225 5.865.75.75 0 00.977-1.138 2.5 2.5 0 01-.142-3.667l3-3z" />
                                    <path d="M11.603 7.963a.75.75 0 00-.977 1.138 2.5 2.5 0 01.142 3.667l-3 3a2.5 2.5 0 01-3.536-3.536l1.225-1.224a.75.75 0 00-1.061-1.06l-1.224 1.224a4 4 0 105.656 5.656l3-3a4 4 0 00-.225-5.865z" />
                                </svg>
                                From URL
                            </button>
                            <button class="flex items-center gap-2 px-4 py-2 rounded-md hover:bg-gray-100">
                                <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M9.25 13.25a.75.75 0 001.5 0V4.636l2.955 3.129a.75.75 0 001.09-1.03l-4.25-4.5a.75.75 0 00-1.09 0l-4.25 4.5a.75.75 0 101.09 1.03L9.25 4.636v8.614z" />
                                    <path d="M3.5 12.75a.75.75 0 00-1.5 0v2.5A2.75 2.75 0 004.75 18h10.5A2.75 2.75 0 0018 15.25v-2.5a.75.75 0 00-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5z" />
                                </svg>
                                From file
                            </button>
                            <button class="flex items-center gap-2 px-4 py-2 rounded-md hover:bg-gray-100">
                                <svg width="20" height="20" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4z" />
                                    <path d="M5.5 9.643a.75.75 0 00-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-1.5v-1.546A6.001 6.001 0 0016 10v-.357a.75.75 0 00-1.5 0V10a4.5 4.5 0 01-9 0v-.357z" />
                                </svg>
                                Record
                            </button>
                        </div>

                        <form id="transcribe-form" action="/transcribe" method="post" class="space-y-4">
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

                        {f'<div class="mt-4 p-4 bg-red-50 text-red-700 rounded-md">{error}</div>' if error else ''}
                        
                        <div id="transcription" class="mt-8 space-y-2">
                            {transcription_html}
                        </div>
                    </div>
                </div>
            </div>

            <script>
            {{
                document.getElementById('transcribe-form').addEventListener('submit', function(e) {{
                    const btn = document.getElementById('submit-btn');
                    const spinner = btn.querySelector('.spinner');
                    const btnText = btn.querySelector('span');
                    
                    btn.disabled = true;
                    spinner.classList.remove('hidden');
                    btnText.textContent = 'Transcribing...';
                }});
            }}
            </script>
        </body>
    </html>
    """

@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe(request: Request):
    try:
        form = await request.form()
        url = form.get("url")
        if not url:
            return await home(error="No URL provided")
        
        try:
            audio_file = download_youtube_audio(url)
        except Exception as e:
            return await home(error=f"Failed to download audio: {str(e)}")
        
        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_file)
            os.remove(audio_file)
            return await home(transcription=result)
            
        except Exception as e:
            return await home(error=f"Failed to transcribe audio: {str(e)}")
            
    except Exception as e:
        return await home(error=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000) 