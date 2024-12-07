import React, { useState } from 'react';
import { FiLink, FiFile, FiMic } from 'react-icons/fi';

function App() {
  const [transcribing, setTranscribing] = useState(false);
  const [transcripts, setTranscripts] = useState([]);
  const [mediaUrl, setMediaUrl] = useState('');

  const handleTranscribe = async (url) => {
    setTranscribing(true);
    try {
      const response = await fetch('/api/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      // Assuming server sends Server-Sent Events for real-time transcripts
      const reader = response.body.getReader();
      // Process chunks as they arrive
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const transcript = new TextDecoder().decode(value);
        setTranscripts(prev => [...prev, JSON.parse(transcript)]);
      }
    } catch (error) {
      console.error('Transcription error:', error);
    } finally {
      setTranscribing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-5xl font-bold text-center text-gray-900 mb-2">Blayze AI</h1>
        <p className="text-center text-gray-600 mb-8">
          ML-powered speech recognition directly in your browser
        </p>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex gap-4 justify-center mb-8">
            <button className="flex items-center gap-2 px-4 py-2 rounded-md hover:bg-gray-100">
              <FiLink /> From URL
            </button>
            <button className="flex items-center gap-2 px-4 py-2 rounded-md hover:bg-gray-100">
              <FiFile /> From file
            </button>
            <button className="flex items-center gap-2 px-4 py-2 rounded-md hover:bg-gray-100">
              <FiMic /> Record
            </button>
          </div>

          {transcribing && (
            <div className="flex justify-center mb-8">
              <div className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-full">
                <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                Transcribing...
              </div>
            </div>
          )}

          <div className="space-y-4">
            {transcripts.map((transcript, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded-lg">
                <span className="text-gray-500 mr-2">{transcript.timestamp}</span>
                <span>{transcript.text}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 