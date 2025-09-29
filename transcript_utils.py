import yt_dlp
import whisper
import os
from dotenv import load_dotenv
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

load_dotenv()

def get_transcript(video_url):
    """
    Fetch transcript from YouTube video.
    Tries captions first, falls back to Whisper transcription.
    Returns: dict with 'text' (full transcript) and 'segments' (list of timed chunks).
    """
    try:
        # Extract video ID from URL
        ydl_opts = {'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            video_id = info.get('id', '')

        if not video_id:
            return {'error': 'Invalid YouTube URL or video ID not found.'}

        # Try YouTube captions first (fast, no download)
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            full_text = ' '.join([entry['text'] for entry in transcript_list])
            segments = [{'start': entry['start'], 'end': entry['start'] + entry['duration'], 'text': entry['text']} for entry in transcript_list]
            return {'text': full_text, 'segments': segments, 'source': 'captions'}
        except (TranscriptsDisabled, NoTranscriptFound):
            pass  # Fall back to Whisper

        # Fallback: Download audio and transcribe with Whisper
        print("No captions found. Downloading audio for transcription...")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': tempfile.gettempdir() + '/%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            audio_file = ydl.prepare_filename(info).rstrip('.webm').rstrip('.m4a') + '.wav'

        # Transcribe with Whisper
        model = whisper.load_model("base")  # Use "small" or "base" for speed
        result = model.transcribe(audio_file)
        full_text = result['text']
        segments = result['segments']  # Whisper gives timed segments

        # Clean up audio file
        os.remove(audio_file)

        return {'text': full_text, 'segments': segments, 'source': 'whisper'}

    except Exception as e:
        return {'error': f'Transcript fetch failed: {str(e)}'}