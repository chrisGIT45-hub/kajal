import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import yt_dlp
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from graphviz import Digraph

# ========== CONFIGURATION ==========
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key in env variable

# ========== FUNCTIONS ==========

def get_video_id(youtube_url):
    """Extract video ID from YouTube URL"""
    if "v=" in youtube_url:
        return youtube_url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[-1].split("?")[0]
    else:
        return None

def get_transcript(video_id):
    """Try to get transcript using youtube-transcript-api"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([t['text'] for t in transcript_list])
        return transcript
    except Exception as e:
        st.warning(f"Transcript not available via API: {e}")
        return None

def download_audio(youtube_url):
    """Download audio from YouTube using yt-dlp"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return "audio.mp3"

def transcribe_audio_whisper(audio_file):
    """Transcribe audio file using OpenAI Whisper API"""
    with open(audio_file, "rb") as audio:
        transcript = openai.Audio.transcribe("whisper-1", audio)
    return transcript['text']

def create_vector_store(text):
    """Split transcript and create FAISS vector store with OpenAI embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def build_qa_chain(vector_store):
    """Build RetrievalQA chain with OpenAI LLM"""
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    return qa_chain

def answer_question(qa_chain, question):
    """Get answer from QA chain"""
    return qa_chain.run(question)

def summarize_text(text):
    """Summarize transcript and extract key points"""
    prompt = f"Summarize the following text and extract key points:\n\n{text}"
    llm = OpenAI(temperature=0)
    summary = llm(prompt)
    return summary

def generate_flow_diagram():
    """Generate and save flow diagram of the pipeline"""
    dot = Digraph(comment='YouTube Video Q&A Bot Flow')
    dot.node('A', 'YouTube Video URL')
    dot.node('B', 'Download Audio')
    dot.node('C', 'Transcribe Audio (Whisper)')
    dot.node('D', 'Split & Embed Transcript')
    dot.node('E', 'Store in Vector DB (FAISS)')
    dot.node('F', 'User  Question')
    dot.node('G', 'Retrieve Relevant Chunks')
    dot.node('H', 'Generate Answer (LLM)')
    dot.node('I', 'Display Answer')

    dot.edges(['AB', 'BC', 'CD', 'DE', 'FG', 'GH', 'HI'])
    dot.edge('F', 'G', constraint='false')
    filename = "ytqa_flow"
    dot.render(filename, format='png', cleanup=True)
    return filename + ".png"

# ========== STREAMLIT APP ==========

def main():
    st.title("🎥 YouTube Video Q&A Bot")

    youtube_url = st.text_input("Enter YouTube Video URL")
    if youtube_url:
        video_id = get_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return

        with st.spinner("Fetching transcript..."):
            transcript = get_transcript(video_id)

        if not transcript:
            st.info("Transcript not found, downloading and transcribing audio (this may take a few minutes)...")
            try:
                audio_file = download_audio(youtube_url)
                transcript = transcribe_audio_whisper(audio_file)
                os.remove(audio_file)  # Clean up audio file
            except Exception as e:
                st.error(f"Error during audio download or transcription: {e}")
                return

        st.subheader("Transcript")
        st.text_area("Transcript", transcript, height=200)

        with st.spinner("Creating vector store and QA chain..."):
            vector_store = create_vector_store(transcript)
            qa_chain = build_qa_chain(vector_store)

        question = st.text_input("Ask a question about the video:")
        if question:
            with st.spinner("Generating answer..."):
                answer = answer_question(qa_chain, question)
            st.markdown(f"**Answer:** {answer}")

        if st.button("Summarize Video"):
            with st.spinner("Generating summary..."):
                summary = summarize_text(transcript)
            st.subheader("Summary & Key Points")
            st.write(summary)

        if st.button("Show Flow Diagram"):
            diagram_path = generate_flow_diagram()
            st.image(diagram_path, caption="YouTube Video Q&A Bot Architecture Flow")

if __name__ == "__main__":
    main()



