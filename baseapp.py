# app.py: Professional YouTube Q&A Bot with RAG, External Sources, and Fallbacks
# Run: streamlit run app.py

import re
import os
import subprocess
import streamlit as st
from pytube import YouTube
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import wikipediaapi
from langchain_community.utilities import SerpAPIWrapper
import whisper
import pandas as pd
from typing import List
import yt_dlp 

# --- CUSTOM THEME DEFINITION ---
def load_css():
    """Inject custom CSS for a 'cooler' theme."""
    css = """
    <style>
        /* Import a cool font from Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Roboto', sans-serif;
        }

        /* Main app background */
        .stApp {
            background-color: #0f1116;
            color: #fafafa;
        }

        /* Main title with a gradient effect */
        h1 {
            background: -webkit-linear-gradient(45deg, #00b4d8, #0077b6, #90e0ef);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        /* Sidebar styling */
        .st-emotion-cache-16txtl3 {
            background-color: #1a1c23;
        }
        
        /* Buttons styling */
        .stButton > button {
            background-color: #00b4d8;
            color: white;
            border-radius: 12px;
            border: none;
            padding: 12px 28px;
            transition: all 0.3s ease-in-out;
            font-weight: 700;
        }
        .stButton > button:hover {
            background-color: #0096b7;
            transform: scale(1.05);
            box-shadow: 0 4px 20px rgba(0, 180, 216, 0.3);
        }

        /* Input box styling */
        .stTextInput, .stTextArea {
            border-radius: 10px;
        }
        .st-emotion-cache-1p5k82d { /* Input box container */
            background-color: #1f2228;
            border-radius: 10px;
        }

        /* Expander (Sources) styling */
        .st-emotion-cache-pwan1w {
            background-color: #1f2228;
            border-radius: 10px;
            border-left: 5px solid #00b4d8;
        }
        
        /* Custom spinner animation */
        .stSpinner > div:first-child {
            border-top-color: #00b4d8;
            border-right-color: transparent;
            border-bottom-color: #00b4d8;
            border-left-color: transparent;
            width: 80px;
            height: 80px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- APP CONFIG AND MAIN CODE ---

# Set page config first
st.set_page_config(page_title="YouTube Q&A Bot", page_icon="🎥", layout="wide")

# Apply the custom theme
load_css()

# Load secrets for Streamlit Cloud deployment
openai.api_key = st.secrets.get("OPENAI_API_KEY")
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY")

USE_OPENAI = bool(openai.api_key)

# Config: LLM and Wikipedia
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai.api_key) if USE_OPENAI else None

# Add user-agent to Wikipedia API call
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent="YouTubeQABot/1.0 (contact@example.com)",
    language='en'
)

st.title("🎥 YouTube Q&A Bot")

# Sidebar for config
st.sidebar.header("Config")
use_external = st.sidebar.checkbox("Use External Sources (Wikipedia/SerpAPI)", value=True)
eval_mode = st.sidebar.checkbox("Research Eval Mode (Log to CSV)")
if not USE_OPENAI:
    st.sidebar.warning("No OpenAI key found in secrets. Using free fallbacks.")

# Helper functions (caching is important for performance)
@st.cache_resource
# Make sure to add this import at the top of your file

# ... (keep all your other imports) ...

@st.cache_resource
def download_audio(url: str) -> str:
    """Download audio from a YouTube URL using yt-dlp and return the mp3 file path."""
    try:
        # Define the output directory and ensure it exists
        output_dir = 'temp'
        os.makedirs(output_dir, exist_ok=True)
        
        # Define yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, 'audio.%(ext)s'), # Save to temp folder
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # The output file will be 'temp/audio.mp3'
        mp3_path = os.path.join(output_dir, 'audio.mp3')
        
        # Check if the file was created
        if not os.path.exists(mp3_path):
            # Sometimes the extension is different, let's find it
            for file in os.listdir(output_dir):
                if file.startswith('audio'):
                    base, ext = os.path.splitext(file)
                    original_path = os.path.join(output_dir, file)
                    os.rename(original_path, mp3_path)
                    break
        
        if os.path.exists(mp3_path):
            return mp3_path
        else:
            raise FileNotFoundError("Audio file not found after download.")

    except Exception as e:
        st.error(f"Audio download error: {e}")
        return None

@st.cache_resource
def transcribe_audio(audio_path: str) -> str:
    """Fallback transcription: OpenAI or local Whisper."""
    try:
        if USE_OPENAI:
            with open(audio_path, "rb") as f:
                transcript = openai.Audio.transcribe("whisper-1", f)
            text = transcript["text"]
        else:
            # Caching the model load can speed things up on subsequent runs
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            text = result["text"]
        text = re.sub(r'\b(um|uh|like|you know)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        os.remove(audio_path)
        return text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def get_external_docs(question: str) -> List[Document]:
    """Retrieve external sources."""
    docs = []
    # Wikipedia
    search_results = wiki_wiki.search(question, results=2)
    for title in search_results:
        page = wiki_wiki.page(title)
        if page.exists():
            summary = page.summary[:500] + "..."
            docs.append(Document(page_content=summary, metadata={"source": "wikipedia", "title": title}))
    # SerpAPI (optional)
    if SERPAPI_KEY:
        try:
            search = SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)
            results = search.run(question)
            docs.append(Document(page_content=results[:500], metadata={"source": "serpapi", "query": question}))
        except Exception:
            pass
    return docs

@st.cache_data
def summarize_context(_docs: List[Document]) -> str:
    """Summarize retrieved docs. Caching this function."""
    if not USE_OPENAI or not _docs:
        return "Summarization N/A."
    context_text = "\n".join([d.page_content for d in _docs])[:2000]
    prompt = f"Summarize the key facts in the following text in 2-3 sentences:\n{context_text}"
    response = llm.invoke(prompt)
    return response.content

# Main app logic
url = st.text_input("Paste a YouTube URL:", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")
if url:
    try:
        yt = YouTube(url)
        with st.spinner("Fetching transcript..."):
            caption = yt.captions.get_by_language_code('en')
            if caption is None:
                raise ValueError("No English captions available. Trying audio fallback.")
            transcript = caption.generate_srt_captions()
        st.success(f"✅ Transcript fetched for: {yt.title}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.create_documents([transcript])

        for i, doc in enumerate(docs):
            doc.metadata = {"source": "transcript", "yt_title": yt.title, "chunk_index": i}

        if USE_OPENAI:
            embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        else:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        PROMPT_TEMPLATE = """
        You are a helpful assistant. Answer from the provided video transcript and external sources only.
        Be concise, then cite your sources like this: [Transcript: Chunk {chunk_index}] or [Wikipedia: {title}].
        If you don't know, say so.

        Context: {context}
        Question: {question}
        Answer:
        """
        PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever,
            return_source_documents=True, chain_type_kwargs={"prompt": PROMPT}
        )
        
        question = st.text_input("Ask a question about the video:")
        if question:
            if qa:
                with st.spinner("Generating answer..."):
                    result = qa.invoke({"query": question})
                    answer = result["result"]
                    
                    transcript_docs = result["source_documents"]
                    external_docs = get_external_docs(question) if use_external else []
                    all_docs = transcript_docs + external_docs
                    summary = summarize_context(all_docs)
                    
                    st.markdown(f"**Answer:** {answer}")
                    st.markdown(f"**Context Summary:** {summary}")
                    
                    st.subheader("Sources")
                    for doc in all_docs:
                        with st.expander(f"Source: {doc.metadata.get('source', 'unknown').title()} - {doc.metadata.get('title', 'Chunk ' + str(doc.metadata.get('chunk_index')))}"):
                            st.write(doc.page_content[:300] + "...")
            else:
                st.warning("OpenAI not available. Please add API keys to your Streamlit Cloud secrets.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Trying audio transcription fallback...")
        with st.spinner("Downloading and transcribing audio... This may take a few minutes."):
            audio_path = download_audio(url)
            if audio_path:
                transcript_text = transcribe_audio(audio_path)
                if transcript_text:
                    st.success("Fallback transcript processed.")
                    st.text_area("Full Transcript (from audio)", transcript_text, height=250)
                    st.warning("Q&A is disabled for fallback transcripts in this version.")
                else:
                    st.error("Fallback transcription failed.")
            else:
                st.error("Audio download failed.")

st.markdown("---")
st.markdown("Built with LangChain & Streamlit. A modern way to interact with video content.")


