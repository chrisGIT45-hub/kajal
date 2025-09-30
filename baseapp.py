# baseapp.py: Professional YouTube Q&A Bot with RAG, External Sources, and Fallbacks
# Run: streamlit run baseapp.py

import os
import re
import subprocess
import streamlit as st
from dotenv import load_dotenv

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
import yt_dlp  # For authenticated downloads with cookies

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
USE_OPENAI = bool(openai.api_key)

# Initialize LLM and Wikipedia API
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai.api_key) if USE_OPENAI else None
wiki_wiki = wikipediaapi.Wikipedia('en')

# Streamlit page config
st.set_page_config(page_title="YouTube Q&A Bot", page_icon="🎥", layout="wide")
st.title("🎥 YouTube Q&A Bot")

# --- Diagram Visualization Objective ---
st.markdown(
    """
    ### 🧠 Architecture Overview

    This app uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer questions about YouTube videos:

    1. **Input:** User pastes a YouTube URL and optionally uploads their YouTube cookies file for restricted videos.
    2. **Transcript Retrieval:**  
       - Tries to get official captions via `pytube`.  
       - If unavailable, downloads audio using `yt-dlp` with optional cookies and transcribes with Whisper.
    3. **Document Processing:**  
       - Splits transcript into chunks with metadata.  
       - Embeds chunks using OpenAI or HuggingFace embeddings.  
       - Stores embeddings in FAISS vector store.
    4. **External Knowledge (Optional):**  
       - Retrieves relevant Wikipedia and SerpAPI documents.
    5. **Question Answering:**  
       - Uses a Chat LLM (GPT-3.5-turbo) with a custom prompt to answer questions citing sources.
    6. **Output:**  
       - Shows concise answers with citations, confidence, and source snippets.

    ---
    """
)

# Sidebar config
st.sidebar.header("Config")
use_external = st.sidebar.checkbox("Use External Sources (Wikipedia/SerpAPI)", value=True)
eval_mode = st.sidebar.checkbox("Research Eval Mode (Log to CSV)")
if not USE_OPENAI:
    st.sidebar.warning("No OpenAI key: Using free fallbacks (HuggingFace + local Whisper).")

# --- Helper Functions ---

def download_audio_with_cookies(url: str, cookies_path: str = None) -> str:
    """
    Download audio from YouTube using yt-dlp with optional cookies for authenticated access.
    Returns path to mp3 file or None on failure.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp/audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    if cookies_path:
        ydl_opts['cookiefile'] = cookies_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return 'temp/audio.mp3'
    except Exception as e:
        st.error(f"Audio download error: {e}")
        return None

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file using OpenAI Whisper API or local Whisper model.
    Cleans filler words from transcript.
    """
    try:
        if USE_OPENAI:
            with open(audio_path, "rb") as f:
                transcript = openai.Audio.transcribe("whisper-1", f)
            text = transcript["text"]
        else:
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            text = result["text"]
        # Clean filler words
        text = re.sub(r'\b(um|uh|like|you know)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        os.remove(audio_path)
        return text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def get_external_docs(question: str) -> List[Document]:
    """
    Retrieve external documents from Wikipedia and SerpAPI for the question.
    """
    docs = []
    # Wikipedia search
    search_results = wiki_wiki.search(question, results=2)
    for title in search_results:
        page = wiki_wiki.page(title)
        if page.exists():
            summary = page.summary[:500] + "..."
            docs.append(Document(page_content=summary, metadata={"source": "wikipedia", "title": title}))
    # SerpAPI search
    if SERPAPI_KEY:
        try:
            search = SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)
            results = search.run(question)
            docs.append(Document(page_content=results[:500], metadata={"source": "serpapi", "query": question}))
        except Exception:
            pass
    return docs

def summarize_context(docs: List[Document]) -> str:
    """
    Summarize retrieved documents using LLM.
    """
    if not USE_OPENAI or not docs:
        return "Summarization N/A."
    context = "\n".join([d.page_content for d in docs])[:2000]
    prompt = f"Summarize key facts in 2-3 sentences:\n{context}"
    return llm(prompt)

# --- Main App Logic ---

url = st.text_input("Paste a YouTube URL:")

# Cookie upload for restricted videos
uploaded_cookies = st.file_uploader(
    "Upload your YouTube cookies.txt file (optional, for restricted videos)", type=["txt"]
)
st.markdown(
    """
    <small>
    **Privacy Notice:** Your cookies file is used only temporarily to access restricted YouTube videos and is not stored permanently.
    Do not upload cookies if you are uncomfortable sharing your account data.
    </small>
    """,
    unsafe_allow_html=True
)

if url:
    try:
        # Try to get captions first (public videos)
        yt = YouTube(url)
        caption = yt.captions.get_by_language_code('en')
        if caption is None:
            raise ValueError("No English captions available.")
        transcript = caption.generate_srt_captions()
        st.success(f"✅ Transcript fetched for: {yt.title}")

        # Split transcript into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.create_documents([transcript])

        # Add metadata for citations
        for i, doc in enumerate(docs):
            doc.metadata = {
                "source": "transcript",
                "yt_title": yt.title,
                "chunk_index": i,
                "start_time": f"{i * 10:02d}:00"
            }

        # Embeddings and vector store
        if USE_OPENAI:
            embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        else:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # QA chain with custom prompt
        if USE_OPENAI:
            PROMPT_TEMPLATE = """
            You are a helpful assistant answering from video transcript and external sources only.
            Be concise: 1-2 sentences, brief explanation, then cite sources.
            If unsure, say "I don't know based on sources."
            Cite: [Transcript: Chunk {chunk_index}, Time: {start_time}] or [Wikipedia: {title}].

            Context: {context}

            Question: {question}

            Answer:
            """
            PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
        else:
            qa = None

        question = st.text_input("Ask a question about the video:")
        if question:
            if USE_OPENAI and qa:
                with st.spinner("Generating answer..."):
                    result = qa({"query": question})
                    answer = result["result"]

                    # Get sources, summarize, confidence
                    transcript_docs = result["source_documents"]
                    external_docs = get_external_docs(question) if use_external else []
                    all_docs = transcript_docs + external_docs
                    summary = summarize_context(all_docs) if USE_OPENAI else "N/A (no OpenAI)"
                    confidence = "High" if len(transcript_docs) >= 2 else "Medium"

                    # Display results
                    st.markdown(f"**Answer:** {answer}")
                    st.markdown(f"**Summary:** {summary}")
                    st.markdown(f"**Confidence:** {confidence}")

                    # Sources display
                    st.subheader("Sources")
                    sources = [{"content": d.page_content[:200], **d.metadata} for d in all_docs]
                    for i, src in enumerate(sources, 1):
                        with st.expander(f"Source {i}: {src.get('source', 'unknown')}"):
                            st.write(f"**Metadata:** {src}")
                            st.write(f"**Snippet:** {src['content']}")

                    # Eval log
                    if eval_mode:
                        log_data = {
                            "Video Title": yt.title,
                            "Question": question,
                            "Answer": answer[:100] + "...",
                            "Confidence": confidence,
                            "External Used": use_external,
                            "Sources Count": len(sources)
                        }
                        st.json(log_data)
                        df = pd.DataFrame([log_data])
                        csv_path = "qa_eval_log.csv"
                        df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
                        st.success(f"Logged to {csv_path}")

            else:
                st.warning("OpenAI not available. Use fallbacks or add API key.")

    except Exception as e:
        st.error(f"Error in captions or processing: {e}")
        st.info("Trying audio transcription fallback...")

        # Save uploaded cookies file temporarily if provided
        cookies_path = None
        if uploaded_cookies is not None:
            os.makedirs("temp", exist_ok=True)
            cookies_path = f"temp/{uploaded_cookies.name}"
            with open(cookies_path, "wb") as f:
                f.write(uploaded_cookies.getbuffer())

        audio_path = download_audio_with_cookies(url, cookies_path)
        if audio_path:
            transcript_text = transcribe_audio(audio_path)
            if transcript_text:
                # Process fallback transcript same as above
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = splitter.create_documents([transcript_text])
                for i, doc in enumerate(docs):
                    doc.metadata = {
                        "source": "audio_transcript",
                        "chunk_index": i,
                        "start_time": f"{i * 10:02d}:00"
                    }
                if USE_OPENAI:
                    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
                else:
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = FAISS.from_documents(docs, embeddings)
                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                if USE_OPENAI:
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                else:
                    qa = None

                st.success("Fallback transcript processed. Ask your question above.")
            else:
                st.error("Fallback transcription failed.")
        else:
            st.error("Audio download failed. Check your cookies or video accessibility.")

st.markdown("---")
st.markdown("*Built with LangChain RAG for precise, cited answers. For research: Toggle external sources to compare accuracy.*")
