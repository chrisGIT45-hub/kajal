# app.py: Professional YouTube Q&A Bot with RAG, External Sources, and Fallbacks
# Run: streamlit run app.py

import re
import os
import subprocess
import streamlit as st
# No longer need load_dotenv for deployment
# from dotenv import load_dotenv
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

# --- FIX 1: USE STREAMLIT SECRETS FOR API KEYS ---
# Load secrets from Streamlit Cloud's secret management
openai.api_key = st.secrets.get("OPENAI_API_KEY")
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY")

USE_OPENAI = bool(openai.api_key)

# Config: LLM and Wikipedia
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai.api_key) if USE_OPENAI else None

# --- FIX 2: ADD USER-AGENT TO WIKIPEDIA API CALL ---
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent="YouTubeQABot/1.0 (contact@example.com)",
    language='en'
)

st.set_page_config(page_title="YouTube Q&A Bot", page_icon="🎥", layout="wide")
st.title("🎥 YouTube Q&A Bot")

# Sidebar for config
st.sidebar.header("Config")
use_external = st.sidebar.checkbox("Use External Sources (Wikipedia/SerpAPI)", value=True)
eval_mode = st.sidebar.checkbox("Research Eval Mode (Log to CSV)")
if not USE_OPENAI:
    st.sidebar.warning("No OpenAI key found in secrets. Using free fallbacks (HuggingFace + local Whisper).")

# Helper functions
@st.cache_resource
def download_audio(url: str) -> str:
    """Fallback: Download audio if no captions."""
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        temp_fn = stream.download(output_path='temp', filename='audio')
        mp3_path = temp_fn.rsplit('.', 1)[0] + '.mp3'
        subprocess.run(["ffmpeg", "-y", "-i", temp_fn, "-vn", "-acodec", "libmp3lame", mp3_path],
                       check=True, capture_output=True)
        os.remove(temp_fn)
        return mp3_path
    except Exception as e:
        st.error(f"Audio download error: {e}")
        return None

def transcribe_audio(audio_path: str) -> str:
    """Fallback transcription: OpenAI or local Whisper."""
    try:
        if USE_OPENAI:
            with open(audio_path, "rb") as f:
                transcript = openai.Audio.transcribe("whisper-1", f)
            text = transcript["text"]
        else:
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            text = result["text"]
        # Clean fillers
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

def summarize_context(docs: List[Document]) -> str:
    """Summarize retrieved docs."""
    if not USE_OPENAI or not docs:
        return "Summarization N/A."
    # The llm object is already configured with the API key
    context_text = "\n".join([d.page_content for d in docs])[:2000]
    prompt = f"Summarize the key facts in the following text in 2-3 sentences:\n{context_text}"
    response = llm.invoke(prompt)
    return response.content


# Main app logic
url = st.text_input("Paste a YouTube URL:")
if url:
    st.info("Fetching transcript...")
    try:
        yt = YouTube(url)
        # Try captions first
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
                "start_time": f"{(i * 10) // 60:02d}:{(i * 10) % 60:02d}" # Simple time estimate
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
            If unsure, say "I don't know based on the provided sources."
            Cite sources like this: [Transcript: Chunk {chunk_index}] or [Wikipedia: {title}].

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
                    result = qa.invoke({"query": question})
                    answer = result["result"]

                    # Get sources, summarize, confidence
                    transcript_docs = result["source_documents"]
                    external_docs = get_external_docs(question) if use_external else []
                    all_docs = transcript_docs + external_docs
                    summary = summarize_context(all_docs) if USE_OPENAI else "N/A (no OpenAI)"
                    confidence = "High" if len(transcript_docs) >= 2 else "Medium"

                    # Display results
                    st.markdown(f"**Answer:** {answer}")
                    st.markdown(f"**Context Summary:** {summary}")
                    st.markdown(f"**Confidence:** {confidence}")

                    # Sources display
                    st.subheader("Sources")
                    sources = [{"content": d.page_content[:200], **d.metadata} for d in all_docs]
                    for i, src in enumerate(sources, 1):
                        with st.expander(f"Source {i}: {src.get('source', 'unknown').title()}"):
                            st.write(f"Metadata: {src}")
                            st.write(f"Snippet: {src['content']}...")

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
                        # This part will not work on Streamlit Cloud's ephemeral filesystem
                        # For persistent logging, you would need to use a database or other service.
                        # df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
                        # st.success(f"Logged to {csv_path}")

            else:
                st.warning("OpenAI not available. Please add API keys to your Streamlit Cloud secrets to ask questions.")

    except Exception as e:
        st.error(f"Error in captions or processing: {e}")
        st.info("Trying audio transcription fallback...")
        audio_path = download_audio(url)
        if audio_path:
            transcript_text = transcribe_audio(audio_path)
            if transcript_text:
                # This fallback logic would need to be fleshed out similar to the main logic above
                st.success("Fallback transcript processed. The full Q&A pipeline for fallback is not yet implemented in this script.")
                st.text_area("Fallback Transcript", transcript_text, height=200)
            else:
                st.error("Fallback transcription failed.")
        else:
            st.error("Audio download failed. Ensure FFmpeg is installed in your deployment environment.")

st.markdown("---")
st.markdown("Built with LangChain RAG for precise, cited answers. For research: Toggle external sources to compare accuracy.")
