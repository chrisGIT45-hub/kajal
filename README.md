# YouTube Video Q&A Bot

A Streamlit app for Q&A on YouTube videos using RAG (Retrieval-Augmented Generation). Extracts transcripts (captions or Whisper transcription), chunks and embeds them with OpenAI, stores in FAISS, retrieves relevant parts, augments with Wikipedia, and generates answers with OpenAI LLM.

## Setup

1. Clone or create the project directory with the files above.
2. Create a virtual environment:


3. Install dependencies:

4. Install system dependencies:
- **ffmpeg**: Required for audio processing.
  - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Use Chocolatey (`choco install ffmpeg`) or download from https://ffmpeg.org/
5. Create `.env` file with your OpenAI API key (required for embeddings/LLM).
6. Run the app:


Open http://localhost:8501 in your browser.

## Usage

1. Paste a YouTube URL (e.g., a tutorial video).
2. Click "Process Video" – fetches transcript or transcribes audio.
3. Ask questions in the input box and click "Ask".
4. View answer and sources (transcript chunks + Wikipedia).

## Features

- **Transcript Extraction**: Prefers YouTube captions; falls back to audio download + local Whisper.
- **RAG Pipeline**: LangChain for chunking, OpenAI embeddings, FAISS vector store.
- **External Knowledge**: Wikipedia search for additional context.
- **Citations**: Sources shown with metadata.
- **UI**: Simple Streamlit interface with previews and expanders.

## Evaluation & Experiments

- Test with 5-10 videos: Measure answer accuracy manually (e.g., correct/incorrect).
- Compare: Transcript-only vs. +Wikipedia (add toggle in `qa_chain.py`).
- Metrics: Log query time, token usage (via OpenAI logs).
- Sample Dataset: Use educational videos like TED talks; create 5 Q&A pairs per video.

## Limitations & Future Work

- **Limitations**: Local Whisper is CPU-slow for long videos (>30min); no video timestamps in basic metadata; Wikipedia may not cover niche topics.
- **Ethics**: App disclaims advice; tracks sources to reduce hallucinations.
- **Future**: Add SerpAPI for web search, arXiv integration, hierarchical indexing for long videos, human eval dataset.

## Troubleshooting

- **No captions/transcription fails**: Ensure video has English audio; check ffmpeg.
- **OpenAI errors**: Verify API key and credits.
- **FAISS issues on Windows**: Use `chromadb` alternative (swap in `indexer.py`).
- **Slow performance**: Use GPU for Whisper; cache indices.

Demo Video: [Link to your recorded screen capture here].

For questions, see the project guide in your notes.
