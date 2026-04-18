# Multi-Modal Document Intelligence

A simple Streamlit app for PDF question answering using multimodal retrieval and grounded answers.

## What It Does

- Upload a PDF and index it page by page.
- Retrieve relevant pages using a ColPali-family model.
- Generate answers grounded in the indexed document.
- Use OpenAI for answer generation when an API key is available.

## Tech Stack

- Streamlit for the interface.
- ColSmol / ColPali-family embeddings for retrieval.
- Qdrant for vector storage.
- PyMuPDF and pdf2image for PDF processing.
- OpenAI for answer generation.

## How It Works

1. The PDF is converted into page images and structured text.
2. Each page is embedded and stored in Qdrant.
3. A user asks a question.
4. The system retrieves the most relevant pages.
5. The app returns a grounded answer based on those results.

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Environment Variables

Create a `.env` file if needed:

```env
OPENAI_API_KEY=sk-REPLACE_ME
OPENAI_MODEL=gpt-4o-mini
COLPALI_MODEL_NAME=vidore/colSmol-256M
COLPALI_DEVICE=cpu
COLPALI_POOL_FACTOR=2
```

If `OPENAI_API_KEY` is set, the app uses OpenAI for answer generation. Otherwise, it uses the local fallback answer path.

## Main Files

- `app.py` - Streamlit interface.
- `pipeline.py` - End-to-end workflow.
- `ingestion.py` - PDF loading and extraction.
- `embedder.py` - Retrieval model wrapper.
- `vector_store.py` - Qdrant search and storage.
- `generator.py` - Answer generation.
- `config.py` - Environment-based settings.

## Notes

- The default model is `vidore/colSmol-256M`.
- First-time startup may be slower because models need to download.
- CPU mode works, but indexing is slower than GPU.
