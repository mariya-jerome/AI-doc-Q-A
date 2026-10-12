# AI-Powered Document Q&A Assistant

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application. Upload one PDF, ask questions about it, and get answers grounded in retrieved passages from that document.

The app extracts PDF text locally, splits it into overlapping chunks, embeds those chunks with a Hugging Face Sentence Transformers model on your machine, stores the vectors in a FAISS CPU index, and calls the **Groq API** only after retrieval. The language model sees the relevant chunks, not the entire PDF.

## Architecture and data flow

```text
PDF upload
    -> validate file type and size
    -> PyPDF extracts text page by page (page numbers kept)
    -> overlapping character chunks
    -> local Sentence Transformers embeddings (normalized vectors)
    -> FAISS IndexFlatIP (inner product = cosine similarity for unit vectors)
User question
    -> same embedding model
    -> top-k similar chunks
    -> grounded prompt (system rules + retrieved context + question)
    -> Groq chat completion
    -> answer + source page numbers in Streamlit
```

Document embeddings are built once per processed PDF and kept in the Streamlit session. Replacing the file resets the index. Asking another question does **not** re-embed the document.

## Features

- One-PDF-at-a-time upload with type, size, and empty-file checks
- Page-aware text extraction and overlapping chunking
- Local embeddings (`sentence-transformers/all-MiniLM-L6-v2` by default)
- FAISS similarity search with configurable `TOP_K` (default 4)
- Groq LLM answers using only retrieved context
- Source page citations for retrieved chunks
- User-friendly errors for missing keys, invalid PDFs, and API failures
- Unit tests that run without a real Groq API key

## Technology stack

| Piece | Library | Role |
| --- | --- | --- |
| Interface | Streamlit | Upload, process, ask, display answers |
| PDF text | PyPDF | Extract text and page numbers |
| Embeddings | Sentence Transformers | Local semantic vectors |
| Vector index | FAISS CPU | Nearest-neighbor search |
| LLM | Groq API | Generate the final answer |
| Config | python-dotenv | Load `.env` values |
| Vectors | NumPy | Array handling |

## Beginner-friendly concepts

1. **Generative AI** creates new text (or other media) from patterns learned during training. This app uses it to write answers in natural language.
2. **An LLM (large language model)** is a neural network trained to predict and generate text. It can write fluent answers, but it does not automatically "look up" your PDF unless you give it the right passages.
3. **An LLM API** lets your program send a prompt to a hosted model and receive generated text. This project uses Groq's chat completions API.
4. **Embeddings** turn text into a list of numbers. Texts with similar meaning have similar vectors. This project generates embeddings locally; document text is not sent to an embedding API.
5. **A vector database or vector index** stores those lists of numbers so you can search for "nearest" vectors quickly.
6. **FAISS** (Facebook AI Similarity Search) is a library for fast vector search. This app uses the CPU build and an in-memory index, which is enough for a single-document portfolio project.
7. **RAG** means: retrieve relevant document pieces first, then generate an answer from those pieces. Retrieval reduces the chance that the model ignores your PDF or invents details.
8. **Text chunking** is necessary because a PDF can be long. Smaller overlapping chunks make search more precise and keep the LLM prompt within a reasonable size.
9. **Similarity search** embeds the question and finds chunk vectors with the highest cosine similarity. With **normalized** embeddings, FAISS inner product is equivalent to cosine similarity.
10. **Retrieved context is sent to the LLM** so the model can quote and reason over the actual document wording instead of guessing from general training data.
11. **Prompt engineering** here means strict system rules: answer only from context, treat the PDF as untrusted data, and say when the answer is missing. That lowers unsupported answers, but it does not make the model perfect.
12. **Limitations** are listed at the end of this file. RAG can still retrieve the wrong passage or the model can still misread a passage.

## Project structure

```text
AI-Doc-QA/
├── app.py
├── config.py
├── document_processor.py
├── embeddings.py
├── vector_store.py
├── rag_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── tests/
    ├── test_chunking.py
    ├── test_document_processing.py
    └── test_prompt.py
```

## Installation

Use **Python 3.10 or newer** (the project was written for 3.12).

### 1. Create a virtual environment

**Windows (PowerShell):**

```powershell
cd C:\Users\mariy\AI-Doc-QA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activate script, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**macOS / Linux:**

```bash
cd /path/to/AI-Doc-QA
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run of the embedding model downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face (about 90 MB). After that, it is cached locally.

## Environment configuration

1. Copy the example file:

   ```powershell
   copy .env.example .env
   ```

2. Open `.env` and set:

   - `GROQ_API_KEY` — your secret key from the [Groq Console](https://console.groq.com/)
   - `GROQ_MODEL` — a currently supported Groq chat model, for example `llama-3.1-8b-instant`
   - `EMBEDDING_MODEL` — default `sentence-transformers/all-MiniLM-L6-v2`
   - `CHUNK_SIZE` — characters per chunk (default `500`)
   - `CHUNK_OVERLAP` — overlapping characters (default `100`)
   - `TOP_K` — number of chunks to retrieve (default `4`, typically 3–5)

Get an API key: create a Groq account, open [API Keys](https://console.groq.com/keys), and create a key. Check live model IDs at [Groq supported models](https://console.groq.com/docs/models). Model names change over time; if generation fails, update `GROQ_MODEL`.

Never put a real key in source files or commit `.env`.

## How to run the application

With the virtual environment active and `.env` configured:

```bash
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in a browser.

## How to test with a sample PDF

1. Create a short PDF that includes a line such as:
   `Students must maintain at least 75% attendance to be eligible for examinations.`
2. Click **Browse files** and upload that PDF.
3. Click **Process Document**. Wait until you see a success message and a chunk count.
4. Type a question and click **Ask Question**.

### Example questions and expected outputs

**In-document question**

- Question: `What is the minimum attendance required for examinations?`
- Expected answer: a statement that students need **75%** attendance to be eligible for examinations.
- Sources: the page that contains that sentence (often page 1).

**Out-of-document question**

- Question: `Who won the 2018 FIFA World Cup?`
- Expected answer: that the answer was **not found in the uploaded document**, because the retrieved context does not contain that fact.

You can ask multiple questions about the same processed PDF without re-uploading it. Upload a different PDF to rebuild the index.

## Testing

Unit tests mock the Groq API and use a lightweight hashing embedder. They do **not** require `GROQ_API_KEY`.

```bash
python -m pytest tests -q
```

These tests cover PDF extraction, empty PDFs, chunk overlap, page numbers, embedding shapes, FAISS retrieval, prompt construction, empty questions, missing API keys, mocked API failures, and a question that is absent from the retrieved context.

They are **unit tests**, not a full browser end-to-end test. An end-to-end check still needs a real Groq key, `streamlit run app.py`, and a sample PDF.

## Troubleshooting

| Problem | What to try |
| --- | --- |
| Missing API key | Copy `.env.example` to `.env` and set `GROQ_API_KEY`. Restart Streamlit. |
| Invalid API key | Create a new key in the Groq Console. Do not add quotes or spaces around the key. |
| Model unavailable | Open Groq's model list and set `GROQ_MODEL` to a current chat model. |
| Rate limit | Wait and retry. Reduce how often you click **Ask Question**. |
| Network failure | Check internet access. Groq is required for answers; embeddings run locally. |
| "Not a valid PDF" | Upload a real `.pdf` file, not a renamed image or Word document. |
| Password-protected PDF | Export or print an unprotected copy, then upload that file. |
| No extractable text | The file may be a scan. This app does not run OCR. |
| Embedding download fails | Connect to the internet on first run so Hugging Face can cache the model. |
| Import errors | Activate `.venv` and run `pip install -r requirements.txt` again. |

## Project limitations

- Only one PDF is indexed at a time, in memory, for the current Streamlit session.
- Image-only or scanned PDFs without a text layer cannot be read.
- Password-protected PDFs are rejected.
- Chunking is character-based, so a chunk may start mid-sentence.
- Retrieval can return the wrong pages if the question is vague or the PDF is noisy.
- Source pages mean "these pages were retrieved as context," not "every sentence in the answer is verified."
- The Groq model can still make mistakes even with a grounded prompt.
- There is no user login, persistence, or multi-user document store.

This project is intended as a clear portfolio demonstration of a working RAG pipeline, not as a production document platform.
