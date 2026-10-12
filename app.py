import streamlit as st
import config
import requests
from document_processor import extract_text_from_pdf, create_chunks
from embeddings import generate_embeddings
from vector_store import VectorStore
from rag_pipeline import generate_answer

st.set_page_config(page_title="AI Document Q&A Assistant", layout="centered")

def init_session_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "doc_processed" not in st.session_state:
        st.session_state.doc_processed = False

init_session_state()

# Sidebar for API Key & Settings
st.sidebar.header("🔑 API Settings")
user_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=config.GROQ_API_KEY if config.GROQ_API_KEY and config.GROQ_API_KEY != "your_groq_api_key_here" else "",
    help="Enter your Groq API key here or add GROQ_API_KEY in your .env file."
)

if not user_api_key:
    st.sidebar.warning("⚠️ Please enter your Groq API Key above to run Q&A queries.")
    st.sidebar.markdown("[Get a free Groq API key](https://console.groq.com/keys)")
else:
    st.sidebar.success("✅ API Key provided")

st.sidebar.markdown("---")

# Dynamically fetch available models from the Groq API
@st.cache_data(show_spinner=False, ttl=300)
def fetch_groq_models(api_key: str):
    """Fetch the list of available model IDs from Groq using the provided API key."""
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            # Filter to chat-capable models (exclude embedding / guard models)
            ids = sorted(
                m["id"] for m in data
                if not any(x in m["id"] for x in ["whisper", "guard", "tts", "embed"])
            )
            return ids if ids else None
    except Exception:
        pass
    return None

# Fallback list used when no API key is entered
FALLBACK_MODELS = ["llama3-8b-8192", "llama3-70b-8192", "gemma2-9b-it"]

if user_api_key:
    with st.sidebar:
        with st.spinner("Loading models…"):
            available_models = fetch_groq_models(user_api_key) or FALLBACK_MODELS
else:
    available_models = FALLBACK_MODELS

selected_model = st.sidebar.selectbox(
    "Groq LLM Model",
    options=available_models,
    index=0,
    help="Models fetched live from your Groq account."
)

st.sidebar.markdown("**Embedding Model:** " + config.EMBEDDING_MODEL)

# Main Page Layout
st.title("AI-Powered Document Q&A Assistant")
st.markdown("Upload a PDF document and ask questions about its contents. Uses Retrieval-Augmented Generation (RAG) with local embeddings and the Groq LLM API.")

with st.expander("How the RAG workflow works"):
    st.markdown("""
        **Retrieval-Augmented Generation (RAG)** process:
        1. **Extract**: Text is extracted from the uploaded PDF page by page.
        2. **Chunk**: The text is split into smaller overlapping chunks.
        3. **Embed**: We generate vector embeddings for each chunk using a local model.
        4. **Index**: The embeddings are stored in a FAISS vector database for fast similarity search.
        5. **Retrieve**: When you ask a question, we convert it to an embedding and find the most relevant chunks.
        6. **Generate**: The retrieved chunks and your question are sent to the Groq LLM to generate an accurate answer based *only* on the document context.
    """)

st.header("1. Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write(f"**Filename:** {uploaded_file.name}")
    
    # Process button
    if st.button("Process Document"):
        st.session_state.doc_processed = False
        st.session_state.vector_store = None
        
        with st.spinner("Processing document... (Extracting text, chunking, and generating embeddings)"):
            try:
                # Extract text
                pages_data = extract_text_from_pdf(uploaded_file)
                
                # Chunk text
                chunks = create_chunks(pages_data, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                
                # Generate embeddings
                texts = [chunk["text"] for chunk in chunks]
                embeddings = generate_embeddings(texts)
                
                # Store in Vector Store
                vs = VectorStore()
                vs.build_index(embeddings, chunks)
                
                st.session_state.vector_store = vs
                st.session_state.doc_processed = True
                
                st.success("Document processed successfully! You can now ask questions.")
            except Exception as e:
                st.error(f"Error processing document: {e}")
                
if st.session_state.doc_processed:
    st.header("2. Ask Questions")
    
    user_question = st.text_input("Ask a question about the document:")
    
    if st.button("Ask Question"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        elif not user_api_key:
            st.error("Please enter your Groq API Key in the sidebar before asking a question.")
        else:
            with st.spinner("Searching for answers..."):
                try:
                    answer, source_pages = generate_answer(
                        user_question,
                        st.session_state.vector_store,
                        api_key=user_api_key,
                        model_name=selected_model
                    )
                    
                    st.subheader("Answer:")
                    st.write(answer)
                    
                    st.subheader("Sources:")
                    if source_pages:
                        pages_str = ", ".join([str(p) for p in source_pages])
                        st.write(f"Pages: {pages_str}")
                    else:
                        st.write("No specific pages found.")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    
elif uploaded_file is not None and not st.session_state.doc_processed:
    st.info("Please click 'Process Document' to enable questions.")
