import os
import requests
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="Local RAG (Llama 3 + Ollama)", page_icon="🦙")
st.title("🦙 Local RAG Chatbot (Llama 3 + Ollama)")
st.caption("Offline-friendly • Streamlit + LangChain + Chroma + Ollama")

# ---------- Helpers ----------
def ollama_ok() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def list_ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        data = r.json()
        return [m.get("name","") for m in data.get("models",[])]
    except Exception:
        return []

def ensure_models(models):
    have = set(list_ollama_models())
    missing = [m for m in models if m not in have]
    return have, missing

def pull_model(model_name: str):
    try:
        code = os.system(f"ollama pull {model_name}")
        return code == 0
    except Exception:
        return False

# ---------- Pre-flight checks ----------
if not ollama_ok():
    st.error(
        "Ollama server is not reachable at http://localhost:11434.\n\n"
        "Please install and start Ollama:\n"
        "  curl -fsSL https://ollama.com/install.sh | sh\n"
        "  ollama serve\n"
        "Then reload this app."
    )
    st.stop()

# Sidebar controls
st.sidebar.header("Settings")
LLM_MODEL = st.sidebar.text_input("LLM model", value="llama3", help="Must exist in `ollama list`")
EMBED_MODEL = st.sidebar.text_input("Embedding model", value="nomic-embed-text", help="Used for vector embeddings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

have, missing = ensure_models([LLM_MODEL, EMBED_MODEL])
if missing:
    st.warning(f"Required model(s) missing in Ollama: {', '.join(missing)}")
    if st.button("Pull missing model(s) now"):
        ok_all = True
        for m in missing:
            st.write(f"Pulling: {m} ...")
            if not pull_model(m):
                ok_all = False
                st.error(f"Failed to pull model: {m}. Try pulling in terminal: `ollama pull {m}`")
        if ok_all:
            st.success("All models pulled. You can proceed.")

uploaded_files = st.file_uploader("Upload PDF(s) to build your knowledge base", type=["pdf"], accept_multiple_files=True)

# Keep state across reruns
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def build_chain(pdfs):
    if not pdfs:
        raise ValueError("No PDFs provided.")
    documents = []
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    for up in pdfs:
        temp_path = os.path.join(temp_dir, up.name)
        with open(temp_path, "wb") as f:
            f.write(up.getbuffer())
        loader = PyPDFLoader(temp_path)
        documents.extend(loader.load())

    if not documents:
        raise ValueError("No text extracted from PDFs. Check that your PDFs are not empty or scanned-only.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Failed to split documents into chunks.")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)  # in-memory use

    llm = ChatOllama(model=LLM_MODEL, temperature=temperature)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        verbose=False
    )
    return chain

col1, col2 = st.columns([1,1])
with col1:
    build_clicked = st.button("📚 Build knowledge base", type="primary")
with col2:
    clear_clicked = st.button("🧹 Clear knowledge & history")

if build_clicked:
    with st.spinner("Indexing PDFs and preparing the chatbot..."):
        try:
            st.session_state.qa_chain = build_chain(uploaded_files)
            st.session_state.chat_history = []
            st.success("Knowledge base ready! Ask a question below.")
        except Exception as e:
            st.error(f"Build failed: {e}")

if clear_clicked:
    st.session_state.qa_chain = None
    st.session_state.chat_history = []
    st.success("Cleared. Re-upload and rebuild to start fresh.")

user_q = st.text_input("Ask a question about your PDFs:")

if user_q and st.session_state.qa_chain is None:
    st.info("Upload PDFs and click **Build knowledge base** first.")

if user_q and st.session_state.qa_chain is not None:
    try:
        result = st.session_state.qa_chain(
            {"question": user_q, "chat_history": st.session_state.chat_history}
        )
        answer = result.get("answer", "").strip()
        sources = result.get("source_documents", [])
        st.session_state.chat_history.append((user_q, answer))

        st.markdown("**Answer:**")
        st.write(answer if answer else "_(no answer generated)_")

        if sources:
            with st.expander("Show sources"):
                for i, doc in enumerate(sources, start=1):
                    src = doc.metadata.get("source","unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"- **Source {i}:** {src} (p. {page})")

        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("**Recent Chat History**")
            for q, a in st.session_state.chat_history[-6:]:
                st.markdown(f"- **Q:** {q}\n  - **A:** {a}")
    except Exception as e:
        st.error(f"Q&A error: {e}")

st.markdown("---")
with st.expander("Setup help (click to expand)"):
    st.markdown(
        """
**Ollama quick start**

```bash
# Install (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start the server (if not auto-started)
ollama serve

# Pull models (once)
ollama pull llama3
ollama pull nomic-embed-text
```

Then run this Streamlit app.
"""
    )
