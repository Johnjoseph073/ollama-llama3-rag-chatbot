Local RAG Chatbot (Streamlit + LangChain + Chroma + Ollama / Llama 3)
====================================================================

Quick start
-----------
1) Create venv & install deps
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2) Install & start Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve

3) Pull models (first time only)
   ollama pull llama3
   ollama pull nomic-embed-text

4) Run app
   streamlit run app.py

Use
---
- Upload one or more PDFs (e.g., sample_5_lines.pdf included).
- Click "Build knowledge base".
- Ask questions. Works fully offline (no API keys).

Notes
-----
- If the app warns about missing models, use the "Pull missing model(s) now" button
  or run `ollama pull <model>` in your terminal.
