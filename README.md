# 🎓 StudentBuddy AI

> **An Intelligent Student Support Assistant powered by Google Gemini and Retrieval-Augmented Generation (RAG).**

StudentBuddy AI is an AI-powered chatbot designed to assist students with academic queries, placement preparation, internships, career guidance, scholarships, resume building, and university-related information. It combines the capabilities of Google's Gemini API with a Retrieval-Augmented Generation (RAG) pipeline to provide context-aware and accurate responses from a custom knowledge base.

---
## 🌐 Live Demo

🔗 **Deployed Application:** https://studentbuddy-ai.streamlit.app/

---

## 🚀 Features

- 💬 Interactive AI chatbot with conversational interface
- 🧠 Powered by Google Gemini API
- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic search using ChromaDB
- 🤖 `all-MiniLM-L6-v2` embeddings via ONNX Runtime (no PyTorch needed)
- 📄 Custom knowledge base support
- 💼 Placement preparation guidance
- 🚀 Internship guidance
- 📚 Academic support
- 📄 Resume building assistance
- 🎯 Career guidance
- 💰 Scholarship information
- 🗂 Chat history support (recent messages sent as conversation context)
- 💡 One-click Quick Questions in the sidebar
- ⚡ Automatic retry mechanism for API overload
- 🔄 Automatic model fallback when the primary model is overloaded
- 🔁 Vector store rebuilt automatically when the knowledge base changes
- 🎨 Modern and responsive Streamlit UI

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend |
| Streamlit | Web Application |
| Google Gemini API | Large Language Model |
| ChromaDB | Vector Database |
| ONNX Runtime (via ChromaDB) | Text Embeddings |
| python-dotenv | Environment Variables |

---

# 📂 Project Structure

```
StudentBuddy-AI/
│
├── app.py                 # Main Streamlit application
├── rag.py                 # RAG implementation
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── .streamlit/
│   └── config.toml        # Light theme, file watcher off
│
├── knowledge_base/
│   ├── academics.txt
│   ├── placements.txt
│   ├── internships.txt
│   └── faq.txt
│
└── vector_store/          # Generated automatically (not pushed to GitHub)
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/shreshth2906/studentbuddy-ai
```

---

## 2. Move into Project

```bash
cd studentbuddy-ai
```

---

## 3. Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Configure Environment Variables

Copy `.env.example` to `.env` and add your key.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

When deploying on Streamlit Community Cloud, add `GEMINI_API_KEY` under **App settings → Secrets** instead.

---

## 6. Run Application

```bash
streamlit run app.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model and builds the `vector_store/` directory. Later runs reuse it.

---

# ☁️ Deploying on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io) with `app.py` as the entry point.
3. Open **App settings → Secrets** and add:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

The app reads the key from the environment / `.env` first and falls back to Streamlit secrets.

---

# 🔧 Configuration

| Setting | File | Default | Description |
|---------|------|---------|-------------|
| `PRIMARY_MODEL` | `app.py` | `gemini-3.1-flash-lite` | Gemini model used for responses |
| `FALLBACK_MODEL` | `app.py` | `gemini-3.5-flash` | Model used when the primary model is overloaded |
| `MAX_HISTORY_MESSAGES` | `app.py` | `20` | Previous chat messages sent with each request |
| `MODEL_NAME` | `rag.py` | `all-MiniLM-L6-v2-onnx` | Embedding model label (changing it rebuilds the vector store) |
| `SIMILARITY_THRESHOLD` | `rag.py` | `1.0` | Maximum distance for a chunk to count as relevant (lower = stricter) |
| `N_RESULTS` | `rag.py` | `5` | Number of chunks retrieved per question |

---

# 🧠 How RAG Works

1. User asks a question.
2. The question is converted into an embedding.
3. ChromaDB performs semantic similarity search.
4. Relevant information is retrieved from the knowledge base.
5. Retrieved context is combined with the user's question.
6. Gemini generates a context-aware response.
7. If no relevant context is found, Gemini answers using its general knowledge.

---

# 📚 Knowledge Base

The chatbot retrieves information from documents stored inside the `knowledge_base/` directory.

Current knowledge sources include:

- Academics
- Placement Preparation
- Internship Guidance
- Frequently Asked Questions

Additional documents can easily be added by placing new `.txt` files inside the `knowledge_base` folder. The vector store is rebuilt automatically on the next start whenever a file is added, removed, renamed or edited, or the embedding model changes.

---

# 📸 Application Preview

Initial Interface-
> <img width="1916" height="1017" alt="Screenshot 2026-07-19 172422" src="https://github.com/user-attachments/assets/533a57aa-6510-4851-a484-fc04324b9097" />

Welcome Message-
><img width="1917" height="896" alt="Screenshot 2026-07-19 172457" src="https://github.com/user-attachments/assets/2dae6ace-0180-48ea-9e76-4ff66cab0c5c" />

Output Response-
><img width="1478" height="725" alt="Screenshot 2026-07-19 172608" src="https://github.com/user-attachments/assets/93684a59-53a0-4f6a-b10d-f930b36c39af" />



---

# 🔮 Future Enhancements

- PDF document support
- Multiple document upload
- Voice input
- Voice responses
- Chat export
- User authentication
- Multi-language support
- University-specific knowledge base
- Admin dashboard

---

# 📌 Key Highlights

- Retrieval-Augmented Generation (RAG)
- Google Gemini Integration
- ChromaDB Vector Database
- Semantic Search
- Context-Aware Responses
- Streamlit User Interface
- Automatic Retry & Fallback Mechanism

---

# 🩺 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Environment Key Missing" message | Add `GEMINI_API_KEY` to `.env` (local) or to the app's Secrets (Streamlit Cloud), then restart the app. |
| "The AI service is currently busy" | Both Gemini models are overloaded. Wait a moment and try again. |
| Model not found error | Change `PRIMARY_MODEL` / `FALLBACK_MODEL` in `app.py` to models available for your API key. |
| Answers ignore the knowledge base | Lower distances mean closer matches; try raising `SIMILARITY_THRESHOLD` slightly, or add more detailed content to `knowledge_base/`. |
| App is very slow on WSL | Keep the project in the Linux filesystem (e.g. `~/studentbuddy-ai`), not under `/mnt/c/...`. |
| Stale or broken search results | Delete the `vector_store/` folder and restart; it is rebuilt from `knowledge_base/`. |

---

# 👨‍💻 Author

**Shreshth Khetan**

Internship Project

StudentBuddy AI

---

# 📄 License

This project is developed for educational and internship submission purposes.
