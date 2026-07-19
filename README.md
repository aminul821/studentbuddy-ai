# 🎓 StudentBuddy AI

> **An Intelligent Student Support Assistant powered by Google Gemini and Retrieval-Augmented Generation (RAG).**

StudentBuddy AI is an AI-powered chatbot designed to assist students with academic queries, placement preparation, internships, career guidance, scholarships, resume building, and university-related information. It combines the capabilities of Google's Gemini API with a Retrieval-Augmented Generation (RAG) pipeline to provide context-aware and accurate responses from a custom knowledge base.

---
## 🌐 Live Demo

🔗 **Deployed Application:** [https://your-app.streamlit.app](https://studentbuddy-ai.streamlit.app/)

---

## 🚀 Features

- 💬 Interactive AI chatbot with conversational interface
- 🧠 Powered by Google Gemini API
- 📚 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic search using ChromaDB
- 🤖 Sentence Transformer embeddings (`all-MiniLM-L6-v2`)
- 📄 Custom knowledge base support
- 💼 Placement preparation guidance
- 🚀 Internship guidance
- 📚 Academic support
- 📄 Resume building assistance
- 🎯 Career guidance
- 💰 Scholarship information
- 🗂 Chat history support
- ⚡ Automatic retry mechanism for API overload
- 🔄 Automatic model fallback
- 🎨 Modern and responsive Streamlit UI

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend |
| Streamlit | Web Application |
| Google Gemini API | Large Language Model |
| ChromaDB | Vector Database |
| Sentence Transformers | Text Embeddings |
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
git clone https://github.com/<your-username>/studentbuddy-ai.git
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

Create a `.env` file.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## 6. Run Application

```bash
streamlit run app.py
```

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

Additional documents can easily be added by placing new `.txt` files inside the `knowledge_base` folder.

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
- Cloud deployment

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

# 👨‍💻 Author

**Shreshth Khetan**

Internship Project

StudentBuddy AI

---

# 📄 License

This project is developed for educational and internship submission purposes.
