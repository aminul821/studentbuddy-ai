import os
import time
import streamlit as st
from rag import retrieve_context
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Load environment variables from .env file
load_dotenv()

# CONSTANTS & CONFIGURATION
SYSTEM_PROMPT = """You are StudentBuddy AI, an intelligent student support assistant. You help students with:
* Academics
* Placements
* Internships
* Scholarships
* Career Guidance
* Resume Building
* University Information

Provide accurate, concise, friendly, and easy-to-understand responses. 
If the user asks something unrelated to student support, politely explain that StudentBuddy AI is primarily designed for helping students, but provide a brief helpful response if appropriate. 
Avoid making up facts. If unsure, clearly mention uncertainty."""

# Fallback setup: If the main model suffers traffic spikes, the code drops to the lighter tier
PRIMARY_MODEL = "gemini-3.1-flash-lite"
FALLBACK_MODEL = "gemini-3.5-flash"

# INITIALIZATION FUNCTIONS
def init_page_config():
    """Configures Streamlit page titles and layouts."""
    st.set_page_config(
        page_title="StudentBuddy AI",
        page_icon="🎓",
        layout="centered"
    )

def load_custom_css():
    st.markdown("""
        <style>

        /* ---------- Hide Streamlit Branding ---------- */

        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        header {visibility:hidden;}

        /* ---------- Main App ---------- */

        .block-container{
            padding-top:1.5rem;
            padding-bottom:2rem;
            max-width:1100px;
        }

        /* ---------- Hero ---------- */

        .hero-card{
            background:linear-gradient(135deg,#2563eb,#1d4ed8);
            color:white;
            padding:32px;
            border-radius:22px;
            margin-bottom:25px;
            box-shadow:0 15px 35px rgba(37,99,235,.18);
        }

        .hero-title{
            font-size:38px;
            font-weight:700;
            margin-bottom:8px;
        }

        .hero-subtitle{
            font-size:18px;
            opacity:.92;
        }

        /* ---------- Sidebar ---------- */

        section[data-testid="stSidebar"]{
            border-right:1px solid #E5E7EB;
            background:#FAFAFA;
        }

        /* ---------- Cards ---------- */

        .info-card{
            background:white;
            border-radius:16px;
            padding:18px;
            border:1px solid #E5E7EB;
            margin-bottom:15px;
            box-shadow:0 4px 12px rgba(0,0,0,.05);
        }

        /* ---------- Footer ---------- */

        .footer{
            text-align:center;
            margin-top:45px;
            color:#94A3B8;
            font-size:13px;
        }

        /* ---------- Buttons ---------- */

        .stButton>button{
            border-radius:12px;
            border:none;
            height:42px;
            font-weight:600;
            transition:.25s;
        }

        .stButton>button:hover{
            transform:translateY(-2px);
        }
    
        .info-card{
            background:white;
            padding:18px;
            border-radius:18px;
            margin-bottom:15px;
            border:1px solid #E5E7EB;
            box-shadow:0 8px 20px rgba(0,0,0,.05);
            transition:.3s;
        }
                
        .info-card:hover{ 
            transform:translateY(-3px);
        }

        .stChatMessage{
            border-radius:15px;
            padding:10px;
        }
                
        button[kind="secondary"]{
            border-radius:12px;
        }

        /* ---------- Chat Input ---------- */

        div[data-testid="stChatInput"]{
            border-radius:16px;
        }

        </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initializes chat history and setup markers if not present."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi there! I'm StudentBuddy AI. How can I help you with your academics, placements, internships, or career goals today?"}
        ]

@st.cache_resource
def get_gemini_client():
    """Initializes and returns the official Google GenAI Client securely."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return st.secrets["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)

# DEFENSIVE GEMINI API INTERACTION LOGIC
def execute_generation_with_retry(client, contents, config, model_name, max_retries=3):
    """Executes call with exponential backoff to absorb 503 high-demand spikes."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return response
        except APIError as e:
            # Check for HTTP 503 Service Unavailable / Overloaded
            if e.code == 503 or "demand" in str(e.message).lower() or "overloaded" in str(e.message).lower():
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Double the backoff sleep timing window
                    continue
            raise e  # Propagate out if out of retry attempts or it's a structural error
    return None

def generate_student_response(client, chat_history, new_user_message):
    """
    Generates a response using Gemini with optional RAG context.
    """
    try:
        # Retrieve relevant context from the knowledge base
        context = retrieve_context(new_user_message)

        formatted_contents = []

        # Add previous conversation (skip initial welcome message)
        for i, msg in enumerate(chat_history):
            if i == 0 and msg["role"] == "assistant":
                continue

            formatted_contents.append(
                types.Content(
                    role="user" if msg["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )

        # Build prompt depending on whether context exists
        if context and context.strip():

            prompt = f"""
You are StudentBuddy AI, an intelligent student support assistant.

You have retrieved the following information from the StudentBuddy Knowledge Base.

=========================
Knowledge Base
=========================
{context}

=========================
Student Question
=========================
{new_user_message}

Instructions:

- Use the retrieved knowledge as the primary source.
- If the knowledge is incomplete, supplement it with your own knowledge.
- If the knowledge and your own knowledge conflict, clearly mention both.
- Mention the knowledge base only if it is relevant.
- Give a friendly, well-structured answer using headings and bullet points where appropriate.
"""

        else:

            prompt = f"""
You are StudentBuddy AI.

No relevant information was found in the knowledge base.

Answer the student's question using your own knowledge.

Student Question:
{new_user_message}

Provide a friendly, accurate and well-structured answer.
"""

        # Add current prompt
        formatted_contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        )

        # Primary Model
        try:
            response = execute_generation_with_retry(
                client,
                formatted_contents,
                config,
                PRIMARY_MODEL
            )

            if response and response.text:
                return response.text

        except APIError as e:

            if e.code != 503:
                raise e

            # Fallback Model
            response = execute_generation_with_retry(
                client,
                formatted_contents,
                config,
                FALLBACK_MODEL
            )

            if response and response.text:
                return response.text

        return (
            "⚠️ The AI service is currently busy. "
            "Please try again in a few moments."
        )

    except APIError as e:
        st.error(f"❌ Gemini API Error: {e.message}")
        return None

    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        return None

# UI COMPONENTS (SIDEBAR & MAIN)
def render_sidebar():
    with st.sidebar:

        col1, col2 = st.columns([5,1])

        with col1:
            st.markdown("""
            <h2 style="margin-bottom:0;">🎓 StudentBuddy AI</h2>
            <p style="margin-top:0;color:#64748B;font-size:14px;">
            Student Support Assistant
            </p>
            """, unsafe_allow_html=True)

        with col2:
            clear = st.button(
                "🗑",
                help="Clear Chat"
            )

        if clear:
            st.session_state.messages = [
                {
                    "role":"assistant",
                    "content":"👋 Welcome! How can I help you today?"
                }
            ]
            st.rerun()

        st.markdown("---")

        st.markdown("""
        <div class="info-card">

        <h4>ℹ About</h4>

        StudentBuddy AI helps students with

        • Academics

        • Placements

        • Internships

        • Resume Building

        • Career Guidance

        </div>
        """,unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">

        <h4>🎯 Capabilities</h4>

        ✅ Academic Support

        ✅ Placement Preparation

        ✅ Internship Guidance

        ✅ Resume Review

        ✅ Career Advice

        </div>
        """,unsafe_allow_html=True)

        st.subheader("💡 Quick Questions")

        examples=[
            "How do I prepare for placements?",
            "Explain CGPA",
            "Resume writing tips",
            "Internship guidance",
            "Scholarship eligibility"
        ]

        for q in examples:

            if st.button(
                q,
                key=q,
                use_container_width=True
            ):
                st.session_state.example_prompt=q

        st.markdown("---")

        st.markdown("""
            <div class="footer">
            Made with ❤️ using Streamlit & Gemini
            StudentBuddy AI • Version 1.0
            </div>
        """,unsafe_allow_html=True)

def render_chat_interface(client):
    """Handles rendering of chat elements and processing of user text inputs."""
    for message in st.session_state.messages:
        avatar="🤖" if message["role"]=="assistant" else "👤"
        with st.chat_message(message["role"],avatar=avatar):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask about placement prep, scholarships, exams..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🤖 StudentBuddy is thinking..."):
                ai_response = generate_student_response(
                    client=client,
                    chat_history=st.session_state.messages[:-1], 
                    new_user_message=user_query
                )
                
                if ai_response:
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# MAIN APPLICATION CONTROLLER
def main():
    init_page_config()
    load_custom_css()
    init_session_state()
    render_sidebar()

    st.markdown("""
        <div class="hero-card">
                
        <div class="hero-title">
                
        🎓 StudentBuddy AI
                
        </div>
                
        <div class="hero-subtitle">
                
        Your AI-powered Student Support Assistant
                
        </div>
                
        <br>
                
        Helping students with

        📚 Academics • 💼 Placements • 🚀 Internships

        📄 Resume Building • 🎯 Career Guidance

        </div>
    """,unsafe_allow_html=True)
    st.divider()

    client = get_gemini_client()
    if not client:
        st.error("🔑 Environment Key Missing: Provide a valid `GEMINI_API_KEY` environment token inside your environment configuration or active `.env` file setup.")
        return

    if len(st.session_state.messages)==1:
        
        st.markdown("""
                    <div class="info-card">
                    
                    <h2>👋 Welcome!</h2>
                    
                    Ask me anything about
                    
                    📚 Academics
                    
                    💼 Placements
                    
                    🚀 Internships
                    
                    📄 Resume
                    
                    🎯 Career
                    
                    </div>
        """,unsafe_allow_html=True)
    render_chat_interface(client)

if __name__ == "__main__":
    main()
