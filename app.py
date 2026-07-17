import os
import time
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Load environment variables from .env file
load_dotenv()

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
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

# ==========================================
# INITIALIZATION FUNCTIONS
# ==========================================
def init_page_config():
    """Configures Streamlit page titles and layouts."""
    st.set_page_config(
        page_title="StudentBuddy AI",
        page_icon="🎓",
        layout="centered"
    )

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
        return None
    return genai.Client(api_key=api_key)

# ==========================================
# DEFENSIVE GEMINI API INTERACTION LOGIC
# ==========================================
def execute_generation_with_retry(client, contents, config, model_name, max_retries=3):
    """Executes call with exponential backoff to absorb 503 high-demand spikes."""
    delay = 1.0  # Initial wait time in seconds
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
    Formats conversation context and executes generation with proactive fallback management.
    """
    try:
        formatted_contents = []
        for msg in chat_history:
            if msg == chat_history and msg["role"] == "assistant":
                continue
            formatted_contents.append(
                types.Content(
                    role="user" if msg["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )
            
        formatted_contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=new_user_message)])
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        )

        # Step 1: Attempt generation with Primary Model (with Retries)
        try:
            response = execute_generation_with_retry(client, formatted_contents, config, PRIMARY_MODEL)
            if response and response.text:
                return response.text
        except APIError as e:
            if e.code != 503:
                raise e  # If it's not a server overload, skip directly to the outer exception block
            
            # Step 2: Fallback to alternative model line if Primary continues to fail
            # st.warning(f"⚠️ Primary model is busy. Route optimization fallback engaged using alternative track...")
            response = execute_generation_with_retry(client, formatted_contents, config, FALLBACK_MODEL)
            if response and response.text:
                return response.text

        return "⚠️ Server limits reached. The model endpoints are overloaded. Please click 'Clear Chat' or wait a few moments before trying again."

    except APIError as e:
        st.error(f"❌ Gemini API Error: {e.message}")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        return None

# ==========================================
# UI COMPONENTS (SIDEBAR & MAIN)
# ==========================================
def render_sidebar():
    """Renders all elements within the sidebar structure."""
    with st.sidebar:
        st.title("🎓 StudentBuddy AI")
        st.markdown("**Your AI-powered Student Support Assistant.**")
        st.divider()
        
        st.subheader("📌 About")
        st.caption(
            "StudentBuddy AI is designed exclusively to help students navigate "
            "university life, exam preparation, internships, resume updates, "
            "and placement tracks."
        )
        st.divider()
        
        st.subheader("💡 Example Questions")
        examples = [
            "How do I prepare for placements?",
            "Explain CGPA.",
            "Internship tips.",
            "Scholarship eligibility.",
            "Resume writing tips."
        ]
        for example in examples:
            st.markdown(f"- *{example}*")
            
        st.divider()
        
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "👋 Welcome back! How can I help you today?"}
            ]
            st.rerun()

def render_chat_interface(client):
    """Handles rendering of chat elements and processing of user text inputs."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask about placement prep, scholarships, exams..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("StudentBuddy is checking details..."):
                ai_response = generate_student_response(
                    client=client,
                    chat_history=st.session_state.messages[:-1], 
                    new_user_message=user_query
                )
                
                if ai_response:
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# ==========================================
# MAIN APPLICATION CONTROLLER
# ==========================================
def main():
    init_page_config()
    init_session_state()
    render_sidebar()

    st.title("🎓 StudentBuddy AI")
    st.write("Welcome to your personal guidance hub. Get immediate assistance with your academic tracks, job profiles, and career workflows.")
    st.divider()

    client = get_gemini_client()
    if not client:
        st.error("🔑 Environment Key Missing: Provide a valid `GEMINI_API_KEY` environment token inside your environment configuration or active `.env` file setup.")
        return

    render_chat_interface(client)

if __name__ == "__main__":
    main()
