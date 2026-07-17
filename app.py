import streamlit as st

st.set_page_config(
    page_title="StudentBuddy AI",
    page_icon="🎓",
)

st.title("🎓 StudentBuddy AI")

st.write("Welcome to StudentBuddy AI!")

question = st.text_input("Ask your question")

if st.button("Send"):
    if question:
        st.success(f"You asked: {question}")
    else:
        st.warning("Please enter a question.")