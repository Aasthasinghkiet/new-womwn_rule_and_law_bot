import streamlit as st
import requests

# Your backend API endpoint
API_URL = "https://women-law-rag-bot-2.onrender.com/ask"

# Streamlit App Config
st.set_page_config(page_title="Women Law Chatbot", page_icon="⚖️")

st.title("⚖️ Women Law & Rights Chatbot")
st.write("Ask any question related to women's legal rights based on Indian laws.")

# Input box and send button
user_question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not user_question.strip():
        st.warning("Please type a question.")
    else:
        # Make POST request to backend
        try:
            response = requests.post(API_URL, json={"question": user_question})
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer received.")
                st.success(answer)
            else:
                st.error(f"Server error: {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# Optional footer
st.markdown("---")
st.caption("Powered by Women Law RAG Engine | Backend running on Render")
