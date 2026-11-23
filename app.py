
import streamlit as st
import requests

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Women Law & Rights Chatbot",
    page_icon="⚖️",
    layout="centered"
)

# Backend API
API_URL = "https://women-law-rag-bot-2.onrender.com/ask"

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.main {
    background-color: #101116;
}

.stChatMessage {
    border-radius: 12px;
}

.title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# HEADER
# --------------------------------

st.title("⚖️ Women Law & Rights Chatbot")

st.markdown(
    "<p class='subtitle'>"
    "An AI-powered legal assistant based on Indian women's laws."
    "</p>",
    unsafe_allow_html=True
)

st.info(
    "Ask questions about Dowry, Domestic Violence, "
    "Workplace Sexual Harassment and other women's legal rights."
)


# --------------------------------
# SESSION STATE
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.header("⚖️ Women Law Assistant")

    st.write("Available legal documents:")

    acts = [
        "Dowry Prohibition Act, 1961",
        "Domestic Violence Act, 2005",
        "Sexual Harassment at Workplace Act, 2013",
        "Immoral Traffic (Prevention) Act, 1956",
        "Commission of Sati (Prevention) Act, 1987",
        "Indecent Representation of Women Act, 1986",
        "Other available legal documents"
    ]

    for act in acts:
        st.write("• " + act)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            st.markdown("**📚 Sources used:**")

            for source in message["sources"]:
                if isinstance(source, dict):
                    source = source.get(
                        "source",
                        source.get("name", "Legal document")
                    )

                st.write("• " + str(source))


# --------------------------------
# ASK QUESTION
# --------------------------------

user_question = st.chat_input(
    "Ask your legal question here..."
)

if user_question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching legal documents and generating answer..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json={"question": user_question},
                    timeout=120
                )

                response.raise_for_status()

                data = response.json()

                answer = data.get(
                    "answer",
                    "No answer received from the backend."
                )

                sources = (
                    data.get("sources")
                    or data.get("source_documents")
                    or []
                )

                st.markdown(answer)

                if sources:

                    st.markdown("**📚 Sources used:**")

                    for source in sources:

                        if isinstance(source, dict):
                            source = source.get(
                                "source",
                                source.get("name", "Legal document")
                            )

                        st.write("• " + str(source))

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except requests.exceptions.Timeout:

                error = "The server is taking too long to respond. Please try again."

                st.error(error)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error
                })

            except requests.exceptions.ConnectionError:

                error = "Unable to connect to the backend. Please check your Render server."

                st.error(error)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error
                })

            except requests.exceptions.HTTPError as e:

                error = f"Backend HTTP error: {e}"

                st.error(error)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error
                })

            except Exception as e:

                error = f"Something went wrong: {str(e)}"

                st.error(error)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error
                })


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Powered by Women Law RAG Engine | "
    "Backend hosted on Render"
)

st.caption(
    "This chatbot provides legal information, "
    "not professional legal advice."
)
