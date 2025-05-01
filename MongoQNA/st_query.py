import streamlit as st
from groq import Groq
from langchain_milvus import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings

groq_api_key = st.secrets["GROQ_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]
zilliz_uri = st.secrets["ZILLIZ_URI_ENDPOINT"]
zilliz_token = st.secrets["ZILLIZ_TOKEN"]

# import os
# from dotenv import load_dotenv

# load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")
# google_api_key = os.getenv("GOOGLE_API_KEY")
# zilliz_uri = os.getenv("ZILLIZ_URI_ENDPOINT")
# zilliz_token = os.getenv("ZILLIZ_TOKEN")

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Groq LLM Client
def get_groq_client():
    return Groq(api_key=groq_api_key)

def generate_chat_response(user_input: str):
    # Vector store
    vector_store = Milvus(
        collection_name="Food_Details",
        connection_args={"uri":zilliz_uri, "token": zilliz_token},
        index_params={"index_type": "IVF_PQ", "metric_type": "COSINE"},
        embedding_function=embeddings
    )

    # Retrieve docs
    retrieved_docs = vector_store.similarity_search_with_relevance_scores(
        query=user_input,
        k=2,
        score_threshold=0.75
    )
    
    if retrieved_docs:
        context = "\n".join(doc[0].page_content for doc in retrieved_docs)
        prompt = f"""
        You are a warm, intelligent, and highly trained nutrition assistant, deeply familiar with the user's health goals and dietary needs.
        Provide thoughtful, precise, and conversational responses grounded only in the provided context.
        Speak naturally and confidently as if you've been trained specifically for this nutrition application.
        Keep responses concise (250 words max), highly relevant, and easy to understand—like a personal nutritionist would.
        Do *not* mention or hint at any documents, sources, or external materials.
        If the context does not contain enough relevant information to accurately answer the question, respond clearly with:
        "Sorry, the question seems irrelevant."

        Question: {user_input}
        Context: {context}

        **Answer should be to the point.**
        Only answer if the context directly supports or informs the question. Never answer from general knowledge or assumptions.
        """

        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            stream=True
        )

        assistant_output = ""
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    assistant_output += content
                    yield content
        client.close()
    else:
        vector_store.client.close()
        yield "Sorry the question seems Irrelevant."

# <---------------- UI ------------------>

st.set_page_config(page_title="SDP Chatbot", layout="wide")
st.title("🤖 Smart Diet Planner Chatbot")

# Sidebar
with st.sidebar:
    st.header("Knowledge Base")
    st.text("Food_Details")

user_input = st.chat_input("Question...")

# Display Chat History
if st.session_state.chat_history:
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

# <---------- MAIN ---------->
if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(("user", user_input))

    # Show assistant response
    with st.chat_message("assistant"):
        assistant_text = ""
        msg_placeholder = st.empty()
        for chunk in generate_chat_response(user_input):
            assistant_text += chunk
            msg_placeholder.markdown(assistant_text)
        st.session_state.chat_history.append(("assistant", assistant_text))