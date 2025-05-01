import streamlit as st
from collections import deque
from groq import Groq
from langchain_milvus import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# google_api_key = os.getenv("GOOGLE_API_KEY")
# zilliz_uri = os.getenv("ZILLIZ_URI_ENDPOINT")
# zilliz_token = os.getenv("ZILLIZ_TOKEN")

groq_api_key = st.secrets["GROQ_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]
zilliz_uri = st.secrets["ZILLIZ_URI_ENDPOINT"]
zilliz_token = st.secrets["ZILLIZ_TOKEN"]

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=google_api_key
)

# Initialize session state
if "user_query_history" not in st.session_state:
    st.session_state.user_query_history = deque(maxlen=3)

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = deque(maxlen=3)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Setup vector store
vector_store = Milvus(
    collection_name="Food_Details",
    connection_args={"uri": zilliz_uri, "token": zilliz_token},
    index_params={"index_type": "IVF_PQ", "metric_type": "COSINE"},
    embedding_function=embeddings
)

# Groq LLM client
def get_groq_client():
    return Groq(api_key=groq_api_key)

# Is query related to the last conversation?
def get_query_relationship(new_query: str):
    if not st.session_state.conversation_history:
        return 0.0
    else:
        last_query = st.session_state.conversation_history[-1][-1]["content"]
        prompt = f"""
            Given two messages, decide whether the second message is a continuation of the topic in the first message, or if it's a new, unrelated topic.
            Return only the score between 0 and 1 where 0 is not related and 1 being 100% related and NOTHING ELSE.
            last_query: "{last_query}"
            new_query: "{new_query}"
        """
        llm = get_groq_client()
        response = llm.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        score = float(response.choices[0].message.content.strip())
        return score

# Generate assistant response
def generate_chat_response(user_input: str):
    client = get_groq_client()
    st.session_state.user_query_history.append(user_input)
    
    score = get_query_relationship(user_input)
    
    if score > 0.75:
        chats = [message for chatpair in st.session_state.conversation_history for message in chatpair]
        prompt = f"""
            You are a highly knowledgeable and empathetic nutritionist assistant.
            Based on the previous conversation answer this followup question without mentioning or hinting at any documents, sources, or external materials.
            Always keep responses concise (under 250 words), accurate, and user-friendly.
            Never disclose your data source or say "Based on the document..." etc.
            Followup Question: {user_input}"""
        chats.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=chats,
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

        st.session_state.conversation_history.append([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output}
        ])
        client.close()
    else:
        retrieved_docs = vector_store.similarity_search_with_relevance_scores(
            query=user_input,
            k=2,
            score_threshold=0.75
        )

        if retrieved_docs:
            context = "\n".join(doc[0].page_content for doc in retrieved_docs)
            chat_history_str = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in list(sum(st.session_state.conversation_history, []))
            )
            prompt = f"""
            You are a highly knowledgeable and empathetic nutritionist assistant.
            Your role is to provide clear, evidence-based answers using retrieved information from a trusted knowledge base.
            Always keep responses concise (under 250 words), accurate, and user-friendly.
            Never disclose your data source or say "Based on the document..." etc.
            Chat History: {chat_history_str or "None"}
            Question: {user_input}
            Context: {context}
            Prioritize chat history when answering follow-ups. Only use context if relevant. 
            If both lack relevant info, say: "Sorry the question seems Irrelevant." Do not guess or answer from your own knowledge.
            """
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

            st.session_state.conversation_history.append([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_output}
            ])

            client.close()
        else:
            yield "Sorry the question seems Irrelevant."

# <---------------- UI ------------------>

st.set_page_config(page_title="SDP Chatbot", layout="wide")
st.title("🤖 Smart Diet Planner Chatbot")

# Sidebar for knowledge base selection
with st.sidebar:
    st.header("Knowledge Base")
    st.text("Food_Details")

# Chat input
user_input = st.chat_input("Question...")

# Display previous messages
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# If new user input
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        assistant_text = ""
        msg_placeholder = st.empty()
        for chunk in generate_chat_response(user_input):
            assistant_text += chunk
            msg_placeholder.markdown(assistant_text)

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
