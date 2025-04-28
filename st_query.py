import streamlit as st
from collections import deque
from groq import Groq
from langchain_milvus import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings

groq_api_key = st.secrets["GROQ_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]
zilliz_uri = st.secrets["ZILLIZ_URI"]
zilliz_token = st.secrets["ZILLIZ_TOKEN"]

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)

if "user_query_history" not in st.session_state:
    st.session_state.user_query_history = deque(maxlen=5)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Groq LLM Client
def get_groq_client():
    return Groq(api_key=groq_api_key)

def get_collections():
    vector_store = Milvus(
        connection_args={"uri":zilliz_uri, "token": zilliz_token},
        index_params={"index_type": "IVF_PQ", "metric_type": "COSINE"},
        embedding_function=embeddings
    )

    collections = vector_store.client.list_collections()
    return collections

def generate_chat_response(document_id: str, user_input: str):
    st.session_state.user_query_history.append(user_input)
    contextual_query = " ".join(st.session_state.user_query_history)

    # Vector store
    vector_store = Milvus(
        collection_name=document_id,
        connection_args={"uri":zilliz_uri, "token": zilliz_token},
        index_params={"index_type": "IVF_PQ", "metric_type": "COSINE"},
        embedding_function=embeddings
    )

    # Retrieve docs
    retrieved_docs = vector_store.similarity_search_with_relevance_scores(
        query=user_input, 
        k=3, 
        score_threshold=0.75
    )
    
    if not retrieved_docs:
        retrieved_docs = vector_store.similarity_search_with_relevance_scores(
            query=contextual_query, 
            k=3, 
            score_threshold=0.75
        )
    
    if retrieved_docs:
        context = "\n".join(doc[0].page_content for doc in retrieved_docs)
        chat_history_str = "\n".join([f"{role}: {message}" for role, message in st.session_state.chat_history])
        prompt = f"""
        You are a highly knowledgeable and empathetic assistant specializing in diabetes.
        Your role is to provide clear, evidence-based answers using retrieved information from a trusted knowledge base.
        Always keep responses concise (do not exceed 250 word limit), accurate, and user-friendly.
        Never disclose your data source or say "Based on the document..." etc.
        Context: {context}
        Question: {user_input}
        chat history: {chat_history_str}
        you can reference this chat history if its present.
        If question is not related to context, or no relevant info is found, reply with:
        "Sorry the question seems Irrelevant." **but do not answer from your knowledge base. EVER**
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
    document_id = st.selectbox("Select Knowledge Base", options=["Hydration_Data","Craving_Data","Oil_Intake_Data"])

    # if st.button("🔄 Reset Chat Context"):
    #     st.session_state.user_query_history.clear()
    #     st.session_state.chat_history.clear()
    #     st.success("Reset successful!")

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
        for chunk in generate_chat_response(document_id, user_input):
            assistant_text += chunk
            msg_placeholder.markdown(assistant_text)
        st.session_state.chat_history.append(("assistant", assistant_text))