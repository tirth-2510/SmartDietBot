import streamlit as st
import ast
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
    
if "conversation_with_context_history" not in st.session_state:
    st.session_state.conversation_with_context_history = deque(maxlen=3)

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
        return {"score": 0, "need_context": False}
    else:
        last_query = "\n".join(["Q: "+chatpair[0]["content"]+"\nA: "+chatpair[-1]["content"] for chatpair in st.session_state.conversation_history])
        # last_query ="Q: "+st.session_state.conversation_history[-1][0]["content"]+"\nA: "+ st.session_state.conversation_history[-1][-1]["content"]
        
        prompt = f"""
            You are given a conversation history between a user and an assistant. Then a new query is asked by the user.

            Determine whether the new query is contextually related to the *topic or intent* of the earlier conversation — even if the words differ — or if it introduces a new, unrelated topic.

            Instructions:
            - Return a dictionary with:
            - "score": float between 0 and 1, where 0 means totally unrelated, 1 means entirely related.
                (NOTE: If it's a follow-up question or a shift in subtopic that still relates to the original intent — like continuing a health-related conversation — then score it above 0.75.)
            - "need_context": True if additional context or information is needed to accurately determine relationship.

            **Always respond in dictionary format:**
            Example:
            Conversation history: Q: I am Maharashtrian, diabetic, suggest me good breakfast
                                  A: I recommend Jowar Dosa + Sambhar as a good diabetic breakfast for Maharashtrians."
            new_query: "Suggest me supplements instead."
            return: {{'score': 0.9, 'need_context': True}}

            Conversation history: "{last_query}"
            new_query: "{new_query}"
            return : 
        """   
        llm = get_groq_client()
        response = llm.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        response = response.choices[0].message.content.strip()
        return ast.literal_eval(response)

# Generate assistant response
def generate_chat_response(user_input: str):
    client = get_groq_client()
    st.session_state.user_query_history.append(user_input)
    response = get_query_relationship(user_input)
    score = response["score"]
    need_context = response["need_context"]
    if score > 0.75:
        chats = [message for chatpair in st.session_state.conversation_with_context_history for message in chatpair]
        prompt = f"""
            You are a highly knowledgeable and empathetic nutritionist assistant.
            Answer this followup question from the previous conversation (Context more preferable if provided), **Do Not from your own knowledge base else you are fired.**, without mentioning or hinting at any documents, sources, or external materials.
            If no relevant answer is found in the conversation history, than deny the user politely explaining No relevant Context was found for their question.
            Always keep responses concise (under 250 words), accurate, and user-friendly.
            Never disclose your data source or say "Based on the document..." etc.
            Followup Question: {user_input}"""
        if need_context:
            new_query = user_input + st.session_state.user_query_history[-2]
            retrieved_docs = vector_store.similarity_search_with_relevance_scores(
                query=new_query,
                k=2,
                score_threshold=0.75
            )
            context = "\n".join(doc[0].page_content for doc in retrieved_docs)
            prompt += f"\nContext: {context}"
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
        st.session_state.conversation_with_context_history.append([
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
            # prompt = f"""
            # You are a highly knowledgeable and empathetic nutritionist assistant.
            # Your role is to provide clear, evidence-based answers using retrieved information from a trusted knowledge base.
            # Always keep responses concise (under 250 words), accurate, and user-friendly.
            # Never disclose your data source or say "Based on the document..." etc.
            # Chat History: {chat_history_str or "None"}
            # Question: {user_input}
            # Context: {context}
            # Prioritize chat history when answering follow-ups. Only use context if relevant. 
            # If both lack relevant info, say: "Sorry the question seems Irrelevant." Do not guess or answer from your own knowledge.
            # """
            prompt = f"""
            You are a highly knowledgeable and empathetic nutritionist assistant.
            Your role is to provide clear, evidence-based answers using below retrieved context from a trusted knowledge base.
            Always keep responses concise (under 250 words), accurate, and user-friendly.
            Never disclose your data source or say "Based on the document..., In the provided context..." etc.
            Question: {user_input}
            Context: {context}
            Do not any extra information from your own knowledge base.
            If Context lacks relevant information to answer the question than deny the user politely explain No relevant Context was found for their question. Do not guess or answer from your own knowledge.
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
            
            st.session_state.conversation_with_context_history.append([
                {"role": "user", "content": "Question: " + user_input + "\n context: " + context},
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
