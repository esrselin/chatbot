import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage
import openai
import streamlit as st
from llama_index.core.node_parser import SentenceSplitter

# Set OpenAI API Key
openai.api_key = ""

system_prompt = """
You are a judicial decisions made by the Yargıtay expert Q&A system expert.

This application is configured to only accept and always respond in Turkish.

Your goal is to provide accurate and detailed information based on the user’s request and context. If you do not know the answer, just say "I don’t know." Always respond in Turkish.

reply with lot emojies.
"""

# Streamlit app title and description
st.set_page_config(
    page_title="Karar Bulucu 🧐🔍",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="auto",
)
st.title("Karar Bulucu 🧐🔍")
st.markdown(
    """
    Bu uygulama, Yargıtay kararlarını sorgulamanıza yardımcı olur. 
    Lütfen tüm sorularınızı Türkçe olarak yazın ve detaylı yanıtlar alın.
    """
)


@st.cache_resource
def get_models():
    llm = OpenAI(
        model="gpt-4",
        temperature=0.7,
        max_tokens=256,
        system_prompt=system_prompt,
    )
    embed_model = OpenAIEmbedding()
    return llm, embed_model


@st.cache_resource
def get_vector_store():
    llm, embed_model = get_models()
    if os.path.exists("vector_store"):
        st.info("Vektör Deposu Yükleniyor.")
        storage_context = StorageContext.from_defaults(persist_dir="vector_store")
        index = load_index_from_storage(storage_context)
    else:
        st.info("Vektör Deposu Oluşturuluyor.")
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index.storage_context.persist(persist_dir="vector_store")
    return index, llm


index, llm = get_vector_store()
query_engine = index.as_query_engine(llm=llm)


def response_generator():
    response = query_engine.query(st.session_state.messages[-1]["content"])
    for word in response.response.split():
        yield word + " "


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar for additional information or settings
st.sidebar.title("Ayarlar")
st.sidebar.markdown(
    """
    If you want to add details or text. You can add. 
    
    """
)
