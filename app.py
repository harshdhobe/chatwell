import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


@st.cache_resource
def load_model():
    # Load PDF files
    file_path = 'data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf'
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Create chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Make embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Vector store
    DB_FAISS_PATH = "vectorstore/db_faiss"
    if not os.path.exists(DB_FAISS_PATH):
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(DB_FAISS_PATH)
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Setup LLM
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        temperature=0.1,
        max_new_tokens=512
    )
    
    chat_model = ChatHuggingFace(llm=llm)
    
    # Prompt Template
    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided context.
          If the context is insufficient, just say you don't know.

          context:{context}
          question:{question}
          Start the answer directly. No small talk please.
        """,
        input_variables=["context", "question"]
    )
    
    # Retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text
    
    # Create chain
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | chat_model | parser
    
    return main_chain

# Streamlit UI 

st.title(" 🤖 ChatWell ")

#configuring stramlit page
st.set_page_config(
    page_title="ChatWell",
    page_icon="💬",
    layout="centered",
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chain' not in st.session_state:
    st.session_state.chain = None

# Load model
if st.session_state.chain is None:
    with st.spinner("Loading AI Model..."):
        st.session_state.chain = load_model()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a medical question"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner(" "):
            response = st.session_state.chain.invoke(prompt)
        st.write(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})