import re
import time
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

load_dotenv()

# --- Config ---
PDF_PATH = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
VECTORSTORE_PATH = "vectorstore/db_faiss"
RETRIEVAL_K = 5
# L2 distance: lower = better match. Queries with no real hit are usually > 1.15.
MAX_RELEVANCE_DISTANCE = 1.15

HF_INFERENCE_PROVIDER = os.getenv("HF_INFERENCE_PROVIDER", "groq")
HF_INFERENCE_MODEL = os.getenv(
    "HF_INFERENCE_MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct"
)
# Used if primary provider fails (network / rate limit)
HF_INFERENCE_PROVIDER_FALLBACK = os.getenv(
    "HF_INFERENCE_PROVIDER_FALLBACK", "featherless-ai"
)
HF_INFERENCE_MODEL_FALLBACK = os.getenv(
    "HF_INFERENCE_MODEL_FALLBACK", "meta-llama/Llama-3.1-8B-Instruct"
)

GREETING_RE = re.compile(
    r"^(hi|hello|hey|hola|good\s+(morning|afternoon|evening)|howdy)[\s!.?]*$",
    re.IGNORECASE,
)


def get_hf_token() -> str | None:
    # Streamlit Community Cloud secrets (Manage app → Secrets)
    try:
        for key in ("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if key in st.secrets:
                return str(st.secrets[key]).strip()
    except Exception:
        pass
    for key in ("HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def validate_hf_token(token: str) -> None:
    from huggingface_hub import HfApi

    try:
        HfApi().whoami(token=token)
    except Exception as exc:
        raise RuntimeError(
            "Invalid or missing Hugging Face API token. "
            "Create one at https://huggingface.co/settings/tokens "
            'with "Inference Providers" enabled, then set:\n'
            "HUGGINGFACEHUB_API_TOKEN=hf_your_token_here"
        ) from exc


def is_greeting(text: str) -> bool:
    return bool(GREETING_RE.match(text.strip()))


def is_network_error(exc: BaseException) -> bool:
    err = str(exc).lower()
    needles = (
        "maxretryerror",
        "nameresolution",
        "getaddrinfo failed",
        "connection",
        "timeout",
        "router.huggingface.co",
        "temporary failure",
    )
    return any(n in err for n in needles)


def make_chat_model(provider: str, model: str, token: str) -> ChatHuggingFace:
    llm = HuggingFaceEndpoint(
        repo_id=model,
        provider=provider,
        huggingfacehub_api_token=token,
        temperature=0.2,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)


def invoke_chat_with_retry(
    chat_model: ChatHuggingFace,
    messages: list,
    *,
    fallback: ChatHuggingFace | None = None,
    attempts: int = 3,
) -> str:
    """Call the LLM with retries and optional fallback provider."""
    last_error: BaseException | None = None
    models_to_try: list[ChatHuggingFace] = [chat_model]
    if fallback is not None:
        models_to_try.append(fallback)

    for model in models_to_try:
        for attempt in range(attempts):
            try:
                result = model.invoke(messages)
                content = getattr(result, "content", result)
                return str(content).strip()
            except Exception as exc:
                last_error = exc
                if not is_network_error(exc) and attempt == attempts - 1:
                    break
                if attempt < attempts - 1:
                    time.sleep(1.5 * (attempt + 1))
        if last_error and not is_network_error(last_error):
            raise last_error

    raise last_error or RuntimeError("LLM request failed")


HUGGINGFACEHUB_API_TOKEN = get_hf_token()

MEDICAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are ChatWell, a careful medical information assistant.
You answer using ONLY the encyclopedia excerpts in Context below.

Guidelines:
- Write 2–4 clear sentences (or a short bullet list for medicines/treatments).
- Synthesize information; do not copy random words or headings from Context.
- Never answer with a single word unless listing drug names.
- If Context is empty or not relevant, say exactly:
  "I don't have enough information in my medical encyclopedia about that. Please ask about a specific symptom, condition, or treatment."
- Do not diagnose or prescribe; share encyclopedia information only.""",
        ),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            """Context:
{context}

Question: {question}

Answer:""",
        ),
    ]
)


@st.cache_resource
def load_rag_stack():
    if not HUGGINGFACEHUB_API_TOKEN:
        raise RuntimeError(
            "Hugging Face API token not found.\n\n"
            "Streamlit Cloud: open Manage app → Settings → Secrets and add:\n"
            'HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"\n\n'
            "Local: add the same line to `.env` in the project folder."
        )
    validate_hf_token(HUGGINGFACEHUB_API_TOKEN)

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    if not os.path.exists(VECTORSTORE_PATH):
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(VECTORSTORE_PATH)

    db = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )

    primary_chat = make_chat_model(
        HF_INFERENCE_PROVIDER, HF_INFERENCE_MODEL, HUGGINGFACEHUB_API_TOKEN
    )
    fallback_chat = None
    if HF_INFERENCE_PROVIDER_FALLBACK and HF_INFERENCE_MODEL_FALLBACK:
        fallback_chat = make_chat_model(
            HF_INFERENCE_PROVIDER_FALLBACK,
            HF_INFERENCE_MODEL_FALLBACK,
            HUGGINGFACEHUB_API_TOKEN,
        )

    return {
        "db": db,
        "primary_chat": primary_chat,
        "fallback_chat": fallback_chat,
        "parser": StrOutputParser(),
    }


def retrieve_context(db: FAISS, question: str) -> str:
    """Return top chunks that are actually relevant (by distance)."""
    scored = db.similarity_search_with_score(question, k=RETRIEVAL_K * 2)
    relevant = [
        doc.page_content.strip()
        for doc, score in scored
        if score <= MAX_RELEVANCE_DISTANCE and doc.page_content.strip()
    ][:RETRIEVAL_K]

    if not relevant:
        return ""

    return "\n\n---\n\n".join(relevant)


def history_to_messages(history: list[dict[str, str]]) -> list:
    """Last 4 turns for follow-up questions (e.g. fever after cough)."""
    messages: list = []
    for msg in history[-8:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def answer_question(
    stack: dict[str, Any],
    question: str,
    history: list[dict[str, str]],
) -> str:
    if is_greeting(question):
        return (
            "Hello! I'm ChatWell, a medical encyclopedia assistant. "
            "Ask me about symptoms, conditions, treatments, or medicines "
            "(for example: fever, cough, or cold remedies)."
        )

    context = retrieve_context(stack["db"], question)

    messages = MEDICAL_PROMPT.format_messages(
        context=context or "(no relevant excerpts found)",
        question=question,
        history=history_to_messages(history),
    )

    return invoke_chat_with_retry(
        stack["primary_chat"],
        messages,
        fallback=stack["fallback_chat"],
    )


# --- Streamlit UI (set_page_config must be first) ---
st.set_page_config(
    page_title="ChatWell",
    page_icon="💬",
    layout="centered",
)

st.title("🤖 ChatWell")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "stack" not in st.session_state:
    st.session_state.stack = None

if st.session_state.stack is None:
    with st.spinner("Loading medical encyclopedia and AI model…"):
        try:
            st.session_state.stack = load_rag_stack()
        except Exception as exc:
            st.error(str(exc))
            st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a medical question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = answer_question(
                    st.session_state.stack,
                    prompt,
                    st.session_state.messages[:-1],
                )
            except Exception as exc:
                err = str(exc)
                if "401" in err or "Unauthorized" in err:
                    st.error(
                        "Hugging Face authentication failed. Update "
                        "`HUGGINGFACEHUB_API_TOKEN` in `.env` "
                        "(https://huggingface.co/settings/tokens)."
                    )
                elif is_network_error(exc):
                    st.error(
                        "Could not reach Hugging Face inference servers "
                        "(network/DNS issue). Check your internet connection, "
                        "disable VPN if it blocks huggingface.co, wait a minute, "
                        "and try again."
                    )
                else:
                    st.error(f"Error generating response: {err}")
                st.stop()
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
