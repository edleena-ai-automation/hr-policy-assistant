import os
import hashlib

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="wide",
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1.05rem;
    }

    .answer-box {
        padding: 1.25rem;
        border-radius: 14px;
        border: 1px solid #dbeafe;
        background: #f8fafc;
        margin-top: 1rem;
    }

    .source-box {
        padding: 0.9rem;
        border-radius: 10px;
        background: #f1f5f9;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>📋 HR Policy Assistant</h1>
        <p>
            Upload an HR policy document and ask questions.
            The assistant retrieves relevant sections from your policy
            before generating an answer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD GROQ API KEY
# =========================================================

def get_groq_api_key():
    """
    First checks Streamlit secrets.
    Falls back to environment variable for flexibility.
    """

    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    """
    Loads a lightweight Sentence Transformer model.
    The model is downloaded once and cached by Streamlit.
    """

    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from every page of the uploaded PDF.
    """

    pdf_bytes = uploaded_file.getvalue()

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        if text and text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    document.close()

    return pages


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """
    Basic cleanup for extracted PDF text.
    """

    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(pages, chunk_size=1000, overlap=150):
    """
    Split document text into overlapping chunks.

    Each chunk keeps its source page number.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = clean_text(page_data["text"])

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# =========================================================
# CREATE FAISS INDEX
# =========================================================

def build_faiss_index(chunks, model):
    """
    Convert chunks into embeddings and store them in FAISS.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(query, index, chunks, model, top_k=5):
    """
    Retrieve the most semantically relevant chunks.
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    retrieved = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        retrieved.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score),
            }
        )

    return retrieved


# =========================================================
# GROQ RESPONSE
# =========================================================

def generate_answer(question, retrieved_chunks, model_name):
    """
    Generate an answer using only the retrieved policy context.
    """

    if not GROQ_API_KEY:
        st.error(
            "GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets."
        )
        return None

    client = Groq(api_key=GROQ_API_KEY)

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        context_parts.append(
            f"""
SOURCE {i}
PAGE: {chunk['page']}

{chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions using the provided HR policy context.

IMPORTANT RULES:

1. Use the provided policy context as the primary source.
2. Do not invent HR rules that are not present in the context.
3. If the policy does not contain enough information to answer the question,
   clearly say that the uploaded policy does not provide enough information.
4. Do not present assumptions as company policy.
5. Be concise but useful.
6. When possible, mention the relevant policy page.
7. If the question is ambiguous, explain what information is missing.
8. Do not make legal claims unless the uploaded policy explicitly states them.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Answer the user's question based on the policy context above.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=1000,
    )

    return response.choices[0].message.content


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Retrieved policy sections",
        min_value=2,
        max_value=8,
        value=5,
        help="Number of relevant chunks retrieved before asking Groq.",
    )

    model_name = st.selectbox(
        "Groq model",
        [
           openai/gpt-oss-120b
        ],
        index=0,
    )

    st.divider()

    st.markdown(
        """
        **How it works**

        1. Upload HR policy
        2. Extract PDF text
        3. Split into chunks
        4. Create embeddings
        5. Search with FAISS
        6. Send relevant context to Groq
        7. Generate grounded answer
        """
    )


# =========================================================
# MODEL LOADING
# =========================================================

with st.spinner("Loading embedding model..."):
    embedding_model = load_embedding_model()


# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("📄 Upload HR Policy")

uploaded_file = st.file_uploader(
    "Upload your HR policy PDF",
    type=["pdf"],
    help="Upload a searchable/text-based HR policy PDF.",
)


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

if uploaded_file:

    # Generate a unique document ID
    file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    # Only process the document when necessary
    if st.session_state.get("file_hash") != file_hash:

        with st.spinner("Processing HR policy..."):

            pages = extract_text_from_pdf(uploaded_file)

            if not pages:
                st.error(
                    "No readable text was found in this PDF. "
                    "Please upload a text-based PDF."
                )
                st.stop()

            chunks = create_chunks(pages)

            if not chunks:
                st.error(
                    "The PDF could not be divided into usable text chunks."
                )
                st.stop()

            index = build_faiss_index(
                chunks,
                embedding_model,
            )

            st.session_state["file_hash"] = file_hash
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["filename"] = uploaded_file.name

        st.success(
            f"✅ {uploaded_file.name} processed successfully."
        )

    else:

        chunks = st.session_state["chunks"]
        index = st.session_state["index"]

    # -----------------------------------------------------
    # DOCUMENT INFO
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Pages",
            len(pages) if "pages" in locals() else "Loaded",
        )

    with col2:
        st.metric(
            "Policy Chunks",
            len(chunks),
        )

    with col3:
        st.metric(
            "Vector Dimension",
            index.d,
        )

    st.divider()

    # =====================================================
    # QUESTION
    # =====================================================

    st.subheader("💬 Ask Your HR Policy")

    example_questions = [
        "What is the annual leave policy?",
        "How many sick leaves are employees entitled to?",
        "What is the probation period?",
        "What is the maternity leave policy?",
        "What is the attendance policy?",
    ]

    selected_question = st.selectbox(
        "Example questions",
        ["Type my own question"] + example_questions,
    )

    if selected_question == "Type my own question":
        question = st.text_input(
            "Ask a question about the uploaded HR policy",
            placeholder="e.g. How many annual leaves can I take?",
        )
    else:
        question = selected_question

    ask_button = st.button(
        "🔎 Ask HR Policy Assistant",
        type="primary",
        use_container_width=True,
    )

    # =====================================================
    # ANSWER
    # =====================================================

    if ask_button:

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching the HR policy..."):

                retrieved_chunks = retrieve_chunks(
                    question,
                    index,
                    chunks,
                    embedding_model,
                    top_k=top_k,
                )

            with st.spinner("Generating answer..."):

                answer = generate_answer(
                    question,
                    retrieved_chunks,
                    model_name,
                )

            if answer:

                st.markdown(
                    '<div class="answer-box">',
                    unsafe_allow_html=True,
                )

                st.subheader("🤖 Answer")

                st.write(answer)

                st.markdown("</div>", unsafe_allow_html=True)

                # =================================================
                # SOURCES
                # =================================================

                st.subheader("📚 Retrieved Policy Sources")

                for i, source in enumerate(
                    retrieved_chunks,
                    start=1,
                ):

                    with st.expander(
                        f"Source {i} — Page {source['page']} "
                        f"— Similarity {source['score']:.2f}"
                    ):

                        st.write(source["text"])

else:

    st.info(
        "👆 Upload an HR policy PDF above to start asking questions."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "HR Policy Assistant • RAG + FAISS + Sentence Transformers + Groq"
)
