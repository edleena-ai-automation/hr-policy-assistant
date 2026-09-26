import os
import hashlib

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "HR Policy Assistant"

# Required Groq model
GROQ_MODEL = "openai/gpt-oss-120b"

# Sentence Transformer embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a5f 100%
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .answer-box {
        padding: 1.25rem;
        border-radius: 14px;
        border: 1px solid #dbeafe;
        background: #f8fafc;
        margin-top: 1rem;
    }

    .source-box {
        padding: 1rem;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.7rem;
    }

    .small-text {
        color: #64748b;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "filename" not in st.session_state:
    st.session_state.filename = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📋 HR Policy Assistant</h1>
        <p>
            Upload an HR policy PDF and ask questions.
            The assistant retrieves relevant policy sections
            before generating an answer with AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GET GROQ API KEY
# ============================================================

def get_groq_api_key():
    """
    Get Groq API key from Streamlit Secrets.

    Streamlit Cloud:
        Settings → Secrets

    Expected secret:
        GROQ_API_KEY = "your-key"
    """

    try:
        api_key = st.secrets.get("GROQ_API_KEY")

        if api_key:
            return api_key

    except Exception:
        pass

    # Optional local/environment fallback
    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load Sentence Transformer model once.
    Streamlit caches it between reruns.
    """

    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

def extract_pdf_pages(uploaded_file):
    """
    Extract text page-by-page from a PDF using PyMuPDF.
    """

    pdf_bytes = uploaded_file.getvalue()

    try:
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise ValueError(
            f"Could not open the PDF: {exc}"
        )

    pages = []

    try:

        for page_number in range(len(document)):

            page = document.load_page(page_number)

            text = page.get_text("text")

            if text and text.strip():

                pages.append(
                    {
                        "page": page_number + 1,
                        "text": text.strip(),
                    }
                )

    finally:
        document.close()

    return pages


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    """
    Clean common PDF extraction artifacts.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")

    cleaned_lines = []

    for line in text.split("\n"):

        line = " ".join(line.split())

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ============================================================
# CREATE TEXT CHUNKS
# ============================================================

def create_chunks(pages):
    """
    Create overlapping chunks while preserving page numbers.
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        text = clean_text(
            page_data["text"]
        )

        if not text:
            continue

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(
                start + CHUNK_SIZE,
                text_length,
            )

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                    }
                )

            if end >= text_length:
                break

            start = max(
                end - CHUNK_OVERLAP,
                start + 1,
            )

    return chunks


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(chunks, embedding_model):
    """
    Generate embeddings and create FAISS similarity index.
    """

    if not chunks:
        raise ValueError(
            "No text chunks were created from the document."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding generation returned an invalid shape."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    embedding_model,
    top_k,
):
    """
    Semantic search using FAISS.
    """

    if index is None:
        return []

    if not chunks:
        return []

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32",
    )

    actual_k = min(
        top_k,
        len(chunks),
    )

    scores, indices = index.search(
        question_embedding,
        actual_k,
    )

    retrieved = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position < 0:
            continue

        chunk = chunks[index_position]

        retrieved.append(
            {
                "text": chunk["text"],
                "page": chunk["page"],
                "score": float(score),
            }
        )

    return retrieved


# ============================================================
# GENERATE GROQ ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
):
    """
    Generate an answer using only retrieved policy context.
    """

    if not GROQ_API_KEY:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add it to Streamlit Cloud Secrets."
        )

    if not retrieved_chunks:

        return (
            "I could not find relevant information "
            "in the uploaded HR policy."
        )

    client = Groq(
        api_key=GROQ_API_KEY
    )

    context_blocks = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        context_blocks.append(
            f"""
SOURCE {number}
POLICY PAGE: {chunk["page"]}

{chunk["text"]}
"""
        )

    context = "\n".join(
        context_blocks
    )

    system_prompt = """
You are an HR Policy Assistant.

Your task is to answer questions about an uploaded HR policy.

STRICT RAG RULES:

1. Use ONLY the provided policy context.
2. Do not invent company policies.
3. Do not assume information that is not present.
4. If the answer cannot be found in the supplied context,
   clearly say that the uploaded policy does not provide
   enough information.
5. Do not make up leave amounts, eligibility requirements,
   procedures, penalties, salaries, benefits, or deadlines.
6. Do not provide legal conclusions unless the policy itself
   explicitly contains the relevant information.
7. Give a clear and concise answer.
8. When possible, mention the relevant policy page.
9. If several policy sections are relevant, combine them carefully.
10. If the user's question is unrelated to the uploaded policy,
    explain that the assistant is intended for questions about
    the uploaded HR policy.
"""

    user_prompt = f"""
UPLOADED HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Answer the question using only the policy context above.
"""

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
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
            max_tokens=1200,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Groq API error: {exc}"
        )

    if not response.choices:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "Groq returned empty answer content."
        )

    return answer.strip()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.markdown(
        f"""
        **AI Model**

        `{GROQ_MODEL}`
        """
    )

    st.markdown(
        f"""
        **Embedding Model**

        `{EMBEDDING_MODEL}`
        """
    )

    top_k = st.slider(
        "Retrieved policy sections",
        min_value=2,
        max_value=8,
        value=DEFAULT_TOP_K,
        help=(
            "Number of policy chunks retrieved "
            "before sending context to Groq."
        ),
    )

    st.divider()

    st.markdown(
        """
        ### How it works

        **1.** Upload HR policy

        **2.** Extract PDF text

        **3.** Split into chunks

        **4.** Create embeddings

        **5.** Search with FAISS

        **6.** Retrieve relevant sections

        **7.** Send context to Groq

        **8.** Generate grounded answer
        """
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

with st.spinner(
    "Loading AI embedding model..."
):

    try:

        embedding_model = (
            load_embedding_model()
        )

    except Exception as exc:

        st.error(
            "Could not load the Sentence Transformer model."
        )

        st.exception(exc)

        st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("📄 Upload HR Policy")

uploaded_file = st.file_uploader(
    "Choose an HR policy PDF",
    type=["pdf"],
    help=(
        "Upload a searchable/text-based HR policy PDF."
    ),
)


# ============================================================
# PROCESS UPLOADED DOCUMENT
# ============================================================

if uploaded_file:

    document_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    # Process only if this is a new document
    if (
        st.session_state.document_hash
        != document_hash
    ):

        with st.spinner(
            "Reading and indexing HR policy..."
        ):

            try:

                pages = extract_pdf_pages(
                    uploaded_file
                )

                if not pages:

                    st.error(
                        "No readable text was found in this PDF."
                    )

                    st.info(
                        "Please upload a text-based/searchable PDF. "
                        "Scanned image-only PDFs require OCR."
                    )

                    st.stop()

                chunks = create_chunks(
                    pages
                )

                if not chunks:

                    st.error(
                        "The document did not produce usable text chunks."
                    )

                    st.stop()

                faiss_index = build_faiss_index(
                    chunks,
                    embedding_model,
                )

                # Save processed document
                st.session_state.document_hash = (
                    document_hash
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.faiss_index = (
                    faiss_index
                )

                st.session_state.filename = (
                    uploaded_file.name
                )

                st.session_state.page_count = (
                    len(pages)
                )

            except Exception as exc:

                st.error(
                    "There was a problem processing the PDF."
                )

                st.exception(exc)

                st.stop()

        st.success(
            f"✅ {uploaded_file.name} processed successfully."
        )

    else:

        chunks = st.session_state.chunks

        faiss_index = (
            st.session_state.faiss_index
        )

    # ========================================================
    # DOCUMENT METRICS
    # ========================================================

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Pages",
            st.session_state.page_count,
        )

    with col2:

        st.metric(
            "🧩 Text Chunks",
            len(st.session_state.chunks),
        )

    with col3:

        st.metric(
            "🔢 Vector Dimension",
            st.session_state.faiss_index.d,
        )

    # ========================================================
    # QUESTION AREA
    # ========================================================

    st.divider()

    st.subheader(
        "💬 Ask Your HR Policy"
    )

    example_questions = [
        "What is the annual leave policy?",
        "How many sick leaves are employees entitled to?",
        "What is the probation period?",
        "What is the maternity leave policy?",
        "What is the attendance policy?",
        "What is the resignation notice period?",
    ]

    selected_question = st.selectbox(
        "Choose an example or write your own question",
        ["Write my own question"]
        + example_questions,
    )

    if selected_question == "Write my own question":

        question = st.text_input(
            "Your question",
            placeholder=(
                "Example: How many annual leaves "
                "can an employee take?"
            ),
        )

    else:

        question = selected_question

    ask_button = st.button(
        "🔎 Ask HR Policy Assistant",
        type="primary",
        use_container_width=True,
    )

    # ========================================================
    # ANSWER
    # ========================================================

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            with st.spinner(
                "Searching the HR policy..."
            ):

                try:

                    retrieved_chunks = (
                        retrieve_relevant_chunks(
                            question=question,
                            index=faiss_index,
                            chunks=chunks,
                            embedding_model=embedding_model,
                            top_k=top_k,
                        )
                    )

                except Exception as exc:

                    st.error(
                        "The policy search failed."
                    )

                    st.exception(exc)

                    st.stop()

            # ------------------------------------------------
            # GENERATION
            # ------------------------------------------------

            with st.spinner(
                "Generating answer with GPT-OSS 120B..."
            ):

                try:

                    answer = generate_answer(
                        question,
                        retrieved_chunks,
                    )

                except Exception as exc:

                    st.error(
                        "The AI response could not be generated."
                    )

                    st.exception(exc)

                    st.stop()

            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            st.markdown(
                '<div class="answer-box">',
                unsafe_allow_html=True,
            )

            st.subheader(
                "🤖 Answer"
            )

            st.markdown(answer)

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            st.subheader(
                "📚 Retrieved Policy Sources"
            )

            for number, source in enumerate(
                retrieved_chunks,
                start=1,
            ):

                with st.expander(
                    f"Source {number} • "
                    f"Page {source['page']} • "
                    f"Similarity {source['score']:.3f}"
                ):

                    st.write(
                        source["text"]
                    )

else:

    st.info(
        "👆 Upload an HR policy PDF above "
        "to start asking questions."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HR Policy Assistant • "
    "RAG + FAISS + Sentence Transformers + "
    "PyMuPDF + Groq GPT-OSS 120B"
)
