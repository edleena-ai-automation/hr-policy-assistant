# 📋 HR Policy Assistant

An AI-powered HR Policy Assistant built with Retrieval-Augmented Generation (RAG).

Users can upload an HR policy PDF and ask questions about the policy.

The application retrieves relevant sections from the uploaded document before sending the context to Groq for answer generation.

## 🚀 Features

- Upload HR policy PDF
- Extract PDF text using PyMuPDF
- Split documents into overlapping chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS
- Semantic search over HR policy
- Generate grounded answers using Groq
- Display retrieved policy sources
- Show source page numbers
- Adjustable number of retrieved chunks
- Selectable Groq model
- Streamlit web interface

## 🧠 RAG Pipeline

PDF
↓
PyMuPDF
↓
Text Extraction
↓
Text Chunking
↓
Sentence Transformers
↓
Embeddings
↓
FAISS
↓
Semantic Retrieval
↓
Relevant Policy Context
↓
Groq
↓
HR Answer

## 🛠️ Technologies

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- NumPy
- Groq API

## 🔐 API Key

The Groq API key should NOT be stored in this repository.

For Streamlit Community Cloud, add the following secret:

GROQ_API_KEY = "your_groq_api_key"

Use Streamlit's Secrets management.

## ▶️ Deployment

This application can be deployed directly from GitHub using Streamlit Community Cloud.

Required files:

- app.py
- requirements.txt
- README.md
- .gitignore

## ⚠️ Notes

This application is designed to answer questions based on the uploaded HR policy.

If the policy does not contain enough information, the assistant should state that instead of inventing a policy.

For production HR systems, consider adding authentication, document access controls, encryption, audit logging, and organization-specific privacy controls.