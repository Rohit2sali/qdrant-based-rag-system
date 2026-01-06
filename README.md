# RAG-based PDF Question Answering System (Qdrant + LLM)

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, index their contents into a vector database (Qdrant), and ask natural language questions grounded strictly in the uploaded documents.  
The application is exposed through an interactive Streamlit interface.

---

## 🚀 Features

- Upload one or multiple PDF documents
- Automatic document preprocessing and chunking
- Dense + sparse embeddings generation
- Vector storage and indexing using Qdrant
- Semantic retrieval of relevant chunks
- Context-aware question answering using an LLM
- Simple and interactive UI built with Streamlit

---

## 🧠 System Architecture

1. **Document Ingestion**
   - User uploads PDF files via Streamlit UI
   - PDFs are parsed and converted into raw text

2. **Preprocessing**
   - Text cleaning and normalization
   - Chunking with overlap to preserve context

3. **Embedding & Indexing**
   - Dense embeddings generated using a sentence-transformer model
   - Optional sparse embeddings (BM25-style) for hybrid retrieval
   - Chunks are stored and indexed in Qdrant

4. **Retrieval**
   - User query is embedded
   - Relevant document chunks are retrieved from Qdrant
   - Optional re-ranking using a cross-encoder

5. **Generation**
   - Retrieved context is passed to the LLM
   - LLM generates an answer grounded in the retrieved documents

---

## 🛠 Tech Stack

- **Frontend / UI**: Streamlit  
- **Vector Database**: Qdrant  
- **Embeddings**: Sentence-Transformers  
- **LLM**: Open-source / API-based LLM (configurable)  
- **Language**: Python  

---

## 📂 Project Structure
.
├── launch.py # Streamlit application entry point
├── doc_processing.py/ # PDF preprocessing
├── embeddings/ # Embedding generation logic
├── retrieval/ # Qdrant indexing and retrieval
├── llm/ # LLM interaction and prompting
├── requirements.txt
└── README.md


## ⚙️ Setup Instructions

### 1. Clone the repository
git clone https://github.com/Rohit2sali/rag-qdrant-pdf-chat.git
cd rag-qdrant-pdf-chat

2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Start Qdrant
Using Docker:
docker run -p 6333:6333 qdrant/qdrant

▶️ Run the Application
streamlit run launch.py
